from __future__ import annotations

from time import perf_counter
from typing import cast

from google.adk.tools import ToolContext
from pydantic import BaseModel, Field, model_validator

from models.llm import LlmConfig
from models.confidence import ConfidenceLevel, ConfidenceMetrics, calibrate_confidence
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import (
    RetryableModelOutputError,
    ToolInputError,
    ValidationError,
)
from modules.logging import logging
from modules.tools.prompts import (
    PRIORITISE_SKILL_GAPS_SYSTEM_PROMPT,
    PRIORITISE_SKILL_GAPS_USER_PROMPT_TEMPLATE,
)
from modules.utils import generate_structured_output
from modules.utils.trace import increment_llm_calls


class PrioritizedSkillSchema(BaseModel):
    skill: str = Field(min_length=1)
    priority_rank: int = Field(ge=1)
    estimated_match_gain_pct: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1)


class PrioritizeSkillGapsOutputSchema(BaseModel):
    prioritized_skills: list[PrioritizedSkillSchema] = Field(default_factory=list)
    confidence_score: int = Field(default=0, ge=0, le=100)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN

    @model_validator(mode="after")
    def validate_rank_integrity(self) -> PrioritizeSkillGapsOutputSchema:
        if not self.prioritized_skills:
            raise ValidationError("prioritized_skills must not be empty.")

        normalized_skills = {
            skill_item.skill.strip().casefold()
            for skill_item in self.prioritized_skills
        }
        if len(normalized_skills) != len(self.prioritized_skills):
            raise ValidationError("prioritized_skills contains duplicate skills.")

        ranks = [skill_item.priority_rank for skill_item in self.prioritized_skills]
        if len(set(ranks)) != len(ranks):
            raise ValidationError("priority_rank values must be unique.")

        expected_ranks = list(range(1, len(self.prioritized_skills) + 1))
        if sorted(ranks) != expected_ranks:
            raise ValidationError("priority_rank values must be consecutive from 1..N.")
        return self


def _parse_gap_skills_from_state(raw_last_score: object) -> list[str]:
    if not isinstance(raw_last_score, dict):
        raise ToolInputError("`context.state['last_score']` must be a JSON object.")
    last_score = cast("dict[str, object]", raw_last_score)
    raw_gap_skills = last_score.get("gap_skills")
    if not isinstance(raw_gap_skills, list):
        raise ToolInputError("`context.state['last_score'].gap_skills` must be a list.")

    gap_skills: list[str] = []
    for raw_skill in raw_gap_skills:
        if not isinstance(raw_skill, str):
            raise ToolInputError("All `gap_skills` values must be strings.")
        normalized_skill = raw_skill.strip()
        if normalized_skill:
            gap_skills.append(normalized_skill)
    return gap_skills


def _deduplicate_gap_skills(gap_skills: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_skills: list[str] = []
    for skill in gap_skills:
        normalized = skill.strip()
        if not normalized:
            continue
        normalized_key = normalized.casefold()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        unique_skills.append(normalized)
    return unique_skills


def _calibrate_prioritization_confidence(
    resolved_skills: list[str],
    result: PrioritizeSkillGapsOutputSchema,
) -> ConfidenceMetrics:
    """Compute confidence from rank separation and match-gain signal."""
    if not result.prioritized_skills:
        return ConfidenceMetrics(confidence_score=0, confidence=ConfidenceLevel.LOW)

    avg_gain = sum(
        skill.estimated_match_gain_pct for skill in result.prioritized_skills
    ) / len(result.prioritized_skills)
    completeness_penalty = 0
    if len(result.prioritized_skills) < len(resolved_skills):
        completeness_penalty += (
            len(resolved_skills) - len(result.prioritized_skills)
        ) * 6

    return calibrate_confidence(avg_gain, penalties=completeness_penalty)


async def prioritise_skill_gaps(
    gap_skills: list[str] | None = None,
    job_market_context: str = "unknown",
    *,
    context: ToolContext,
) -> PrioritizeSkillGapsOutputSchema:
    """Prioritize skill gaps by expected match gain for a target market context."""
    started_at = perf_counter()
    logging.info(
        "prioritise_skill_gaps started | has_gap_skills_arg=%s job_market_context=%s",
        gap_skills is not None,
        job_market_context,
    )

    resolved_gap_skills = gap_skills
    if resolved_gap_skills is None:
        logging.info(
            "prioritise_skill_gaps loading skills from context.state['last_score'].gap_skills"
        )
        raw_last_score = context.state.get("last_score")
        if raw_last_score is None:
            raise ToolInputError(
                "Missing `gap_skills` argument and `context.state['last_score']`."
            )
        try:
            resolved_gap_skills = _parse_gap_skills_from_state(raw_last_score)
        except ToolInputError as error:
            raise ToolInputError(
                "Invalid gap skills in `context.state['last_score']`."
            ) from error

    if resolved_gap_skills is None:
        raise ToolInputError("Gap skills could not be resolved.")

    unique_gap_skills = _deduplicate_gap_skills(resolved_gap_skills)
    if not unique_gap_skills:
        raise ToolInputError("gap_skills must include at least one non-empty skill.")

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    increment_llm_calls(context)
    try:
        prioritized = await _prioritise_skill_gaps_llm(
            gap_skills=unique_gap_skills,
            job_market_context=job_market_context,
            llm_config=llm_config,
        )
    except RetryableModelOutputError:
        logging.warning("prioritise_skill_gaps retryable model output failure")
        raise

    expected_skill_keys = {skill.casefold() for skill in unique_gap_skills}
    output_skill_keys = {
        item.skill.strip().casefold() for item in prioritized.prioritized_skills
    }
    if output_skill_keys != expected_skill_keys:
        raise RetryableModelOutputError(
            "Model prioritized skills did not match input gap skills; retry may succeed."
        )

    confidence = _calibrate_prioritization_confidence(unique_gap_skills, prioritized)
    result_payload = prioritized.model_copy(
        update={
            "confidence_score": confidence.confidence_score,
            "confidence": confidence.confidence,
        }
    )
    context.state["last_prioritized_skill_gaps"] = result_payload.model_dump()
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logging.info(
        "prioritise_skill_gaps success | gap_skills=%d confidence=%s confidence_score=%d elapsed_ms=%d",
        len(result_payload.prioritized_skills),
        result_payload.confidence,
        result_payload.confidence_score,
        elapsed_ms,
    )
    return result_payload


async def _prioritise_skill_gaps_llm(
    *,
    gap_skills: list[str],
    job_market_context: str,
    llm_config: LlmConfig,
) -> PrioritizeSkillGapsOutputSchema:
    user_prompt = PRIORITISE_SKILL_GAPS_USER_PROMPT_TEMPLATE.render(
        gap_skills=gap_skills,
        job_market_context=job_market_context,
    )
    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=PRIORITISE_SKILL_GAPS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=PrioritizeSkillGapsOutputSchema,
    )
    return result.unwrap()
