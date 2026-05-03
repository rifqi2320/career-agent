from __future__ import annotations

from time import perf_counter
from typing import cast

from google.adk.tools import ToolContext
from jinja2 import Template
from pydantic import BaseModel, Field, model_validator
from safe_result import safe, safe_async

from models.llm import LlmConfig
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError
from modules.logging import logging
from modules.tools.wrapper import wrap_safe_tool
from modules.utils import generate_structured_output

PRIORITISE_SKILL_GAPS_SYSTEM_PROMPT = """
You are a career skill-gap prioritization engine.

Given:
- a list of gap skills,
- a short job-market context string,
- optional requirement/score context,

return strict JSON that matches this schema:
- prioritized_skills: list items with
  - skill: string
  - priority_rank: int (1..N, no ties)
  - estimated_match_gain_pct: int (0..100)
  - rationale: string

Rules:
- Return exactly one item per provided unique gap skill.
- Keep skills semantically aligned with input names; do not invent new gaps.
- Rank by expected impact on employability for the target role and market context.
- estimated_match_gain_pct is the expected contribution to closing the match gap.
- Rationale must be concise and practical.
- Output JSON only.
""".strip()

PRIORITISE_SKILL_GAPS_USER_PROMPT_TEMPLATE = Template(
    """
Gap skills:
{{ gap_skills }}

Job market context:
{{ job_market_context }}

Last requirements (optional JSON):
{{ last_requirements }}

Last score (optional JSON):
{{ last_score }}
""".strip()
)


class PrioritizedSkillSchema(BaseModel):
    skill: str = Field(min_length=1)
    priority_rank: int = Field(ge=1)
    estimated_match_gain_pct: int = Field(ge=0, le=100)
    rationale: str = Field(min_length=1)


class PrioritizeSkillGapsOutputSchema(BaseModel):
    prioritized_skills: list[PrioritizedSkillSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_rank_integrity(self) -> PrioritizeSkillGapsOutputSchema:
        if not self.prioritized_skills:
            raise ValueError("prioritized_skills must not be empty.")

        normalized_skills = {
            skill_item.skill.strip().casefold() for skill_item in self.prioritized_skills
        }
        if len(normalized_skills) != len(self.prioritized_skills):
            raise ValueError("prioritized_skills contains duplicate skills.")

        ranks = [skill_item.priority_rank for skill_item in self.prioritized_skills]
        if len(set(ranks)) != len(ranks):
            raise ValueError("priority_rank values must be unique.")

        expected_ranks = list(range(1, len(self.prioritized_skills) + 1))
        if sorted(ranks) != expected_ranks:
            raise ValueError("priority_rank values must be consecutive from 1..N.")
        return self


@safe
def _parse_gap_skills_from_state(raw_last_score: object) -> list[str]:
    if not isinstance(raw_last_score, dict):
        raise ValueError("`context.state['last_score']` must be a JSON object.")
    last_score = cast("dict[str, object]", raw_last_score)
    raw_gap_skills = last_score.get("gap_skills")
    if not isinstance(raw_gap_skills, list):
        raise ValueError("`context.state['last_score'].gap_skills` must be a list.")

    gap_skills: list[str] = []
    for raw_skill in raw_gap_skills:
        if not isinstance(raw_skill, str):
            raise ValueError("All `gap_skills` values must be strings.")
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


@safe_async
async def _prioritise_skill_gaps(
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
            raise ValueError(
                "Missing `gap_skills` argument and `context.state['last_score']`."
            )
        parse_result = _parse_gap_skills_from_state(raw_last_score)
        if parse_result.is_err():
            raise ValueError(
                "Invalid gap skills in `context.state['last_score']`."
            ) from parse_result.error
        resolved_gap_skills = parse_result.value
        if resolved_gap_skills is None:
            raise ValueError(
                "Invalid gap skills in `context.state['last_score']`."
            )

    if resolved_gap_skills is None:
        raise ValueError("Gap skills could not be resolved.")

    unique_gap_skills = _deduplicate_gap_skills(resolved_gap_skills)
    if not unique_gap_skills:
        raise ValueError("gap_skills must include at least one non-empty skill.")

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    result = await _prioritise_skill_gaps_llm(
        gap_skills=unique_gap_skills,
        job_market_context=job_market_context,
        last_requirements=context.state.get("last_requirements"),
        last_score=context.state.get("last_score"),
        llm_config=llm_config,
    )

    if result.is_ok() and result.value is not None:
        expected_skill_keys = {skill.casefold() for skill in unique_gap_skills}
        output_skill_keys = {
            item.skill.strip().casefold() for item in result.value.prioritized_skills
        }
        if output_skill_keys != expected_skill_keys:
            raise RetryableModelOutputError(
                "Model prioritized skills did not match input gap skills; retry may succeed."
            )

        context.state["last_prioritized_skill_gaps"] = result.value.model_dump()
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logging.info(
            "prioritise_skill_gaps success | gap_skills=%d elapsed_ms=%d",
            len(result.value.prioritized_skills),
            elapsed_ms,
        )
        return result.value

    if isinstance(result.error, RetryableModelOutputError):
        logging.warning(
            "prioritise_skill_gaps retryable model output failure | error=%s",
            result.error,
        )
        raise result.error

    raise RuntimeError(f"LLM prioritization failed: {result.error}")


@safe_async
async def _prioritise_skill_gaps_llm(
    *,
    gap_skills: list[str],
    job_market_context: str,
    last_requirements: object,
    last_score: object,
    llm_config: LlmConfig,
) -> PrioritizeSkillGapsOutputSchema:
    user_prompt = PRIORITISE_SKILL_GAPS_USER_PROMPT_TEMPLATE.render(
        gap_skills=gap_skills,
        job_market_context=job_market_context,
        last_requirements=last_requirements,
        last_score=last_score,
    )
    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=PRIORITISE_SKILL_GAPS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=PrioritizeSkillGapsOutputSchema,
    )
    return result.unwrap()


prioritise_skill_gaps = wrap_safe_tool(_prioritise_skill_gaps)
