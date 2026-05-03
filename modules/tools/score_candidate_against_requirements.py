from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter

from google.adk.tools import ToolContext

from models.llm import LlmConfig
from models.confidence import ConfidenceMetrics, calibrate_confidence
from modules.candidates.schemas import CandidateProfileInputSchema
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError, ToolInputError
from modules.logging import logging
from modules.matches.schemas import (
    RequirementsInputSchema,
    ScoreCandidateOutputSchema,
    ScoreDimensionsSchema,
)
from modules.tools.prompts import (
    SCORE_CANDIDATE_SYSTEM_PROMPT,
    SCORE_CANDIDATE_USER_PROMPT_TEMPLATE,
)
from modules.utils import generate_structured_output
from modules.utils.trace import increment_llm_calls

__all__ = [
    "CandidateProfileInputSchema",
    "RequirementsInputSchema",
    "ScoreCandidateOutputSchema",
    "ScoreDimensionsSchema",
    "score_candidate_against_requirements",
]


def _calibrate_candidate_confidence(
    result: ScoreCandidateOutputSchema,
    requirements: RequirementsInputSchema,
    candidate_profile: CandidateProfileInputSchema,
) -> ConfidenceMetrics:
    """Deterministically calibrate confidence from score and input signal quality."""
    penalty = 0
    if not requirements.required_skills:
        penalty += 25
    if (
        not requirements.seniority_level
        or requirements.seniority_level.lower() == "unknown"
    ):
        penalty += 8
    if (
        not candidate_profile.seniority_level
        or candidate_profile.seniority_level.lower() == "unknown"
    ):
        penalty += 8
    if not requirements.domain or requirements.domain.lower() == "unknown":
        penalty += 5
    if not candidate_profile.domain or candidate_profile.domain.lower() == "unknown":
        penalty += 5
    if not candidate_profile.skills:
        penalty += 10

    return calibrate_confidence(result.overall_score, penalties=penalty)


def _parse_requirements_from_state(raw_requirements: object) -> RequirementsInputSchema:
    return RequirementsInputSchema.model_validate(raw_requirements)


def _parse_candidate_profile(
    raw_candidate_profile: object,
) -> CandidateProfileInputSchema:
    """Validate a tool-facing candidate profile payload."""
    if isinstance(raw_candidate_profile, CandidateProfileInputSchema):
        return raw_candidate_profile
    if isinstance(raw_candidate_profile, str):
        try:
            raw_candidate_profile = json.loads(raw_candidate_profile)
        except json.JSONDecodeError as error:
            raise ToolInputError("candidate_profile must be valid JSON.") from error
    if not isinstance(raw_candidate_profile, Mapping):
        raise ToolInputError("candidate_profile must be a JSON object string.")
    try:
        return CandidateProfileInputSchema.model_validate(raw_candidate_profile)
    except ValueError as error:
        raise ToolInputError("candidate_profile is invalid.") from error


def _parse_requirements_argument(raw_requirements: object) -> RequirementsInputSchema:
    """Validate a tool-facing requirements payload."""
    if isinstance(raw_requirements, RequirementsInputSchema):
        return raw_requirements
    if isinstance(raw_requirements, str):
        try:
            raw_requirements = json.loads(raw_requirements)
        except json.JSONDecodeError as error:
            raise ToolInputError("requirements must be valid JSON.") from error
    if not isinstance(raw_requirements, Mapping):
        raise ToolInputError("requirements must be a JSON object string when provided.")
    try:
        return RequirementsInputSchema.model_validate(raw_requirements)
    except ValueError as error:
        raise ToolInputError("requirements is invalid.") from error


async def score_candidate_against_requirements(
    candidate_profile: str,
    requirements: str | None = None,
    *,
    context: ToolContext,
) -> ScoreCandidateOutputSchema:
    """Score candidate fit from JSON strings and persist the latest score.

    Candidate JSON keys: skills, years_experience, seniority_level, domain.
    Requirements JSON keys: required_skills, seniority_level, domain, responsibilities.
    """
    started_at = perf_counter()
    resolved_candidate_profile = _parse_candidate_profile(candidate_profile)
    logging.info(
        "score_candidate_against_requirements started | candidate_skills=%d has_requirements_arg=%s",
        len(resolved_candidate_profile.skills),
        requirements is not None,
    )
    resolved_requirements = (
        _parse_requirements_argument(requirements) if requirements is not None else None
    )
    if resolved_requirements is None:
        logging.info(
            "score_candidate_against_requirements loading requirements from context.state['last_requirements']"
        )
        raw_requirements = context.state.get("last_requirements")
        if raw_requirements is None:
            raise ToolInputError(
                "Missing requirements input and `context.state['last_requirements']`."
            )
        try:
            resolved_requirements = _parse_requirements_from_state(raw_requirements)
        except ValueError as error:
            raise ToolInputError(
                "Invalid requirements in `context.state['last_requirements']`."
            ) from error
        logging.info(
            "score_candidate_against_requirements loaded requirements from state | required_skills=%d responsibilities=%d",
            len(resolved_requirements.required_skills),
            len(resolved_requirements.responsibilities),
        )

    if resolved_requirements is None:
        raise ToolInputError("Requirements could not be resolved.")

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    logging.info(
        "score_candidate_against_requirements LLM scoring started | model=%s",
        llm_config.model_name,
    )
    increment_llm_calls(context)
    try:
        score = await _score_candidate_against_requirements_llm(
            resolved_candidate_profile,
            resolved_requirements,
            llm_config,
        )
    except RetryableModelOutputError:
        logging.warning(
            "score_candidate_against_requirements retryable model output failure"
        )
        raise

    confidence = _calibrate_candidate_confidence(
        result=score,
        requirements=resolved_requirements,
        candidate_profile=resolved_candidate_profile,
    )
    result_payload = score.model_copy(
        update={
            "confidence_score": confidence.confidence_score,
            "confidence": confidence.confidence,
        }
    )
    context.state["last_score"] = result_payload.model_dump()
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logging.info(
        "score_candidate_against_requirements success | overall_score=%d confidence=%s confidence_score=%d matched=%d gaps=%d elapsed_ms=%d",
        result_payload.overall_score,
        result_payload.confidence,
        result_payload.confidence_score,
        len(result_payload.matched_skills),
        len(result_payload.gap_skills),
        elapsed_ms,
    )
    return result_payload


async def _score_candidate_against_requirements_llm(
    candidate_profile: CandidateProfileInputSchema,
    requirements: RequirementsInputSchema,
    llm_config: LlmConfig,
) -> ScoreCandidateOutputSchema:
    """Use an LLM to score candidate fit against parsed requirements."""
    user_prompt = SCORE_CANDIDATE_USER_PROMPT_TEMPLATE.render(
        candidate_profile=candidate_profile.model_dump_json(indent=2),
        requirements=requirements.model_dump_json(indent=2),
    )

    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=SCORE_CANDIDATE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=ScoreCandidateOutputSchema,
    )
    return result.unwrap()
