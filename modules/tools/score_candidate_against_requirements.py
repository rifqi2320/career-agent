from __future__ import annotations

from enum import StrEnum
from time import perf_counter

from google.adk.tools import ToolContext
from jinja2 import Template
from pydantic import BaseModel, Field
from safe_result import safe, safe_async

from models.llm import LlmConfig
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError
from modules.logging import logging
from modules.tools.wrapper import wrap_safe_tool
from modules.utils import generate_structured_output

SCORE_CANDIDATE_SYSTEM_PROMPT = """
You are a strict scoring engine.

Given a candidate profile and job requirements, return JSON only that matches this schema:
- overall_score: integer 0-100
- dimension_scores: {skills: int, experience: int, seniority_fit: int}
- matched_skills: list[str]
- gap_skills: list[str]
- confidence: "low" | "medium" | "high"

Scoring rules:
- skills score should reflect required-skill match quality.
- experience score should reflect years/scope fit for the required seniority.
- seniority_fit should reflect level alignment.
- overall_score should be a balanced rollup of dimensions.

Confidence rules:
- Derive confidence from observable signals only: JD completeness, required-skill match ratio, and domain/seniority clarity.
- Use low confidence when key requirement data is sparse or ambiguous.

Constraints:
- Do not invent skills that are absent from both inputs.
- matched_skills and gap_skills must be deduplicated and concise.
- If inputs are incomplete, still return a valid object with conservative scoring.
""".strip()

SCORE_CANDIDATE_USER_PROMPT_TEMPLATE = Template(
    """
Candidate profile (JSON):
{{ candidate_profile }}

Requirements (JSON):
{{ requirements }}
""".strip()
)


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> ConfidenceLevel:
        return cls.UNKNOWN


class CandidateProfileInputSchema(BaseModel):
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = Field(default=None, ge=0)
    seniority_level: str = "unknown"
    domain: str = "unknown"


class RequirementsInputSchema(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    seniority_level: str = "unknown"
    domain: str = "unknown"
    responsibilities: list[str] = Field(default_factory=list)


class ScoreDimensionsSchema(BaseModel):
    skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    seniority_fit: int = Field(ge=0, le=100)


class ScoreCandidateOutputSchema(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    dimension_scores: ScoreDimensionsSchema
    matched_skills: list[str] = Field(default_factory=list)
    gap_skills: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


@safe
def _parse_requirements_from_state(raw_requirements: object) -> RequirementsInputSchema:
    return RequirementsInputSchema.model_validate(raw_requirements)


@safe_async
async def _score_candidate_against_requirements(
    candidate_profile: CandidateProfileInputSchema,
    requirements: RequirementsInputSchema | None = None,
    *,
    context: ToolContext,
) -> ScoreCandidateOutputSchema:
    """Score candidate fit using LLM reasoning and persist the latest score."""
    started_at = perf_counter()
    logging.info(
        "score_candidate_against_requirements started | candidate_skills=%d has_requirements_arg=%s",
        len(candidate_profile.skills),
        requirements is not None,
    )
    resolved_requirements = requirements
    if resolved_requirements is None:
        logging.info(
            "score_candidate_against_requirements loading requirements from context.state['last_requirements']"
        )
        raw_requirements = context.state.get("last_requirements")
        if raw_requirements is None:
            raise ValueError(
                "Missing requirements input and `context.state['last_requirements']`."
            )
        parse_result = _parse_requirements_from_state(raw_requirements)
        if parse_result.is_err():
            raise ValueError(
                "Invalid requirements in `context.state['last_requirements']`."
            ) from parse_result.error
        resolved_requirements = parse_result.value
        if resolved_requirements is None:
            raise ValueError(
                "Invalid requirements in `context.state['last_requirements']`."
            )
        logging.info(
            "score_candidate_against_requirements loaded requirements from state | required_skills=%d responsibilities=%d",
            len(resolved_requirements.required_skills),
            len(resolved_requirements.responsibilities),
        )

    if resolved_requirements is None:
        raise ValueError("Requirements could not be resolved.")

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    logging.info(
        "score_candidate_against_requirements LLM scoring started | model=%s",
        llm_config.model_name,
    )
    result = await _score_candidate_against_requirements_llm(
        candidate_profile,
        resolved_requirements,
        llm_config,
    )
    if result.is_ok() and result.value is not None:
        context.state["last_score"] = result.value.model_dump()
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logging.info(
            "score_candidate_against_requirements success | overall_score=%d confidence=%s matched=%d gaps=%d elapsed_ms=%d",
            result.value.overall_score,
            result.value.confidence,
            len(result.value.matched_skills),
            len(result.value.gap_skills),
            elapsed_ms,
        )
        return result.value

    if isinstance(result.error, RetryableModelOutputError):
        logging.warning(
            "score_candidate_against_requirements retryable model output failure | error=%s",
            result.error,
        )
        raise result.error

    error_message = f"Scoring failed: {result.error}"
    logging.error(error_message)
    raise RuntimeError(error_message)


@safe_async
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


score_candidate_against_requirements = wrap_safe_tool(
    _score_candidate_against_requirements
)
