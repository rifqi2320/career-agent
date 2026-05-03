from __future__ import annotations

from time import perf_counter

from google.adk.tools import ToolContext
from jinja2 import Template
from pydantic import BaseModel, Field

from models.llm import LlmConfig
from models.confidence import ConfidenceLevel, ConfidenceMetrics, calibrate_confidence
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError, ToolInputError
from modules.logging import logging
from modules.utils import generate_structured_output
from modules.utils.trace import increment_llm_calls

SCORE_CANDIDATE_SYSTEM_PROMPT = """
You are a career-fit scoring engine.

Return only a JSON object, with no markdown, prose, or code fences.
The object must match this exact schema:
- overall_score: integer 0-100
- dimension_scores: object with keys skills, experience, seniority_fit; each integer 0-100
- matched_skills: list[str]
- gap_skills: list[str]
- confidence: "low" | "medium" | "high"

Scoring guidance:
- Compare skills semantically, not only by exact string match.
- Treat common equivalents as related, for example:
  - "api", "api design", and "apis"
  - "prompt design" and "prompt engineering"
  - "llm", "llms", and "llm applications"
  - "rag", "rag architectures", and "retrieval augmented generation"
  - "function calling", "function/tool calling", and "tool calling"
- `matched_skills` must contain requirement skill names that are reasonably evidenced by the candidate profile.
- `gap_skills` must contain requirement skill names that are not reasonably evidenced.
- Do not invent candidate skills or job requirements.
- Score `skills` from matched required skills and strength of evidence.
- Score `experience` from years_experience and relevance of work history to the role.
- Score `seniority_fit` from seniority alignment, scope, ownership, and role expectations.
- Compute `overall_score` from the three dimensions with strongest weight on skills fit.
- Use confidence "low" when input evidence is thin or ambiguous, "medium" for adequate evidence, and "high" for strong structured evidence.
- Sort `matched_skills` and `gap_skills` alphabetically.
""".strip()

SCORE_CANDIDATE_USER_PROMPT_TEMPLATE = Template(
    """
Candidate profile (JSON):
{{ candidate_profile }}

Requirements (JSON):
{{ requirements }}
""".strip()
)


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
    confidence_score: int = Field(default=0, ge=0, le=100)


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


async def score_candidate_against_requirements(
    candidate_profile: CandidateProfileInputSchema,
    requirements: RequirementsInputSchema | None = None,
    *,
    context: ToolContext,
) -> ScoreCandidateOutputSchema:
    """Score candidate fit using LLM judgment and persist the latest score."""
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
            candidate_profile,
            resolved_requirements,
            llm_config,
        )
    except RetryableModelOutputError:
        logging.warning("score_candidate_against_requirements retryable model output failure")
        raise

    confidence = _calibrate_candidate_confidence(
        result=score,
        requirements=resolved_requirements,
        candidate_profile=candidate_profile,
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
