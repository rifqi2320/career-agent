from __future__ import annotations

from google.adk.tools import ToolContext
import pytest
from safe_result import safe_async

from modules.error.common import RetryableModelOutputError
from modules.tools import score_candidate_against_requirements as tool_module
from modules.tools.score_candidate_against_requirements import (
    CandidateProfileInputSchema,
    ConfidenceLevel,
    RequirementsInputSchema,
    ScoreCandidateOutputSchema,
    ScoreDimensionsSchema,
)


@pytest.mark.asyncio
async def test_score_candidate_uses_state_requirements_and_stores_last_score(
    monkeypatch: pytest.MonkeyPatch,
    sample_candidate_profile_text: str,  # noqa: ARG001
    tool_context: ToolContext,
) -> None:
    tool_context.state["last_requirements"] = {
        "required_skills": ["python", "sql"],
        "seniority_level": "senior",
        "domain": "data",
        "responsibilities": ["build pipelines"],
    }
    expected = ScoreCandidateOutputSchema(
        overall_score=81,
        dimension_scores=ScoreDimensionsSchema(
            skills=84,
            experience=78,
            seniority_fit=80,
        ),
        matched_skills=["python"],
        gap_skills=["sql"],
        confidence=ConfidenceLevel.MEDIUM,
    )

    @safe_async
    async def fake_score_llm(
        candidate_profile: CandidateProfileInputSchema,  # noqa: ARG001
        requirements: RequirementsInputSchema,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ScoreCandidateOutputSchema:
        return expected

    monkeypatch.setattr(
        tool_module,
        "_score_candidate_against_requirements_llm",
        fake_score_llm,
    )

    result = await tool_module._score_candidate_against_requirements(
        candidate_profile=CandidateProfileInputSchema(
            skills=["python"],
            years_experience=5,
            seniority_level="mid",
            domain="data",
        ),
        requirements=None,
        context=tool_context,
    )

    assert result.is_ok()
    assert result.value == expected
    assert tool_context.state["last_score"] == expected.model_dump()


@pytest.mark.asyncio
async def test_score_candidate_requires_requirements_when_state_missing(
    tool_context: ToolContext,
) -> None:
    result = await tool_module._score_candidate_against_requirements(
        candidate_profile=CandidateProfileInputSchema(skills=["python"]),
        requirements=None,
        context=tool_context,
    )

    assert result.is_err()
    assert isinstance(result.error, ValueError)
    assert "Missing requirements input" in str(result.error)


@pytest.mark.asyncio
async def test_score_candidate_retryable_error_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    requirements = RequirementsInputSchema(required_skills=["python"])

    @safe_async
    async def fake_score_error(
        candidate_profile: CandidateProfileInputSchema,  # noqa: ARG001
        requirements: RequirementsInputSchema,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ScoreCandidateOutputSchema:
        raise RetryableModelOutputError("schema mismatch")

    monkeypatch.setattr(
        tool_module,
        "_score_candidate_against_requirements_llm",
        fake_score_error,
    )

    result = await tool_module._score_candidate_against_requirements(
        candidate_profile=CandidateProfileInputSchema(skills=["python"]),
        requirements=requirements,
        context=tool_context,
    )

    assert result.is_err()
    assert isinstance(result.error, RetryableModelOutputError)
