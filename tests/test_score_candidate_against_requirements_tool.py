from __future__ import annotations

from google.adk.tools import ToolContext
import pytest

from modules.error.common import ToolInputError
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
        confidence_score=81,
    )

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

    result = await tool_module.score_candidate_against_requirements(
        candidate_profile=CandidateProfileInputSchema(
            skills=["python"],
            years_experience=5,
            seniority_level="mid",
            domain="data",
        ),
        requirements=None,
        context=tool_context,
    )

    assert result == expected
    assert tool_context.state["last_score"] == expected.model_dump()


@pytest.mark.asyncio
async def test_score_candidate_passes_explicit_requirements_to_llm(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    requirements = RequirementsInputSchema(
        required_skills=["prompt engineering", "rag", "apis"],
        seniority_level="mid-level",
        domain="ai",
    )
    expected = ScoreCandidateOutputSchema(
        overall_score=92,
        dimension_scores=ScoreDimensionsSchema(
            skills=100,
            experience=75,
            seniority_fit=90,
        ),
        matched_skills=["apis", "prompt engineering", "rag"],
        gap_skills=[],
        confidence=ConfidenceLevel.HIGH,
        confidence_score=92,
    )

    async def fake_score_llm(
        candidate_profile: CandidateProfileInputSchema,
        requirements: RequirementsInputSchema,
        llm_config: object,  # noqa: ARG001
    ) -> ScoreCandidateOutputSchema:
        assert candidate_profile.skills == [
            "prompt design",
            "rag architectures",
            "api design",
        ]
        assert requirements.required_skills == ["prompt engineering", "rag", "apis"]
        return expected

    monkeypatch.setattr(
        tool_module,
        "_score_candidate_against_requirements_llm",
        fake_score_llm,
    )

    result = await tool_module.score_candidate_against_requirements(
        candidate_profile=CandidateProfileInputSchema(
            skills=["prompt design", "rag architectures", "api design"],
            years_experience=3,
            seniority_level="mid",
            domain="ai",
        ),
        requirements=requirements,
        context=tool_context,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_score_candidate_requires_requirements_when_state_missing(
    tool_context: ToolContext,
) -> None:
    with pytest.raises(ToolInputError, match="Missing requirements input"):
        await tool_module.score_candidate_against_requirements(
            candidate_profile=CandidateProfileInputSchema(skills=["python"]),
            requirements=None,
            context=tool_context,
        )
