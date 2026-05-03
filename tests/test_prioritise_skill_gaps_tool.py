from __future__ import annotations

from google.adk.tools import ToolContext
import pytest

from modules.error.common import ToolInputError
from modules.tools import prioritise_skill_gaps as tool_module
from modules.tools.prioritise_skill_gaps import (
    PrioritizeSkillGapsOutputSchema,
    PrioritizedSkillSchema,
)


@pytest.mark.asyncio
async def test_prioritise_skill_gaps_uses_state_and_stores_output(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    tool_context.state["last_score"] = {"gap_skills": ["sql", "airflow", "sql"]}
    expected = PrioritizeSkillGapsOutputSchema(
        prioritized_skills=[
            PrioritizedSkillSchema(
                skill="sql",
                priority_rank=1,
                estimated_match_gain_pct=10,
                rationale="Core requirement in most target roles.",
            ),
            PrioritizedSkillSchema(
                skill="airflow",
                priority_rank=2,
                estimated_match_gain_pct=6,
                rationale="Common orchestration expectation.",
            ),
        ],
        confidence=tool_module.ConfidenceLevel.LOW,
        confidence_score=8,
    )

    async def fake_prioritize_llm(
        *,
        gap_skills: list[str],
        job_market_context: str,
        llm_config: object,  # noqa: ARG001
    ) -> PrioritizeSkillGapsOutputSchema:
        assert gap_skills == ["sql", "airflow"]
        assert job_market_context == "us sql data platform"
        return expected

    monkeypatch.setattr(tool_module, "_prioritise_skill_gaps_llm", fake_prioritize_llm)

    result = await tool_module.prioritise_skill_gaps(
        gap_skills=None, job_market_context="us sql data platform", context=tool_context
    )

    assert result == expected
    assert tool_context.state["last_prioritized_skill_gaps"] == expected.model_dump()


@pytest.mark.asyncio
async def test_prioritise_skill_gaps_requires_state_when_argument_missing(
    tool_context: ToolContext,
) -> None:
    with pytest.raises(ToolInputError, match="Missing `gap_skills` argument"):
        await tool_module.prioritise_skill_gaps(gap_skills=None, context=tool_context)


@pytest.mark.asyncio
async def test_prioritise_skill_gaps_deduplicates_before_llm(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    expected = PrioritizeSkillGapsOutputSchema(
        prioritized_skills=[
            PrioritizedSkillSchema(
                skill="Python",
                priority_rank=1,
                estimated_match_gain_pct=10,
                rationale="Important in the target market.",
            ),
            PrioritizedSkillSchema(
                skill="airflow",
                priority_rank=2,
                estimated_match_gain_pct=5,
                rationale="Useful for workflow orchestration.",
            ),
        ],
        confidence=tool_module.ConfidenceLevel.LOW,
        confidence_score=8,
    )

    async def fake_prioritize_llm(
        *,
        gap_skills: list[str],
        job_market_context: str,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> PrioritizeSkillGapsOutputSchema:
        assert gap_skills == ["Python", "airflow"]
        return expected

    monkeypatch.setattr(tool_module, "_prioritise_skill_gaps_llm", fake_prioritize_llm)

    result = await tool_module.prioritise_skill_gaps(
        gap_skills=["Python", "airflow", "python"], context=tool_context
    )

    assert result == expected
