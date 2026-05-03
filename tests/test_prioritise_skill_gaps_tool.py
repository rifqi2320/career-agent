from __future__ import annotations

from google.adk.tools import ToolContext
import pytest
from safe_result import safe_async

from modules.error.common import RetryableModelOutputError
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
        ]
    )

    @safe_async
    async def fake_prioritize_llm(
        *,
        gap_skills: list[str],  # noqa: ARG001
        job_market_context: str,  # noqa: ARG001
        last_requirements: object,  # noqa: ARG001
        last_score: object,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> PrioritizeSkillGapsOutputSchema:
        return expected

    monkeypatch.setattr(tool_module, "_prioritise_skill_gaps_llm", fake_prioritize_llm)

    result = await tool_module._prioritise_skill_gaps(
        gap_skills=None, job_market_context="us data platform", context=tool_context
    )

    assert result.is_ok()
    assert result.value == expected
    assert tool_context.state["last_prioritized_skill_gaps"] == expected.model_dump()


@pytest.mark.asyncio
async def test_prioritise_skill_gaps_requires_state_when_argument_missing(
    tool_context: ToolContext,
) -> None:
    result = await tool_module._prioritise_skill_gaps(
        gap_skills=None, context=tool_context
    )

    assert result.is_err()
    assert isinstance(result.error, ValueError)
    assert "Missing `gap_skills` argument" in str(result.error)


@pytest.mark.asyncio
async def test_prioritise_skill_gaps_detects_output_skill_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    mismatched = PrioritizeSkillGapsOutputSchema(
        prioritized_skills=[
            PrioritizedSkillSchema(
                skill="kubernetes",
                priority_rank=1,
                estimated_match_gain_pct=8,
                rationale="High demand.",
            )
        ]
    )

    @safe_async
    async def fake_prioritize_mismatch(
        *,
        gap_skills: list[str],  # noqa: ARG001
        job_market_context: str,  # noqa: ARG001
        last_requirements: object,  # noqa: ARG001
        last_score: object,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> PrioritizeSkillGapsOutputSchema:
        return mismatched

    monkeypatch.setattr(
        tool_module,
        "_prioritise_skill_gaps_llm",
        fake_prioritize_mismatch,
    )

    result = await tool_module._prioritise_skill_gaps(
        gap_skills=["python"], context=tool_context
    )

    assert result.is_err()
    assert isinstance(result.error, RetryableModelOutputError)
