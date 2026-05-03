from __future__ import annotations

from google.adk.tools import ToolContext
import pytest

from modules.error.common import ToolInputError
from modules.tools import research_skill_resources as tool_module
from modules.tools.research_skill_resources import (
    ResearchSkillResourcesOutputSchema,
    ResourceType,
    SkillResourceItemSchema,
)


@pytest.mark.asyncio
async def test_research_skill_resources_success_stores_last_state(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    expected = ResearchSkillResourcesOutputSchema(
        resources=[
            SkillResourceItemSchema(
                title="Advanced SQL",
                url="https://example.com/sql",
                estimated_hours=12,
                type=ResourceType.COURSE,
            ),
            SkillResourceItemSchema(
                title="SQL Indexing Lab",
                url="https://example.com/sql-lab",
                estimated_hours=8,
                type=ResourceType.PROJECT,
            ),
            SkillResourceItemSchema(
                title="PostgreSQL Documentation",
                url="https://example.com/postgres-docs",
                estimated_hours=6,
                type=ResourceType.DOC,
            ),
        ],
        relevance_score=88,
        confidence_score=90,
        confidence=tool_module.ConfidenceLevel.HIGH,
    )

    async def fake_research_agent(
        *,
        skill_name: str,
        seniority_context: str,
    ) -> ResearchSkillResourcesOutputSchema:
        assert skill_name == "sql"
        assert seniority_context == "senior"
        return expected

    monkeypatch.setattr(
        tool_module,
        "_research_skill_resources_agent",
        fake_research_agent,
    )

    result = await tool_module.research_skill_resources(
        skill_name="sql", seniority_context="senior", context=tool_context
    )

    assert result == expected
    assert tool_context.state["last_resources_research"] == expected.model_dump()


@pytest.mark.asyncio
async def test_research_skill_resources_requires_non_empty_skill(
    tool_context: ToolContext,
) -> None:
    with pytest.raises(ToolInputError, match="must not be empty"):
        await tool_module.research_skill_resources(
            skill_name="   ", context=tool_context
        )


@pytest.mark.asyncio
async def test_research_skill_resources_errors_when_no_resources_found(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    async def fake_research_agent(
        *,
        skill_name: str,  # noqa: ARG001
        seniority_context: str,  # noqa: ARG001
    ) -> ResearchSkillResourcesOutputSchema:
        raise ToolInputError("No skill resources found from DB or GitHub.")

    monkeypatch.setattr(
        tool_module,
        "_research_skill_resources_agent",
        fake_research_agent,
    )

    with pytest.raises(ToolInputError, match="No skill resources found"):
        await tool_module.research_skill_resources(
            skill_name="python", context=tool_context
        )
