from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from google.adk.tools import ToolContext
import pytest

from models.match import AgentStreamEvent
from safe_result import safe

from modules.error.common import ToolInputError, ToolTimeoutError
from modules.tools import research_skill_resources as tool_module
from modules.tools.research_skill_resources import (
    ResearchSkillResourcesOutputSchema,
    ResourceType,
    SkillResourceItemSchema,
)


@dataclass
class FakeResourceRow:
    title: str
    abstracts: str | None
    url: str
    estimated_hours: int
    resource_type: str
    skill_name: str
    seniority_context: str | None
    source: str | None


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
        parent_job_id: str | None = None,
        event_sink: Callable[[AgentStreamEvent], None] | None = None,
    ) -> ResearchSkillResourcesOutputSchema:
        assert skill_name == "sql"
        assert seniority_context == "senior"
        assert parent_job_id is None
        assert event_sink is not None
        return expected

    monkeypatch.setattr(
        tool_module,
        "run_resource_research_agent",
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
        parent_job_id: str | None = None,  # noqa: ARG001
        event_sink: Callable[[AgentStreamEvent], None] | None = None,  # noqa: ARG001
    ) -> ResearchSkillResourcesOutputSchema:
        raise ToolInputError("No skill resources found from DB or GitHub.")

    monkeypatch.setattr(
        tool_module,
        "run_resource_research_agent",
        fake_research_agent,
    )

    with pytest.raises(ToolInputError, match="No skill resources found"):
        await tool_module.research_skill_resources(
            skill_name="python", context=tool_context
        )


@pytest.mark.asyncio
async def test_research_skill_resources_retries_timeout_then_curated_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    rows = [
        FakeResourceRow(
            title="Python Docs",
            abstracts=None,
            url="https://docs.python.org/3/",
            estimated_hours=8,
            resource_type="doc",
            skill_name="python",
            seniority_context="mid",
            source="seed",
        ),
        FakeResourceRow(
            title="Python Project",
            abstracts=None,
            url="https://example.com/python-project",
            estimated_hours=12,
            resource_type="project",
            skill_name="python",
            seniority_context="mid",
            source="seed",
        ),
        FakeResourceRow(
            title="Python Course",
            abstracts=None,
            url="https://example.com/python-course",
            estimated_hours=20,
            resource_type="course",
            skill_name="python",
            seniority_context="mid",
            source="seed",
        ),
    ]
    attempts = 0

    async def fake_research_agent(
        *,
        skill_name: str,  # noqa: ARG001
        seniority_context: str,  # noqa: ARG001
        parent_job_id: str | None = None,  # noqa: ARG001
        event_sink: Callable[[AgentStreamEvent], None] | None = None,  # noqa: ARG001
    ) -> ResearchSkillResourcesOutputSchema:
        nonlocal attempts
        attempts += 1
        raise ToolTimeoutError("timed out")

    @safe
    def fake_list_resources(
        *,
        skill_name: str,
        seniority_context: str,
        limit: int = 5,  # noqa: ARG001
    ) -> list[FakeResourceRow]:
        assert skill_name == "python"
        assert seniority_context == "mid"
        return rows

    monkeypatch.setattr(
        tool_module,
        "run_resource_research_agent",
        fake_research_agent,
    )
    monkeypatch.setattr(tool_module, "list_skill_resources", fake_list_resources)

    result = await tool_module.research_skill_resources(
        skill_name="python",
        seniority_context="mid",
        context=tool_context,
    )

    assert attempts == 2
    assert len(result.resources) == 3
    assert result.resources[0].title == "Python Docs"
    assert tool_context.state["total_llm_calls"] == 2
    assert tool_context.state["fallbacks_triggered"] == 1
