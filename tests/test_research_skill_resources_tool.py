from __future__ import annotations

from google.adk.tools import ToolContext
from dataclasses import dataclass

import pytest
from safe_result import safe, safe_async

from modules.error.common import RetryableModelOutputError
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
    fake_rows = [
        FakeResourceRow(
            title="Advanced SQL",
            abstracts="Indexing and query planning",
            url="https://example.com/sql",
            estimated_hours=12,
            resource_type="course",
            skill_name="sql",
            seniority_context="senior",
            source="seed",
        )
    ]
    expected = ResearchSkillResourcesOutputSchema(
        resources=[
            SkillResourceItemSchema(
                title="Advanced SQL",
                url="https://example.com/sql",
                estimated_hours=12,
                type=ResourceType.COURSE,
            )
        ],
        relevance_score=88,
    )

    @safe
    def fake_list_resources(
        *,
        skill_name: str,  # noqa: ARG001
        seniority_context: str,  # noqa: ARG001
        limit: int = 30,  # noqa: ARG001
    ) -> list[FakeResourceRow]:
        return fake_rows

    @safe_async
    async def fake_research_llm(
        *,
        skill_name: str,  # noqa: ARG001
        seniority_context: str,  # noqa: ARG001
        candidate_resources: object,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ResearchSkillResourcesOutputSchema:
        return expected

    monkeypatch.setattr(tool_module, "list_skill_resources", fake_list_resources)
    monkeypatch.setattr(tool_module, "_research_skill_resources_llm", fake_research_llm)

    result = await tool_module._research_skill_resources(
        skill_name="sql", seniority_context="senior", context=tool_context
    )

    assert result.is_ok()
    assert result.value == expected
    assert tool_context.state["last_resources_research"] == expected.model_dump()


@pytest.mark.asyncio
async def test_research_skill_resources_requires_non_empty_skill(
    tool_context: ToolContext,
) -> None:
    result = await tool_module._research_skill_resources(
        skill_name="   ", context=tool_context
    )

    assert result.is_err()
    assert isinstance(result.error, ValueError)
    assert "must not be empty" in str(result.error)


@pytest.mark.asyncio
async def test_research_skill_resources_retryable_error_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    fake_rows = [
        FakeResourceRow(
            title="Python Docs",
            abstracts=None,
            url="https://example.com/python-docs",
            estimated_hours=8,
            resource_type="doc",
            skill_name="python",
            seniority_context=None,
            source="seed",
        )
    ]

    @safe
    def fake_list_resources(
        *,
        skill_name: str,  # noqa: ARG001
        seniority_context: str,  # noqa: ARG001
        limit: int = 30,  # noqa: ARG001
    ) -> list[FakeResourceRow]:
        return fake_rows

    @safe_async
    async def fake_research_error(
        *,
        skill_name: str,  # noqa: ARG001
        seniority_context: str,  # noqa: ARG001
        candidate_resources: object,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ResearchSkillResourcesOutputSchema:
        raise RetryableModelOutputError("schema mismatch")

    monkeypatch.setattr(tool_module, "list_skill_resources", fake_list_resources)
    monkeypatch.setattr(
        tool_module,
        "_research_skill_resources_llm",
        fake_research_error,
    )

    result = await tool_module._research_skill_resources(
        skill_name="python", context=tool_context
    )

    assert result.is_err()
    assert isinstance(result.error, RetryableModelOutputError)
