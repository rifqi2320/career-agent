from __future__ import annotations

from google.adk.tools import ToolContext
import pytest
from safe_result import safe_async

from modules.error.common import RetryableModelOutputError
from modules.tools import extract_jd_requirements as tool_module
from modules.tools.extract_jd_requirements import ExtractJDRequirementOutputSchema


@pytest.mark.asyncio
async def test_extract_jd_requirements_text_success_stores_state(
    monkeypatch: pytest.MonkeyPatch,
    sample_jobreq_text: str,
    tool_context: ToolContext,
) -> None:
    expected = ExtractJDRequirementOutputSchema(
        required_skills=["python", "sql"],
        nice_to_have_skills=["airflow"],
        seniority_level="senior",
        domain="data",
        responsibilities=["build pipelines"],
    )

    @safe_async
    async def fake_parse_requirements(
        text: str,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ExtractJDRequirementOutputSchema:
        return expected

    monkeypatch.setattr(
        tool_module,
        "_parse_requirements_from_text_llm",
        fake_parse_requirements,
    )

    result = await tool_module._extract_jd_requirements(
        sample_jobreq_text, context=tool_context
    )

    assert result.is_ok()
    assert result.value == expected
    assert tool_context.state["last_requirements"] == expected.model_dump()


@pytest.mark.asyncio
async def test_extract_jd_requirements_url_length_validation(
    tool_context: ToolContext,
) -> None:
    too_long_url = "https://example.com/" + "a" * (tool_module.MAX_URL_LENGTH + 1)
    result = await tool_module._extract_jd_requirements(
        too_long_url, context=tool_context
    )

    assert result.is_err()
    assert isinstance(result.error, ValueError)
    assert "maximum length" in str(result.error)


@pytest.mark.asyncio
async def test_extract_jd_requirements_retryable_error_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    sample_jobreq_text: str,
    tool_context: ToolContext,
) -> None:
    @safe_async
    async def fake_parse_requirements_error(
        text: str,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ExtractJDRequirementOutputSchema:
        raise RetryableModelOutputError("schema mismatch")

    monkeypatch.setattr(
        tool_module,
        "_parse_requirements_from_text_llm",
        fake_parse_requirements_error,
    )

    result = await tool_module._extract_jd_requirements(
        sample_jobreq_text, context=tool_context
    )

    assert result.is_err()
    assert isinstance(result.error, RetryableModelOutputError)
