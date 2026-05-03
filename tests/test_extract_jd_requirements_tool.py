from __future__ import annotations

from google.adk.tools import ToolContext
import pytest

from modules.error.common import RetryableModelOutputError, ToolInputError
from modules.tools import extract_jd_requirements as tool_module
from modules.tools.extract_jd_requirements import ExtractJDRequirementOutputSchema
from modules.utils.text import MAXIMUM_URL_LENGTH


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
        confidence_score=39,
        confidence=tool_module.ConfidenceLevel.LOW,
    )

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

    result = await tool_module.extract_jd_requirements(
        sample_jobreq_text, context=tool_context
    )

    assert result == expected
    assert tool_context.state["last_requirements"] == expected.model_dump()


@pytest.mark.asyncio
async def test_extract_jd_requirements_url_length_validation(
    tool_context: ToolContext,
) -> None:
    too_long_url = "https://example.com/" + "a" * (MAXIMUM_URL_LENGTH + 1)

    with pytest.raises(ToolInputError, match="maximum length"):
        await tool_module.extract_jd_requirements(too_long_url, context=tool_context)


@pytest.mark.asyncio
async def test_extract_jd_requirements_retryable_error_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    sample_jobreq_text: str,
    tool_context: ToolContext,
) -> None:
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

    with pytest.raises(RetryableModelOutputError):
        await tool_module.extract_jd_requirements(
            sample_jobreq_text, context=tool_context
        )
