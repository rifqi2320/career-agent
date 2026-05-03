from __future__ import annotations

from typing import cast

from google.adk.tools import ToolContext
import pytest
from safe_result import Ok

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
async def test_extract_jd_requirements_accepts_object_text_payload(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    expected = ExtractJDRequirementOutputSchema(
        required_skills=["python"],
        confidence_score=10,
        confidence=tool_module.ConfidenceLevel.LOW,
    )

    async def fake_parse_requirements(
        text: str,
        llm_config: object,  # noqa: ARG001
    ) -> ExtractJDRequirementOutputSchema:
        assert text == "We need Python."
        return expected

    monkeypatch.setattr(
        tool_module,
        "_parse_requirements_from_text_llm",
        fake_parse_requirements,
    )

    result = await tool_module.extract_jd_requirements(
        cast("str", {"job_description": "We need Python."}),
        context=tool_context,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_extract_jd_requirements_passes_string_url_to_page_reader(
    monkeypatch: pytest.MonkeyPatch,
    tool_context: ToolContext,
) -> None:
    expected = ExtractJDRequirementOutputSchema(
        required_skills=["sap"],
        confidence_score=10,
        confidence=tool_module.ConfidenceLevel.LOW,
    )
    captured_urls: list[str] = []

    async def fake_read_page_content(url: str) -> Ok[str]:
        captured_urls.append(url)
        return Ok("Job page text")

    async def fake_parse_requirements(
        text: str,
        llm_config: object,  # noqa: ARG001
    ) -> ExtractJDRequirementOutputSchema:
        assert text == "Job page text"
        return expected

    monkeypatch.setattr(tool_module, "read_page_content", fake_read_page_content)
    monkeypatch.setattr(
        tool_module,
        "_parse_requirements_from_text_llm",
        fake_parse_requirements,
    )

    result = await tool_module.extract_jd_requirements(
        "https://www.accenture.com/pl-en/careers/jobdetails?id=R00325310_en",
        context=tool_context,
    )

    assert result == expected
    assert captured_urls == [
        "https://www.accenture.com/pl-en/careers/jobdetails?id=R00325310_en"
    ]


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


@pytest.mark.asyncio
async def test_extract_jd_requirements_retries_one_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
    sample_jobreq_text: str,
    tool_context: ToolContext,
) -> None:
    expected = ExtractJDRequirementOutputSchema(
        required_skills=["python"],
        confidence_score=10,
        confidence=tool_module.ConfidenceLevel.LOW,
    )
    attempts = 0

    async def fake_parse_requirements(
        text: str,  # noqa: ARG001
        llm_config: object,  # noqa: ARG001
    ) -> ExtractJDRequirementOutputSchema:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableModelOutputError("schema mismatch")
        return expected

    monkeypatch.setattr(
        tool_module,
        "_parse_requirements_from_text_llm",
        fake_parse_requirements,
    )

    result = await tool_module.extract_jd_requirements(
        sample_jobreq_text,
        context=tool_context,
    )

    assert result == expected
    assert attempts == 2
    assert tool_context.state["total_llm_calls"] == 2
