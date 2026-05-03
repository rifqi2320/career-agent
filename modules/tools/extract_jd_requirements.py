from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import cast

from google.adk.tools import ToolContext
from pydantic import BaseModel, Field
from safe_result import ok

from models.llm import LlmConfig
from models.confidence import ConfidenceMetrics, ConfidenceLevel, calibrate_confidence
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import (
    RetryableModelOutputError,
    ToolExecutionError,
    ToolInputError,
)
from modules.extractor.html import read_page_content
from modules.logging import logging
from modules.tools.prompts import (
    EXTRACT_JD_SYSTEM_PROMPT,
    EXTRACT_JD_USER_PROMPT_TEMPLATE,
)
from modules.utils import generate_structured_output
from modules.utils.text import validate_url
from modules.utils.trace import increment_llm_calls


class ExtractJDRequirementOutputSchema(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority_level: str = "unknown"
    domain: str = "unknown"
    responsibilities: list[str] = Field(default_factory=list)
    confidence_score: int = 0
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN


def _estimate_extraction_confidence(
    requirements: ExtractJDRequirementOutputSchema,
) -> ConfidenceMetrics:
    """Compute deterministic confidence from extraction completeness and signal density."""
    required_signal = min(len(requirements.required_skills), 5) * 10
    nice_to_have_signal = min(len(requirements.nice_to_have_skills), 3) * 5
    responsibility_signal = min(len(requirements.responsibilities), 8) * 4
    metadata_signal = 0
    if requirements.domain and requirements.domain.lower() != "unknown":
        metadata_signal += 5
    if (
        requirements.seniority_level
        and requirements.seniority_level.lower() != "unknown"
    ):
        metadata_signal += 5

    completeness_score = (
        required_signal + nice_to_have_signal + responsibility_signal + metadata_signal
    )
    return calibrate_confidence(completeness_score)


async def extract_jd_requirements(
    url_or_text: str | dict[str, object],
    *,
    context: ToolContext,
) -> ExtractJDRequirementOutputSchema:
    """Extract job requirements from a given URL or text input."""
    started_at = perf_counter()
    resolved_url_or_text = _coerce_url_or_text(url_or_text)
    logging.info(
        "extract_jd_requirements started | input_length=%d",
        len(resolved_url_or_text),
    )
    text: str

    # Parse URL to text
    url = validate_url(resolved_url_or_text)
    if ok(url):
        logging.info("extract_jd_requirements input resolved as URL")
        logging.info("extract_jd_requirements fetching URL content")
        text_result = await read_page_content(url.value.geturl())
        if text_result.is_err():
            error_message = (
                "Failed to read page content from URL input. "
                f"Error: {text_result.error}"
            )
            logging.error(error_message)
            raise ToolExecutionError(error_message, original_error=text_result.error)
        else:
            if text_result.value and text_result.value.strip():
                text = text_result.value
                logging.info(
                    "extract_jd_requirements URL content read | text_length=%d",
                    len(text),
                )
            else:
                error_message = (
                    "Page content from URL input is empty and cannot be parsed."
                )
                logging.error(error_message)
                raise ToolExecutionError(error_message)
    else:
        if resolved_url_or_text.strip().lower().startswith(("http://", "https://")):
            raise ToolInputError(str(url.error), original_error=url.error)
        text = resolved_url_or_text
        logging.info(
            "extract_jd_requirements input resolved as raw text | text_length=%d",
            len(text),
        )

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    logging.info(
        "extract_jd_requirements LLM extraction started | model=%s",
        llm_config.model_name,
    )
    requirements = await _parse_requirements_from_text_with_retry(
        text=text,
        llm_config=llm_config,
        context=context,
    )

    calibrated_confidence = _estimate_extraction_confidence(requirements)
    result_payload = requirements.model_copy(
        update={
            "confidence_score": calibrated_confidence.confidence_score,
            "confidence": calibrated_confidence.confidence,
        }
    )
    context.state["last_requirements"] = result_payload.model_dump()
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logging.info(
        "extract_jd_requirements success | required=%d nice_to_have=%d responsibilities=%d confidence=%s confidence_score=%d elapsed_ms=%d",
        len(requirements.required_skills),
        len(requirements.nice_to_have_skills),
        len(requirements.responsibilities),
        result_payload.confidence,
        result_payload.confidence_score,
        elapsed_ms,
    )
    return result_payload


async def _parse_requirements_from_text_with_retry(
    *,
    text: str,
    llm_config: LlmConfig,
    context: ToolContext,
) -> ExtractJDRequirementOutputSchema:
    """Parse JD text, retrying one malformed model output before failing."""
    last_error: RetryableModelOutputError | None = None
    for attempt in range(1, 3):
        increment_llm_calls(context)
        try:
            return await _parse_requirements_from_text_llm(text, llm_config)
        except RetryableModelOutputError as error:
            last_error = error
            logging.warning(
                "extract_jd_requirements retryable model output failure | attempt=%d",
                attempt,
            )
    if last_error is None:
        raise RetryableModelOutputError("JD extraction failed for an unknown reason.")
    raise last_error


def _coerce_url_or_text(raw_url_or_text: object) -> str:
    """Normalize LLM-supplied URL/text payloads at the tool boundary."""
    if isinstance(raw_url_or_text, str):
        value = raw_url_or_text.strip()
    elif isinstance(raw_url_or_text, Mapping):
        value = _extract_text_field(cast("Mapping[object, object]", raw_url_or_text))
    else:
        raise ToolInputError("url_or_text must be a string or object with text/url.")

    if not value:
        raise ToolInputError("url_or_text must not be empty.")
    return value


def _extract_text_field(payload: Mapping[object, object]) -> str:
    for key in (
        "url_or_text",
        "url",
        "text",
        "job_description",
        "job_description_text",
        "content",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ToolInputError(
        "url_or_text object must include a non-empty string field named "
        "url_or_text, url, text, job_description, job_description_text, or content."
    )


async def _parse_requirements_from_text_llm(
    text: str, llm_config: LlmConfig
) -> ExtractJDRequirementOutputSchema:
    """Use an LLM to parse JD text into the required structured schema."""
    system_prompt = EXTRACT_JD_SYSTEM_PROMPT
    user_prompt = EXTRACT_JD_USER_PROMPT_TEMPLATE.render(job_description=text)
    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=ExtractJDRequirementOutputSchema,
    )
    return result.unwrap()
