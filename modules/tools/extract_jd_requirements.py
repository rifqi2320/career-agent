from __future__ import annotations

from time import perf_counter

from google.adk.tools import ToolContext
from jinja2 import Template
from pydantic import BaseModel, Field
from safe_result import ok, safe_async

from models.llm import LlmConfig
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError
from modules.extractor.html import read_page_content
from modules.logging import logging
from modules.tools.wrapper import wrap_safe_tool
from modules.utils import generate_structured_output
from modules.utils.text import validate_url

MAX_URL_LENGTH = 2048

EXTRACT_JD_SYSTEM_PROMPT = """
You are an information extraction engine.
Extract job requirements from the provided job description.
Return strict JSON only, matching this schema:
- required_skills: list[str]
- nice_to_have_skills: list[str]
- seniority_level: str
- domain: str
- responsibilities: list[str]

Rules:
- Do not include explanations.
- If information is missing, return empty lists and "unknown" for strings.
- Keep list items concise and deduplicated.
""".strip()
EXTRACT_JD_USER_PROMPT_TEMPLATE = Template(
    """
Job description text:
{{ job_description }}
""".strip()
)


class ExtractJDRequirementOutputSchema(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority_level: str = "unknown"
    domain: str = "unknown"
    responsibilities: list[str] = Field(default_factory=list)


@safe_async
async def _extract_jd_requirements(
    url_or_text: str,
    *,
    context: ToolContext,
) -> ExtractJDRequirementOutputSchema:
    """Extract job requirements from a given URL or text input."""
    started_at = perf_counter()
    logging.info(
        "extract_jd_requirements started | input_length=%d",
        len(url_or_text),
    )
    text: str

    # Parse URL to text
    url = validate_url(url_or_text)
    if ok(url):
        logging.info("extract_jd_requirements input resolved as URL")
        if len(url_or_text) > MAX_URL_LENGTH:
            raise ValueError(
                f"URL exceeds maximum length of {MAX_URL_LENGTH} characters."
            )
        logging.info("extract_jd_requirements fetching URL content")
        text_result = await read_page_content(url.value)
        if text_result.is_err():
            error_message = (
                "Failed to read page content from URL input. "
                f"Error: {text_result.error}"
            )
            logging.error(error_message)
            raise RuntimeError(error_message)
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
                raise RuntimeError(error_message)
    else:
        text = url_or_text
        logging.info(
            "extract_jd_requirements input resolved as raw text | text_length=%d",
            len(text),
        )

    llm_config: LlmConfig = get_llm_config(profile=LlmProfile.MAIN)
    logging.info(
        "extract_jd_requirements LLM extraction started | model=%s",
        llm_config.model_name,
    )
    result = await _parse_requirements_from_text_llm(text, llm_config)
    if result.is_ok() and result.value is not None:
        context.state["last_requirements"] = result.value.model_dump()
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logging.info(
            "extract_jd_requirements success | required=%d nice_to_have=%d responsibilities=%d elapsed_ms=%d",
            len(result.value.required_skills),
            len(result.value.nice_to_have_skills),
            len(result.value.responsibilities),
            elapsed_ms,
        )
        return result.value

    if isinstance(result.error, RetryableModelOutputError):
        logging.warning(
            "extract_jd_requirements retryable model output failure | error=%s",
            result.error,
        )
        raise result.error

    error_message = f"LLM extraction failed: {result.error}"
    logging.error(error_message)
    raise RuntimeError(error_message)


@safe_async
async def _parse_requirements_from_text_llm(
    text: str, LlmConfig: LlmConfig
) -> ExtractJDRequirementOutputSchema:
    """Use an LLM to parse JD text into the required structured schema."""
    system_prompt = EXTRACT_JD_SYSTEM_PROMPT
    user_prompt = EXTRACT_JD_USER_PROMPT_TEMPLATE.render(job_description=text)
    result = await generate_structured_output(
        llm_config=LlmConfig,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=ExtractJDRequirementOutputSchema,
    )
    return result.unwrap()


extract_jd_requirements = wrap_safe_tool(_extract_jd_requirements)
