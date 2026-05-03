from __future__ import annotations

from time import perf_counter

from google.adk.tools import ToolContext
from jinja2 import Template
from pydantic import BaseModel, Field
from safe_result import ok

from models.llm import LlmConfig
from models.confidence import ConfidenceMetrics, ConfidenceLevel, calibrate_confidence
from modules.config.llm import LlmProfile, get_llm_config
from modules.error.common import RetryableModelOutputError, ToolExecutionError, ToolInputError
from modules.extractor.html import read_page_content
from modules.logging import logging
from modules.utils import generate_structured_output
from modules.utils.text import validate_url
from modules.utils.trace import increment_llm_calls

EXTRACT_JD_SYSTEM_PROMPT = """
You are an information extraction engine.
Extract job requirements from the provided job description.

Output strictly a JSON object only, and nothing else (no markdown, no prose, no fences).
The object must match this exact schema:
- required_skills: list[str]
- nice_to_have_skills: list[str]
- seniority_level: str
- domain: str
- responsibilities: list[str]

Rules:
- If information is missing, use empty lists and "unknown" for strings.
- Use only details present in the input text; do not infer unstated items.
- Normalize each skill to lowercase.
- Normalize near-equivalent wording with these mappings before deduplication:
  - "api design", "api development", "api integration", "apis" -> "apis"
  - "embedding models", "embeddings", "vector databases", "vector db" -> "vector database"
  - "llms" -> "llm"
  - "prompt design", "prompt engineering" -> "prompt engineering"
  - "rag architecture", "rag architectures", "retrieval augmented generation" -> "rag"
  - "function/tool calling", "function calling" -> "tool calling"
- Do not include "tool calling" in `required_skills` unless explicitly required in the job description.
- Sort `required_skills` and `nice_to_have_skills` alphabetically.
- Keep only unique values in each list.
- Ensure `nice_to_have_skills` contains no item that is already in `required_skills`.
- Keep responsibilities concise, action-oriented, de-duplicated, sorted alphabetically, and at most 8 items.
- Normalize seniority_level to one of: "junior", "mid-level", "senior", "lead", "unknown".
- Use these seniority rules:
  - explicit "junior" or 0-1 years -> "junior"
  - explicit "mid" or 2-4 years -> "mid-level"
  - explicit "senior" or 5+ years -> "senior"
  - explicit "lead", "staff", "principal", or people leadership -> "lead"
  - no explicit title or years -> "unknown"
- Normalize domain to a short lowercase label (or "unknown" if unclear).
- Return an object in this example shape:
{
  "required_skills": ["api design", "python", "rag"],
  "nice_to_have_skills": ["langchain"],
  "seniority_level": "mid-level",
  "domain": "fintech",
  "responsibilities": ["build evaluation loops", "ship ai features"]
}
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
    if requirements.seniority_level and requirements.seniority_level.lower() != "unknown":
        metadata_signal += 5

    completeness_score = (
        required_signal + nice_to_have_signal + responsibility_signal + metadata_signal
    )
    return calibrate_confidence(completeness_score)


async def extract_jd_requirements(
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
        logging.info("extract_jd_requirements fetching URL content")
        text_result = await read_page_content(url.value)
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
        if url_or_text.strip().lower().startswith(("http://", "https://")):
            raise ToolInputError(str(url.error), original_error=url.error)
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
    increment_llm_calls(context)
    try:
        requirements = await _parse_requirements_from_text_llm(text, llm_config)
    except RetryableModelOutputError:
        logging.warning("extract_jd_requirements retryable model output failure")
        raise

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
