"""Utilities for structured LLM calls via ADK models."""

from __future__ import annotations
from safe_result import safe, safe_async

import json
import re
from typing import TypeVar

from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from models.llm import LlmConfig
from modules.builder.llm.builder import build_llm
from modules.error.common import DependencyError, RetryableModelOutputError

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


def extract_json_text_payload(response_text: str) -> str:
    """Extract raw JSON text from a model response."""
    text = response_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def _response_to_text(response_parts: list[types.Part]) -> str:
    """Convert model response parts into one text payload."""
    texts = [part.text for part in response_parts if part.text]
    if not texts:
        raise RetryableModelOutputError("LLM response contains no text parts to parse.")
    return "\n".join(texts).strip()


@safe
def _parse_json_payload(json_text: str) -> object:
    return json.loads(json_text)


def parse_model_json_payload(response_text: str) -> object:
    """Parse a possibly fenced model JSON response into a Python payload."""
    json_text = extract_json_text_payload(response_text)
    parse_result = _parse_json_payload(json_text)
    if parse_result.is_err():
        raise RetryableModelOutputError(
            "Model output failed JSON parsing; retry may succeed."
        ) from parse_result.error
    data = parse_result.value
    if data is None:
        raise RetryableModelOutputError(
            "Model output failed JSON parsing; retry may succeed."
        )
    return data


@safe
def _validate_schema_payload(
    data: object,
    schema: type[SchemaModelT],
) -> SchemaModelT:
    return schema.model_validate(data)


@safe_async
async def generate_structured_output(
    llm_config: LlmConfig,
    system_prompt: str,
    user_prompt: str,
    schema: type[SchemaModelT],
) -> SchemaModelT:
    """Generate schema-validated structured output using ADK LLM APIs."""
    llm_result = build_llm(llm_config)
    if llm_result.is_err():
        raise DependencyError(f"Failed to build LLM from config: {llm_result.error}")
    llm = llm_result.value
    if llm is None:
        raise DependencyError("LLM builder returned no model instance.")

    request = LlmRequest(
        model=llm.model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    request.set_output_schema(output_schema=schema)

    final_response = None
    async for response in llm.generate_content_async(llm_request=request, stream=False):
        final_response = response

    if final_response is None:
        raise RetryableModelOutputError("LLM returned no response.")
    if final_response.error_message:
        raise RetryableModelOutputError(
            f"LLM returned an error: {final_response.error_code} {final_response.error_message}"
        )
    if not final_response.content or not final_response.content.parts:
        raise RetryableModelOutputError("LLM response has no content parts.")

    raw_text = _response_to_text(final_response.content.parts)
    data = parse_model_json_payload(raw_text)

    validate_result = _validate_schema_payload(data, schema)
    if validate_result.is_err():
        raise RetryableModelOutputError(
            "Model output failed schema validation; retry may succeed."
        ) from validate_result.error
    validated_output = validate_result.value
    if validated_output is None:
        raise RetryableModelOutputError(
            "Model output failed schema validation; retry may succeed."
        )
    return validated_output
