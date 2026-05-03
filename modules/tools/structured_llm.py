"""Shared structured LLM helper for ADK tool implementations."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from models.llm import LlmConfig
from modules.utils import generate_structured_output

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


async def generate_tool_structured_output(
    *,
    llm_config: LlmConfig,
    system_prompt: str,
    user_prompt: str,
    schema: type[SchemaModelT],
) -> SchemaModelT:
    """Generate and unwrap a schema-validated LLM response for a tool."""
    result = await generate_structured_output(
        llm_config=llm_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
    )
    return result.unwrap()
