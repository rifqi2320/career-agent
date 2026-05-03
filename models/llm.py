"""Pydantic models for LLM configuration and build results."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from modules.llm.const import (
    DEFAULT_MODEL,
    LlmModelName,
    LlmProvider,
)

class LlmClientType(Enum):
    """Logical client types for LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "LlmClientType":
        """Fallback to UNKNOWN for unrecognized client types."""
        return cls.UNKNOWN


class LiteLlmProviderConfig(LlmConfig):
    """Provider config for LiteLLM-backed models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["litellm"] = "litellm"
    
    client_type: LlmClientType
    api_base: str
    api_key: str
    api_version: str | None = None

    timeout: float = 120
    max_retries: int = 3
    stream: bool = True
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


ProviderConfig = Annotated[LiteLlmProviderConfig, Field(discriminator="type")]


class LlmConfig(BaseModel):
    """Input configuration for building an ADK BaseLlm instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: LlmProvider | None = None
    model_name: LlmModelName = DEFAULT_MODEL
    provider_config: ProviderConfig | None = None
