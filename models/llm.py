"""Pydantic models for LLM configuration and build results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LlmProvider(StrEnum):
    """Supported logical providers for model naming."""

    LITELLM = "litellm"
    GOOGLE = "google"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> LlmProvider:
        """Fallback to UNKNOWN for unrecognized providers."""
        return cls.UNKNOWN


class LlmModelName(StrEnum):
    """Supported model identifiers."""

    GEMMA_3_27B = "gemma-3-27b-it"
    GEMMA_4_31B = "gemma-4-31b-it"
    GLM_4_5_AIR = "z-ai/glm-4.5-air"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> LlmModelName:
        """Fallback to UNKNOWN for unrecognized model names."""
        return cls.UNKNOWN


class LlmClientType(StrEnum):
    """Logical client types for LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> LlmClientType:
        """Fallback to UNKNOWN for unrecognized client types."""
        return cls.UNKNOWN


class LLMProviderConfig(BaseModel):
    """Base provider config for LLM-backed models."""

    stream: bool = True
    timeout: float = 120
    max_retries: int = 3

    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class LiteLlmProviderConfig(LLMProviderConfig):
    """Provider config for LiteLLM-backed models."""

    client_type: LlmClientType
    api_base: str
    api_key: str
    api_version: str | None = None


class GoogleProviderConfig(LLMProviderConfig):
    """Provider config for Google Gemini-backed models."""

    # Placeholder for future Google Gemini provider configuration options
    pass


LLMProviderType = LiteLlmProviderConfig | GoogleProviderConfig
ProviderConfig = LLMProviderType


class LlmConfig(BaseModel):
    """Input configuration for building an ADK BaseLlm instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: LlmProvider
    model_name: LlmModelName
    provider_config: ProviderConfig | None = None
