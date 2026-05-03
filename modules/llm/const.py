"""Constants and enums for LLM construction."""

from enum import StrEnum


class LlmProvider(StrEnum):
    """Supported logical providers for model naming."""

    LITELLM = "litellm"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "LlmProvider":
        """Fallback to UNKNOWN for unrecognized providers."""
        return cls.UNKNOWN


class LlmModelName(StrEnum):
    """Supported model identifiers."""

    GEMMA_4_31B = "google/gemma-4-31b"
    GLM_4_5_AIR = "z-ai/glm-4.5-air"

    UNKNOWN = "unknown"
    
    @classmethod
    def _missing_(cls, value: object) -> "LlmModelName":
        """Fallback to UNKNOWN for unrecognized model names."""
        return cls.UNKNOWN


DEFAULT_PROVIDER = LlmProvider.LITELLM
DEFAULT_MODEL = LlmModelName.GLM_4_5_AIR
MODEL_PROVIDER_MAP: dict[LlmModelName, LlmProvider] = {
    LlmModelName.GEMMA_4_31B: LlmProvider.LITELLM,
    LlmModelName.GLM_4_5_AIR: LlmProvider.LITELLM,
    LlmModelName.UNKNOWN: LlmProvider.UNKNOWN,
}
