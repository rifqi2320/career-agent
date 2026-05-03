"""Constants and enums for LLM construction."""

from models.llm import LlmModelName, LlmProvider

DEFAULT_PROVIDER = LlmProvider.LITELLM
DEFAULT_MODEL = LlmModelName.GLM_4_5_AIR
MODEL_PROVIDER_MAP: dict[LlmModelName, LlmProvider] = {
    LlmModelName.GEMMA_3_27B: LlmProvider.GOOGLE,
    LlmModelName.GEMMA_4_31B: LlmProvider.GOOGLE,
    LlmModelName.GLM_4_5_AIR: LlmProvider.LITELLM,
    LlmModelName.UNKNOWN: LlmProvider.UNKNOWN,
}
