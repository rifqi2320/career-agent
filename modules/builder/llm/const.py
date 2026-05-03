"""Constants and enums for LLM construction."""

from models.llm import LlmModelName, LlmProvider

MODEL_PROVIDER_MAP: dict[LlmModelName, LlmProvider] = {
    LlmModelName.GEMINI_3_1_FLASH_LITE_PREVIEW: LlmProvider.GOOGLE,
    LlmModelName.GEMINI_2_5_FLASH_LITE: LlmProvider.GOOGLE,
    LlmModelName.GEMMA_3_27B: LlmProvider.GOOGLE,
    LlmModelName.GEMMA_4_31B: LlmProvider.GOOGLE,
    LlmModelName.GLM_4_5_AIR: LlmProvider.LITELLM,
    LlmModelName.UNKNOWN: LlmProvider.UNKNOWN,
}
