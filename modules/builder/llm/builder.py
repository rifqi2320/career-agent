"""Builder helpers for creating ADK BaseLlm objects."""

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from models.llm import (
    LiteLlmProviderConfig,
    LlmConfig,
    LlmProvider,
    GoogleProviderConfig,
)
from safe_result import safe
from google.adk.models import Gemini

from modules.builder.llm.const import MODEL_PROVIDER_MAP
from modules.error.common import (
    DependencyError,
    IncorrectCombinationError,
    UnknownOptionsError,
)


@safe
def build_llm(config: LlmConfig) -> BaseLlm:
    provider = config.provider or MODEL_PROVIDER_MAP[config.model_name]
    if provider is LlmProvider.LITELLM:
        return _build_litellm_llm(config).unwrap()
    if provider is LlmProvider.GOOGLE:
        return _build_google_llm(config).unwrap()
    raise UnknownOptionsError(f"Unrecognized provider: {config.provider!s}")


def build_required_llm(config: LlmConfig, *, purpose: str) -> BaseLlm:
    """Build an LLM or raise a dependency error with caller context."""
    result = build_llm(config)
    if result.is_err():
        raise DependencyError(f"Failed to build {purpose} model: {result.error}")
    model = result.value
    if model is None:
        raise DependencyError(f"Failed to build {purpose} model: empty result")
    return model


@safe
def _build_google_llm(google_config: LlmConfig) -> BaseLlm:
    """Build and return a Google Gemini LLM instance."""
    if google_config.provider_config is None:
        return Gemini(model=google_config.model_name.value)

    if not isinstance(google_config.provider_config, GoogleProviderConfig):
        raise IncorrectCombinationError(
            "Expected provider_config of type GoogleProviderConfig for Google provider"
        )
    return Gemini(
        model=google_config.model_name.value,
        **google_config.provider_config.extra_kwargs,
    )


@safe
def _build_litellm_llm(litellm_config: LlmConfig) -> LiteLlm:
    """Build and return a LiteLlm instance."""
    if not isinstance(litellm_config.provider_config, LiteLlmProviderConfig):
        raise IncorrectCombinationError(
            "Expected provider_config of type LiteLlmProviderConfig for LiteLLM provider"
        )

    return LiteLlm(
        **litellm_config.provider_config.extra_kwargs,
        model=litellm_config.model_name.value,
        custom_llm_provider=litellm_config.provider_config.client_type.value,
        api_base=litellm_config.provider_config.api_base,
        api_key=litellm_config.provider_config.api_key,
        api_version=litellm_config.provider_config.api_version,
        timeout=litellm_config.provider_config.timeout,
        max_retries=litellm_config.provider_config.max_retries,
        stream=litellm_config.provider_config.stream,
    )
