"""Builder helpers for creating ADK BaseLlm objects."""
from modules.error.common import IncorrectCombinationError
from modules.error import UnknownOptionsError
from modules.llm.const import LlmProvider

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from models.llm import LiteLlmProviderConfig, LlmConfig
from modules.llm.const import MODEL_PROVIDER_MAP
from returns.result import safe, Success, Failure 


@safe
def build_llm(config: LlmConfig) -> BaseLlm:
    provider = config.provider or MODEL_PROVIDER_MAP[config.model_name]
    match provider:
        case LlmProvider.LITELLM:
            return _build_litellm(config)
        case LlmProvider.UNKNOWN:
            raise UnknownOptionsError(f"Unrecognized provider: {config.provider!s}")

@safe
def _build_litellm(litellm_config: LlmConfig) -> LiteLlm:
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
