"""Project LLM config used by agents, tools, and modules."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from models.config.llm import LlmProfile, LlmProfilesConfig
from models.llm import LlmConfig
from modules.error.common import ConfigurationError, UnknownOptionsError
from modules.utils.config import parse_json_text, read_text_file, validate_model

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "default.json"
CONFIG_PATH = DEFAULT_CONFIG_PATH


def load_project_llm_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> LlmProfilesConfig:
    """Load and validate `llm` profiles from a project config JSON file."""
    read_result = read_text_file(config_path)
    if read_result.is_err():
        if isinstance(read_result.error, FileNotFoundError):
            raise ConfigurationError(
                f"Missing project config file: {config_path}. "
                "Expected top-level `llm` key."
            ) from read_result.error
        raise ConfigurationError(
            f"Could not read config file: {config_path}"
        ) from read_result.error
    raw_text = read_result.value
    if raw_text is None:
        raise ConfigurationError(
            f"Config file read returned no content: {config_path}"
        )

    parse_result = parse_json_text(raw_text)
    if parse_result.is_err():
        raise ConfigurationError(
            f"Invalid JSON in config file: {config_path}"
        ) from parse_result.error
    payload = parse_result.value
    if payload is None:
        raise ConfigurationError(f"Invalid JSON in config file: {config_path}")

    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Invalid config format in {config_path}: expected JSON object."
        )
    payload_dict = cast("dict[str, object]", payload)

    llm_payload = payload_dict.get("llm")
    if llm_payload is None:
        raise ConfigurationError(
            f"Missing `llm` key in config file: {config_path}. "
            'Expected shape: {"llm": {"small": {...}, "main": {...}}}'
        )

    validate_result = validate_model(llm_payload, LlmProfilesConfig)
    if validate_result.is_err():
        raise ConfigurationError(
            f"Invalid `llm` config in {config_path}: {validate_result.error}"
        ) from validate_result.error
    validated_llm_config = validate_result.value
    if validated_llm_config is None:
        raise ConfigurationError(f"Invalid `llm` config in {config_path}: empty result")
    return validated_llm_config


PROJECT_LLM_CONFIG = load_project_llm_config()


def get_llm_config(*, profile: LlmProfile = LlmProfile.MAIN) -> LlmConfig:
    """Return one configured LLM profile from root project config."""
    if profile is LlmProfile.SMALL:
        return PROJECT_LLM_CONFIG.small
    if profile is LlmProfile.MAIN:
        return PROJECT_LLM_CONFIG.main
    raise UnknownOptionsError(f"Unsupported LLM profile: {profile}")
