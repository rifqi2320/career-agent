"""Root-level LLM config used by tools and modules."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from models.config.llm import LlmProfile, LlmProfilesConfig
from models.llm import LlmConfig
from modules.utils.config import parse_json_text, read_text_file, validate_model

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config.json"


def _load_project_llm_config() -> LlmProfilesConfig:
    """Load and validate `llm` profiles from root `config.json`."""
    read_result = read_text_file(CONFIG_PATH)
    if read_result.is_err():
        if isinstance(read_result.error, FileNotFoundError):
            raise FileNotFoundError(
                f"Missing root config file: {CONFIG_PATH}. Expected top-level `llm` key."
            ) from read_result.error
        raise RuntimeError(
            f"Could not read config file: {CONFIG_PATH}"
        ) from read_result.error
    raw_text = read_result.value
    if raw_text is None:
        raise RuntimeError(f"Config file read returned no content: {CONFIG_PATH}")

    parse_result = parse_json_text(raw_text)
    if parse_result.is_err():
        raise ValueError(
            f"Invalid JSON in config file: {CONFIG_PATH}"
        ) from parse_result.error
    payload = parse_result.value
    if payload is None:
        raise ValueError(f"Invalid JSON in config file: {CONFIG_PATH}")

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid config format in {CONFIG_PATH}: expected JSON object."
        )
    payload_dict = cast("dict[str, object]", payload)

    llm_payload = payload_dict.get("llm")
    if llm_payload is None:
        raise ValueError(
            f"Missing `llm` key in config file: {CONFIG_PATH}. "
            'Expected shape: {"llm": {"small": {...}, "main": {...}}}'
        )

    validate_result = validate_model(llm_payload, LlmProfilesConfig)
    if validate_result.is_err():
        raise ValueError(
            f"Invalid `llm` config in {CONFIG_PATH}: {validate_result.error}"
        ) from validate_result.error
    validated_llm_config = validate_result.value
    if validated_llm_config is None:
        raise ValueError(f"Invalid `llm` config in {CONFIG_PATH}: empty result")
    return validated_llm_config


PROJECT_LLM_CONFIG = _load_project_llm_config()


def get_llm_config(*, profile: LlmProfile = LlmProfile.MAIN) -> LlmConfig:
    """Return one configured LLM profile from root project config."""
    if profile is LlmProfile.SMALL:
        return PROJECT_LLM_CONFIG.small
    if profile is LlmProfile.MAIN:
        return PROJECT_LLM_CONFIG.main
    raise ValueError(f"Unsupported LLM profile: {profile}")
