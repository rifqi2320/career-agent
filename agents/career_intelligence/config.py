"""Runtime configuration for the career intelligence agent."""

from pathlib import Path

from models.config.llm import LlmProfilesConfig
from modules.config.llm import DEFAULT_CONFIG_PATH, load_project_llm_config


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> LlmProfilesConfig:
    """Load validated project settings for the agent."""
    return load_project_llm_config(config_path)
