"""Configuration for learning-resource retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceResearchConfig:
    """Runtime settings for resource research dependencies."""

    github_api_base: str = "https://api.github.com"
    github_token: str | None = None
    github_timeout_seconds: float = 10.0
    agent_timeout_seconds: float = 30.0


def load_resource_research_config() -> ResourceResearchConfig:
    """Load resource research settings from environment variables."""
    github_api_base = os.environ.get("GITHUB_API_BASE", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    github_pat_token = os.environ.get("GITHUB_PAT_TOKEN", "").strip()
    github_timeout_raw = os.environ.get("GITHUB_TIMEOUT_SECONDS", "").strip()
    agent_timeout_raw = os.environ.get("RESOURCE_RESEARCH_TIMEOUT_SECONDS", "").strip()

    return ResourceResearchConfig(
        github_api_base=github_api_base or "https://api.github.com",
        github_token=github_token or github_pat_token or None,
        github_timeout_seconds=_parse_positive_float(
            github_timeout_raw,
            default=10.0,
        ),
        agent_timeout_seconds=_parse_positive_float(
            agent_timeout_raw,
            default=30.0,
        ),
    )


def _parse_positive_float(raw_value: str, *, default: float) -> float:
    """Parse a positive float environment value with a safe local default."""
    if not raw_value:
        return default
    try:
        parsed_value = float(raw_value)
    except ValueError:
        return default
    if parsed_value <= 0:
        return default
    return parsed_value
