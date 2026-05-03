"""Runtime configuration for the career intelligence agent."""

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for the agent."""

    model: str = DEFAULT_MODEL


def load_settings() -> Settings:
    """Load settings from environment variables."""
    model = os.getenv("CAREER_INTELLIGENCE_MODEL", DEFAULT_MODEL).strip()
    if not model:
        message = "CAREER_INTELLIGENCE_MODEL must not be empty."
        raise ValueError(message)

    return Settings(model=model)


settings = load_settings()
