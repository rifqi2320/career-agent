"""Database runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from modules.error.common import ConfigurationError

DEFAULT_DATABASE_URL = "postgresql+psycopg://career:career@localhost:15432/career_agent"


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Validated database settings."""

    url: str = DEFAULT_DATABASE_URL


def load_database_settings() -> DatabaseSettings:
    """Load database settings from environment."""
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip()
    if not url:
        raise ConfigurationError("DATABASE_URL must not be empty.")
    return DatabaseSettings(url=url)


database_settings = load_database_settings()
