"""Database runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from modules.error.common import ConfigurationError


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Validated database settings."""

    url: str


def load_database_settings() -> DatabaseSettings:
    """Load database settings from environment."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise ConfigurationError("DATABASE_URL must not be empty.")
    return DatabaseSettings(url=url)


database_settings = load_database_settings()
