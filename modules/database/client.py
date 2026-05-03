"""SQLAlchemy engine/session client."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from modules.config.database import load_database_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process SQLAlchemy engine, creating it lazily."""
    global _engine
    if _engine is None:
        _engine = create_engine(load_database_settings().url, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process SQLAlchemy session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            class_=Session,
        )
    return _session_factory


def create_session() -> Session:
    """Create a database session using the lazy session factory."""
    return get_session_factory()()
