"""SQLAlchemy models for learning resource storage."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class SkillResource(Base):
    """Curated skill learning resource for job-gap recommendations."""

    __tablename__ = "skill_resources"
    __table_args__ = (
        CheckConstraint(
            "resource_type IN ('course', 'project', 'cert', 'doc')",
            name="ck_skill_resources_resource_type",
        ),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    skill_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    seniority_context: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    abstracts: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    estimated_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    resource_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
