"""Schemas for candidate profile ingestion and matching."""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class CandidateProfileInputSchema(BaseModel):
    """Minimal structured candidate profile used by career matching."""

    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = Field(default=None, ge=0)
    seniority_level: str = Field(
        default="unknown",
        validation_alias=AliasChoices("seniority_level", "seniority"),
    )
    domain: str = "unknown"
