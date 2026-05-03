"""Domain helpers for converting resource rows into resource schemas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from modules.resources.schemas import (
    CandidateResourceSchema,
    ResourceType,
    SkillResourceItemSchema,
)


class SkillResourceRow(Protocol):
    """Database row shape needed for resource schema conversion."""

    title: str
    abstracts: str | None
    url: str
    estimated_hours: int
    resource_type: str
    skill_name: str
    seniority_context: str | None
    source: str | None


def rows_to_candidate_resources(
    rows: Iterable[SkillResourceRow],
) -> list[CandidateResourceSchema]:
    """Convert curated DB rows into internal research-agent candidates."""
    return [
        CandidateResourceSchema(
            title=row.title,
            abstracts=row.abstracts,
            url=row.url,
            estimated_hours=row.estimated_hours,
            type=ResourceType(row.resource_type),
            skill_name=row.skill_name,
            seniority_context=row.seniority_context or "unknown",
            source=row.source,
        )
        for row in rows
    ]


def rows_to_skill_resource_items(
    rows: Iterable[SkillResourceRow],
    *,
    limit: int,
) -> list[SkillResourceItemSchema]:
    """Convert curated DB rows into selected learning resources."""
    return [
        SkillResourceItemSchema(
            title=row.title,
            url=row.url,
            estimated_hours=row.estimated_hours,
            type=ResourceType(row.resource_type),
        )
        for row in list(rows)[:limit]
    ]
