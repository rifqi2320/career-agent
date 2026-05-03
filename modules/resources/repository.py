"""Data access helpers for skill resources."""

from __future__ import annotations

from typing import Any

from safe_result import safe
from sqlalchemy import case, or_, select

from models.resource import SkillResource
from modules.database.client import create_session
from modules.error.common import ToolInputError


@safe
def list_skill_resources(
    *,
    skill_name: str,
    seniority_context: str,
    limit: int = 30,
) -> list[SkillResource]:
    """Fetch candidate resources by skill name and optional seniority."""
    normalized_skill = skill_name.strip()
    if not normalized_skill:
        raise ToolInputError("skill_name must not be empty.")
    if limit <= 0:
        raise ToolInputError("limit must be greater than 0.")

    normalized_seniority = seniority_context.strip().casefold()
    with create_session() as session:
        ordering: list[Any] = [
            case(
                (SkillResource.skill_name.ilike(normalized_skill), 0),
                else_=1,
            ),
        ]
        if normalized_seniority and normalized_seniority != "unknown":
            ordering.append(
                case(
                    (
                        SkillResource.seniority_context.ilike(
                            f"%{normalized_seniority}%"
                        ),
                        0,
                    ),
                    else_=1,
                )
            )
        ordering.append(SkillResource.title.asc())

        query = select(SkillResource).where(
            or_(
                SkillResource.skill_name.ilike(f"%{normalized_skill}%"),
                SkillResource.title.ilike(f"%{normalized_skill}%"),
                SkillResource.abstracts.ilike(f"%{normalized_skill}%"),
            )
        )
        query = query.order_by(*ordering).limit(limit)

        rows = session.scalars(query).all()
    return list(rows)
