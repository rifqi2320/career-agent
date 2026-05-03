"""Data access helpers for skill resources."""

from __future__ import annotations

from safe_result import safe
from sqlalchemy import and_, or_, select

from models.resource import SkillResource
from modules.database.client import SessionLocal
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
    with SessionLocal() as session:
        query = (
            select(SkillResource)
            .where(
                or_(
                    SkillResource.skill_name.ilike(f"%{normalized_skill}%"),
                    SkillResource.title.ilike(f"%{normalized_skill}%"),
                    SkillResource.abstracts.ilike(f"%{normalized_skill}%"),
                )
            )
            .limit(limit)
        )

        if normalized_seniority and normalized_seniority != "unknown":
            query = query.order_by(
                and_(
                    SkillResource.seniority_context.is_not(None),
                    SkillResource.seniority_context.ilike(f"%{normalized_seniority}%"),
                ).desc()
            )

        rows = session.scalars(query).all()
    return list(rows)
