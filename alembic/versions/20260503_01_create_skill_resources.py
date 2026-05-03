"""create skill_resources table

Revision ID: 20260503_01
Revises:
Create Date: 2026-05-03 10:15:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260503_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "skill_resources",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("skill_name", sa.String(length=255), nullable=False),
        sa.Column("seniority_context", sa.String(length=64), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("estimated_hours", sa.Integer(), nullable=False),
        sa.Column("resource_type", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "resource_type IN ('course', 'project', 'cert', 'doc')",
            name="ck_skill_resources_resource_type",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_skill_resources_skill_name"),
        "skill_resources",
        ["skill_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_skill_resources_seniority_context"),
        "skill_resources",
        ["seniority_context"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_skill_resources_seniority_context"), table_name="skill_resources"
    )
    op.drop_index(op.f("ix_skill_resources_skill_name"), table_name="skill_resources")
    op.drop_table("skill_resources")
