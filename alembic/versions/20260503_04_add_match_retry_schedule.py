"""add match retry schedule

Revision ID: 20260503_04
Revises: 20260503_03
Create Date: 2026-05-03 18:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260503_04"
down_revision = "20260503_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "match_jobs",
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_match_jobs_next_attempt_at"),
        "match_jobs",
        ["next_attempt_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_match_jobs_next_attempt_at"), table_name="match_jobs")
    op.drop_column("match_jobs", "next_attempt_at")
