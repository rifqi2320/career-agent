"""add abstracts column to skill_resources

Revision ID: 20260503_02
Revises: 20260503_01
Create Date: 2026-05-03 10:45:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260503_02"
down_revision = "20260503_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("skill_resources", sa.Column("abstracts", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("skill_resources", "abstracts")
