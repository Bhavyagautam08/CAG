"""Add cess_uncertain and location_band to stamp_duty_assessments.

Revision ID: 0008_cess_uncertain_and_location_band
Revises: 0007_three_regime_stamp_duty
Create Date: 2026-11-05
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_cess_uncertain_and_location_band"
down_revision = "0007_three_regime_stamp_duty"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("stamp_duty_assessments") as batch_op:
        batch_op.add_column(
            sa.Column("cess_uncertain", sa.Boolean(), nullable=False, server_default=sa.text("0"))
        )
        batch_op.add_column(
            sa.Column("location_band", sa.String(length=64), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("stamp_duty_assessments") as batch_op:
        batch_op.drop_column("location_band")
        batch_op.drop_column("cess_uncertain")

