"""Add directional relationship, schedule uncertainty, and three-regime fields to stamp_duty_assessments.

Revision ID: 0007_three_regime_stamp_duty
Revises: 0006_stamp_duty_assessments
Create Date: 2026-11-05
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_three_regime_stamp_duty"
down_revision = "0006_stamp_duty_assessments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("stamp_duty_assessments") as batch_op:
        batch_op.add_column(sa.Column("donor_role", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("donee_role", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("donee_relation_to_donor", sa.String(length=64), nullable=True))
        batch_op.add_column(
            sa.Column("property_type_uncertain", sa.Boolean(), nullable=False, server_default=sa.text("0"))
        )
        batch_op.add_column(
            sa.Column("selected_rule", sa.String(length=64), nullable=True, server_default="review_required")
        )
        batch_op.add_column(sa.Column("duty_rate", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("cess_rule", sa.String(length=64), nullable=True))
        batch_op.add_column(
            sa.Column("rule_version", sa.String(length=64), nullable=True, server_default="MH-ART34-2024.1")
        )
        batch_op.add_column(
            sa.Column("effective_date", sa.String(length=32), nullable=True, server_default="2017-05-16")
        )
        batch_op.add_column(sa.Column("calculation_components", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source_evidence", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("stamp_duty_assessments") as batch_op:
        batch_op.drop_column("source_evidence")
        batch_op.drop_column("calculation_components")
        batch_op.drop_column("effective_date")
        batch_op.drop_column("rule_version")
        batch_op.drop_column("cess_rule")
        batch_op.drop_column("duty_rate")
        batch_op.drop_column("selected_rule")
        batch_op.drop_column("property_type_uncertain")
        batch_op.drop_column("donee_relation_to_donor")
        batch_op.drop_column("donee_role")
        batch_op.drop_column("donor_role")

