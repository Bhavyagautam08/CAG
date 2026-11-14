"""Add identification_documents column to stamp_duty_assessments.

Revision ID: 0009_extracted_id_documents
Revises: 0008_cess_uncertain_and_location_band
Create Date: 2026-11-12
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_extracted_id_documents"
down_revision = "0008_cess_uncertain_and_location_band"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("stamp_duty_assessments") as batch_op:
        batch_op.add_column(
            sa.Column("identification_documents", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("stamp_duty_assessments") as batch_op:
        batch_op.drop_column("identification_documents")
