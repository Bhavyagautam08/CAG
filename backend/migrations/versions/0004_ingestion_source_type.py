"""Identify whether an ingestion came from a PDF or uploaded images."""

import sqlalchemy as sa
from alembic import op

revision = "0004_ingestion_source"
down_revision = "0003_ocr_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("ingestions")}
    if "source_type" not in columns:
        op.add_column(
            "ingestions",
            sa.Column("source_type", sa.String(length=16), nullable=False, server_default="pdf"),
        )


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("ingestions")}
    if "source_type" in columns:
        op.drop_column("ingestions", "source_type")
