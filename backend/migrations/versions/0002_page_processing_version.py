"""Track the page processing pipeline version."""

import sqlalchemy as sa
from alembic import op

revision = "0002_page_version"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing_columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("pages")}
    if "processing_version" not in existing_columns:
        op.add_column(
            "pages",
            sa.Column("processing_version", sa.String(length=32), nullable=False, server_default="0.1.0"),
        )


def downgrade() -> None:
    existing_columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("pages")}
    if "processing_version" in existing_columns:
        op.drop_column("pages", "processing_version")
