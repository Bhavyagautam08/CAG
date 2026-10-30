"""Persist page-level OCR confidence, word evidence, and processed preview."""

import sqlalchemy as sa
from alembic import op

revision = "0003_ocr_evidence"
down_revision = "0002_page_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("pages")}
    if "ocr_confidence_proxy" not in columns:
        op.add_column("pages", sa.Column("ocr_confidence_proxy", sa.Float(), nullable=True))
    if "ocr_word_boxes" not in columns:
        op.add_column("pages", sa.Column("ocr_word_boxes", sa.Text(), nullable=True))
    if "processed_image_path" not in columns:
        op.add_column("pages", sa.Column("processed_image_path", sa.Text(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("pages")}
    for column in ("processed_image_path", "ocr_word_boxes", "ocr_confidence_proxy"):
        if column in columns:
            op.drop_column("pages", column)

