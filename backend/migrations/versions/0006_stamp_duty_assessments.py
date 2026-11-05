"""Create stamp duty assessments table for Article 34 gift deeds."""

import sqlalchemy as sa
from alembic import op

revision = "0006_stamp_duty_assessments"
down_revision = "0005_document_comparisons"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    if "stamp_duty_assessments" not in tables:
        op.create_table(
            "stamp_duty_assessments",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "document_id",
                sa.String(length=36),
                sa.ForeignKey("logical_documents.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
            ),
            sa.Column("donor_name", sa.String(length=255), nullable=True),
            sa.Column("donee_name", sa.String(length=255), nullable=True),
            sa.Column("relationship", sa.String(length=100), nullable=True),
            sa.Column("relationship_normalized", sa.String(length=100), nullable=True),
            sa.Column("relationship_eligibility", sa.String(length=32), nullable=False, server_default="unknown"),
            sa.Column("property_type", sa.String(length=64), nullable=False, server_default="unknown"),
            sa.Column("property_description", sa.Text(), nullable=True),
            sa.Column("property_location", sa.Text(), nullable=True),
            sa.Column("survey_number", sa.String(length=100), nullable=True),
            sa.Column("area", sa.String(length=100), nullable=True),
            sa.Column("jurisdiction", sa.String(length=64), nullable=False, server_default="unknown"),
            sa.Column("market_value", sa.BigInteger(), nullable=True),
            sa.Column("base_duty", sa.Integer(), nullable=True),
            sa.Column("metro_cess_rate", sa.Float(), nullable=True),
            sa.Column("metro_cess_amount", sa.BigInteger(), nullable=True),
            sa.Column("expected_total", sa.BigInteger(), nullable=True),
            sa.Column("stamp_duty_paid", sa.BigInteger(), nullable=True),
            sa.Column("metro_cess_paid", sa.BigInteger(), nullable=True),
            sa.Column("total_stamp_duty_paid", sa.BigInteger(), nullable=True),
            sa.Column("challan_number", sa.String(length=100), nullable=True),
            sa.Column("difference", sa.BigInteger(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="requires_review"),
            sa.Column("legal_rule_version", sa.String(length=64), nullable=False, server_default="MH-ART34-2024.1"),
            sa.Column("evidence", sa.Text(), nullable=True),
            sa.Column("source_pages", sa.Text(), nullable=True),
            sa.Column("review_status", sa.String(length=32), nullable=False, server_default="unreviewed"),
            sa.Column("reviewer_notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_stamp_duty_assessments_document_id", "stamp_duty_assessments", ["document_id"], unique=True)
        op.create_index("ix_stamp_duty_assessments_status", "stamp_duty_assessments", ["status"])
        op.create_index("ix_stamp_duty_assessments_review_status", "stamp_duty_assessments", ["review_status"])


def downgrade() -> None:
    op.drop_index("ix_stamp_duty_assessments_review_status", table_name="stamp_duty_assessments")
    op.drop_index("ix_stamp_duty_assessments_status", table_name="stamp_duty_assessments")
    op.drop_index("ix_stamp_duty_assessments_document_id", table_name="stamp_duty_assessments")
    op.drop_table("stamp_duty_assessments")

