"""Persist independent document-engine comparison runs."""

import sqlalchemy as sa
from alembic import op

revision = "0005_document_comparisons"
down_revision = "0004_ingestion_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "comparison_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("ingestion_id", sa.String(length=36), sa.ForeignKey("ingestions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("page_comparison_json_path", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_comparison_runs_ingestion_id", "comparison_runs", ["ingestion_id"])
    op.create_index("ix_comparison_runs_status", "comparison_runs", ["status"])
    op.create_table(
        "engine_processing_results",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("comparison_id", sa.String(length=36), sa.ForeignKey("comparison_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("package_version", sa.String(length=64), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("confidence_json", sa.Text(), nullable=True),
        sa.Column("output_json_path", sa.Text(), nullable=True),
        sa.Column("markdown_path", sa.Text(), nullable=True),
        sa.Column("normalized_json_path", sa.Text(), nullable=True),
        sa.Column("processing_seconds", sa.Float(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("comparison_id", "engine", name="uq_engine_comparison"),
    )
    op.create_index("ix_engine_processing_results_comparison_id", "engine_processing_results", ["comparison_id"])
    op.create_index("ix_engine_processing_results_status", "engine_processing_results", ["status"])


def downgrade() -> None:
    op.drop_index("ix_engine_processing_results_status", table_name="engine_processing_results")
    op.drop_index("ix_engine_processing_results_comparison_id", table_name="engine_processing_results")
    op.drop_table("engine_processing_results")
    op.drop_index("ix_comparison_runs_status", table_name="comparison_runs")
    op.drop_index("ix_comparison_runs_ingestion_id", table_name="comparison_runs")
    op.drop_table("comparison_runs")
