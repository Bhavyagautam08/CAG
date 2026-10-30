import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import document_comparison
from backend.app import comparison_jobs
from backend.app.db import Base
from backend.app.models import ComparisonRun, EngineProcessingResult, Ingestion


def test_paddle_blocks_preserve_structure_without_duplicate_entries():
    result = {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "Registration No: SYN-123",
                "block_bbox": [1, 2, 3, 4],
            },
            {
                "block_label": "table",
                "block_content": "<table><tr><td>Value</td></tr></table>",
            },
        ]
    }

    blocks = document_comparison._paddle_blocks(
        [result, result],
        page_number=1,
    )

    assert len(blocks) == 2
    assert blocks[0]["text"] == "Registration No: SYN-123"
    assert blocks[0]["bbox"] == [1, 2, 3, 4]
    assert blocks[1]["kind"] == "table"


def test_page_comparison_normalizes_whitespace_and_reports_missing_pages():
    result = document_comparison.build_page_comparison(
        {"pages": [{"page_number": 1, "text": "A  B\nC", "blocks": [], "tables": []}]},
        {
            "pages": [
                {"page_number": 1, "text": "A B C", "blocks": [], "tables": []},
                {"page_number": 2, "text": "Only Docling", "blocks": [], "tables": []},
            ]
        },
    )

    assert result["accuracy"].startswith("Not Measured")
    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    assert result["pages"][0]["text_equal_after_whitespace_normalization"] is True
    assert result["pages"][0]["word_differences"] == []
    assert result["pages"][1]["pp_structure_text"] == ""
    assert result["pages"][1]["docling_text"] == "Only Docling"


def test_worker_failure_is_reported_without_running_engine_in_api_process(tmp_path, monkeypatch):
    observed = {}

    def failed_worker(*args, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(args[0], -1073741819, stdout="", stderr="native runtime crash")

    monkeypatch.setattr(document_comparison.subprocess, "run", failed_worker)

    with pytest.raises(RuntimeError, match="native runtime crash"):
        document_comparison.run_engine("pp_structure", tmp_path / "input.pdf", tmp_path / "output")

    assert observed["capture_output"] is True
    assert observed["timeout"] == 3600


def test_worker_result_is_read_and_removed(tmp_path, monkeypatch):
    def successful_worker(args, **kwargs):
        result_path = tmp_path / "output" / "_worker-result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({"page_count": 16}), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(document_comparison.subprocess, "run", successful_worker)

    result = document_comparison.run_engine(
        "docling",
        tmp_path / "input.pdf",
        tmp_path / "output",
    )

    assert result == {"page_count": 16}
    assert not (tmp_path / "output" / "_worker-result.json").exists()


def test_comparison_run_persists_success_when_other_engine_fails(tmp_path, monkeypatch):
    database_engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=database_engine)
    session_factory = sessionmaker(bind=database_engine, expire_on_commit=False)
    source_dir = tmp_path / "ingestion"
    source_dir.mkdir()
    source_pdf = source_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 synthetic")
    source_hash = hashlib.sha256(source_pdf.read_bytes()).hexdigest()

    with session_factory() as db:
        ingestion = Ingestion(
            id="comparison-test-ingestion",
            filename="synthetic.pdf",
            sha256=source_hash,
            file_size=source_pdf.stat().st_size,
            page_count=1,
            status="completed",
            storage_path=str(source_pdf),
        )
        db.add(ingestion)
        db.flush()
        comparison = comparison_jobs.create_comparison(db, ingestion, "compare")
        db.commit()
        comparison_id = comparison.id

    monkeypatch.setattr(
        comparison_jobs,
        "get_settings",
        lambda: SimpleNamespace(storage_dir=tmp_path),
    )

    def fake_engine(engine_name, _source_pdf, output_dir):
        if engine_name == "pp_structure":
            raise RuntimeError("native runtime failure")
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "native_path": output_dir / "native.json",
            "markdown_path": output_dir / "document.md",
            "normalized_path": output_dir / "normalized.json",
        }
        artifacts["native_path"].write_text("{}", encoding="utf-8")
        artifacts["markdown_path"].write_text("# Synthetic", encoding="utf-8")
        artifacts["normalized_path"].write_text(
            json.dumps({"engine": engine_name, "pages": [{"page_number": 1, "text": "Synthetic"}]}),
            encoding="utf-8",
        )
        return {
            "package_version": "test",
            "config": {},
            "confidence": {"availability": "not_available"},
            **{key: str(path) for key, path in artifacts.items()},
            "duration_seconds": 0.1,
            "page_count": 1,
        }

    monkeypatch.setattr(comparison_jobs, "run_engine", fake_engine)
    comparison_jobs.run_comparison(comparison_id, session_factory)

    with session_factory() as db:
        comparison = db.get(ComparisonRun, comparison_id)
        results = {result.engine: result for result in db.query(EngineProcessingResult).all()}
        assert comparison.status == "completed_with_warnings"
        assert results["docling"].status == "completed"
        assert results["docling"].normalized_json_path
        assert results["pp_structure"].status == "failed"
        assert "native runtime failure" in results["pp_structure"].error

    database_engine.dispose()
