import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .db import SessionLocal
from .document_comparison import build_page_comparison, run_engine
from .models import ComparisonRun, EngineProcessingResult, Ingestion, utcnow

logger = logging.getLogger(__name__)


def create_comparison(
    db: Session,
    ingestion: Ingestion,
    mode: str,
) -> ComparisonRun:
    engines = {
        "pp_structure": ["pp_structure"],
        "docling": ["docling"],
        "compare": ["pp_structure", "docling"],
    }.get(mode)
    if engines is None:
        raise ValueError(f"Unsupported comparison mode: {mode}")
    comparison = ComparisonRun(
        ingestion_id=ingestion.id,
        mode=mode,
        source_sha256=ingestion.sha256,
        status="queued",
        engines=[EngineProcessingResult(engine=engine, status="queued") for engine in engines],
    )
    db.add(comparison)
    db.flush()
    return comparison


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_failed(engine_run: EngineProcessingResult, exc: Exception) -> None:
    engine_run.status = "failed"
    engine_run.error = f"{type(exc).__name__}: {str(exc)[:800]}"
    engine_run.updated_at = utcnow()
    logger.exception(
        "Document comparison engine failed",
        extra={"comparison_id": engine_run.comparison_id, "engine": engine_run.engine},
    )


def run_comparison(
    comparison_id: str,
    db_factory: Callable[[], Session] = SessionLocal,
) -> None:
    db = db_factory()
    try:
        comparison = db.scalar(
            select(ComparisonRun)
            .options(selectinload(ComparisonRun.engines))
            .where(ComparisonRun.id == comparison_id)
        )
        if comparison is None:
            return
        ingestion = db.get(Ingestion, comparison.ingestion_id)
        if ingestion is None:
            comparison.status = "failed"
            comparison.error = "Source ingestion no longer exists."
            db.commit()
            return
        source_root = Path(get_settings().storage_dir).resolve()
        source = Path(ingestion.storage_path).resolve()
        if source_root not in source.parents or not source.is_file() or source.suffix.casefold() != ".pdf":
            comparison.status = "failed"
            comparison.error = "The original source PDF is unavailable inside private storage."
            db.commit()
            return
        if _file_hash(source) != comparison.source_sha256 or comparison.source_sha256 != ingestion.sha256:
            comparison.status = "failed"
            comparison.error = "The source PDF hash changed; this run was stopped."
            db.commit()
            return

        run_dir = (source.parent / "comparisons" / comparison.id).resolve()
        if source_root not in run_dir.parents:
            comparison.status = "failed"
            comparison.error = "Invalid comparison output location."
            db.commit()
            return
        run_dir.mkdir(parents=True, exist_ok=False)
        comparison.status = "processing"
        comparison.updated_at = utcnow()
        db.commit()

        successes: dict[str, dict] = {}
        for engine_run in comparison.engines:
            engine_run.status = "processing"
            engine_run.error = None
            engine_run.updated_at = utcnow()
            db.commit()
            output_dir = run_dir / engine_run.engine
            try:
                result = run_engine(engine_run.engine, source, output_dir)
                for artifact_key in ("native_path", "markdown_path", "normalized_path"):
                    artifact = Path(result[artifact_key]).resolve()
                    if output_dir.resolve() not in artifact.parents or not artifact.is_file():
                        raise ValueError("An engine returned an artifact outside its private output directory.")
                engine_run.package_version = result["package_version"]
                engine_run.config_json = json.dumps(result["config"], ensure_ascii=False)
                engine_run.confidence_json = json.dumps(result["confidence"], ensure_ascii=False)
                engine_run.output_json_path = str(Path(result["native_path"]).relative_to(source_root))
                engine_run.markdown_path = str(Path(result["markdown_path"]).relative_to(source_root))
                engine_run.normalized_json_path = str(Path(result["normalized_path"]).relative_to(source_root))
                engine_run.processing_seconds = result["duration_seconds"]
                engine_run.page_count = result["page_count"]
                engine_run.status = "completed"
                engine_run.updated_at = utcnow()
                successes[engine_run.engine] = json.loads(Path(result["normalized_path"]).read_text(encoding="utf-8"))
            except Exception as exc:
                db.rollback()
                _set_failed(engine_run, exc)
                shutil.rmtree(output_dir, ignore_errors=True)
            db.commit()

        if "pp_structure" in successes and "docling" in successes:
            comparison_data = build_page_comparison(successes["pp_structure"], successes["docling"])
            comparison_path = run_dir / "page-comparison.json"
            comparison_path.write_text(json.dumps(comparison_data, ensure_ascii=False, indent=2), encoding="utf-8")
            comparison.page_comparison_json_path = str(comparison_path.relative_to(source_root))
        statuses = [engine_run.status for engine_run in comparison.engines]
        if all(status == "completed" for status in statuses):
            comparison.status = "completed"
        elif any(status == "completed" for status in statuses):
            comparison.status = "completed_with_warnings"
        else:
            comparison.status = "failed"
            comparison.error = "All selected document engines failed."
        comparison.updated_at = utcnow()
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Document comparison run failed", extra={"comparison_id": comparison_id})
        comparison = db.get(ComparisonRun, comparison_id)
        if comparison is not None:
            comparison.status = "failed"
            comparison.error = "Comparison orchestration failed; see local server log for details."
            comparison.updated_at = utcnow()
            db.commit()
    finally:
        db.close()
