import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import mimetypes
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .config import (
    METRO_CESS_JURISDICTIONS,
    NON_METRO_INDICATORS,
    STAMP_DUTY_BASE_AMOUNT,
    STAMP_DUTY_METRO_CESS_RATE,
    STAMP_DUTY_RULE_VERSION,
    get_settings,
)
from .comparison_jobs import create_comparison, run_comparison
from .db import Base, SessionLocal, engine, get_db
from .image_utils import validate_image
from .models import (
    AuditEvent,
    CaseRecord,
    ComparisonRun,
    EngineProcessingResult,
    DocumentPerson,
    DocumentProperty,
    ExtractedField,
    Ingestion,
    LogicalDocument,
    Page,
    ProcessingJob,
    Person,
    PropertyRecord,
    ReviewDecision,
    StampDutyAssessment,
    ValidationFinding,
)
from .ocr import detect_table_lines, get_ocr_status
from .pdf_utils import LocalImageBundle, LocalPdf
from .pdf_report import IngestionNotCompleteError, SourceChangedError, build_ingestion_report
from .processing import (
    RelationshipEligibility,
    evaluate_article_34_relationship,
    parse_indian_currency_amount,
    process_ingestion,
    refresh_ingestion_extraction,
    resolve_authoritative_location_band,
    select_stamp_duty_rule,
    sync_case_entities,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()
Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
Base.metadata.create_all(bind=engine)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="case-processing")

app = FastAPI(title="Local Case Document Intelligence", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def protect_api(request: Request, call_next):
    if settings.app_api_token and request.url.path.startswith("/api"):
        expected = f"Bearer {settings.app_api_token}"
        if request.headers.get("authorization") != expected:
            return JSONResponse(status_code=401, content={"error": "Authentication required"})
    return await call_next(request)


class FieldCorrection(BaseModel):
    value: str | None = Field(max_length=4000)
    verification_status: Literal["verified", "uncertain", "incorrect"] = "verified"


class PageCorrection(BaseModel):
    corrected_text: str = Field(max_length=100000)


class CategoryCorrection(BaseModel):
    category: str = Field(min_length=1, max_length=80)


class CaseLinkDecision(BaseModel):
    case_id: str | None = None
    decision: Literal["approved", "rejected"]
    note: str | None = Field(default=None, max_length=2000)


class SplitDocumentRequest(BaseModel):
    split_before_page: int = Field(ge=1)


class MergeDocumentRequest(BaseModel):
    other_document_id: str


class FindingDecision(BaseModel):
    status: Literal["resolved", "dismissed", "open"]
    note: str | None = Field(default=None, max_length=2000)


class ComparisonRequest(BaseModel):
    mode: Literal["pp_structure", "docling", "compare"]


class StampDutyAssessmentResponse(BaseModel):
    id: int
    document_id: str | int
    donor_name: str | None = None
    donee_name: str | None = None
    donor_role: str | None = None
    donee_role: str | None = None
    donee_relation_to_donor: str | None = None
    property_type: str = "unknown"
    property_type_uncertain: bool = False
    cess_uncertain: bool = False
    location_band: str | None = None
    property_description: str | None = None
    property_location: str | None = None
    survey_number: str | None = None
    area: str | None = None
    relationship: str | None = None
    relationship_normalized: str | None = None
    relationship_eligibility: str = "unknown"
    jurisdiction: str = "unknown"
    selected_rule: str | None = "review_required"
    duty_rate: str | None = None
    cess_rule: str | None = None
    rule_version: str | None = "MH-ART34-2024.1"
    effective_date: str | None = "2017-05-16"
    calculation_components: Any = None
    source_evidence: Any = None
    identification_documents: dict[str, Any] = Field(
        default_factory=lambda: _parse_identification_documents(None)
    )
    market_value: int | None = None
    base_stamp_duty: int | None = None
    metro_cess_rate: float | None = None
    metro_cess: int | None = None
    expected_total: int | None = None
    stamp_duty_paid: int | None = None
    metro_cess_paid: int | None = None
    total_stamp_duty_paid: int | None = None
    challan_number: str | None = None
    difference: int | None = None
    status: str = "requires_review"
    legal_rule_version: str = "MH-ART34-2024.1"
    evidence: Any = None
    source_pages: list[int] = Field(default_factory=list)
    review_status: str = "unreviewed"
    reviewer_notes: str | None = None


class StampDutyAssessmentUpdateRequest(BaseModel):
    property_type: str | None = None
    property_type_uncertain: bool | None = None
    cess_uncertain: bool | None = None
    location_band: str | None = None
    property_description: str | None = None
    property_location: str | None = None
    survey_number: str | None = None
    area: str | None = None
    donor_role: str | None = None
    donee_role: str | None = None
    donee_relation_to_donor: str | None = None
    relationship: str | None = None
    relationship_eligibility: Literal["eligible", "not_eligible", "unknown", "ambiguous", "article_34_200", "article_34_3_percent", "article_25"] | None = None
    jurisdiction: str | None = None
    selected_rule: Literal["article_34_200", "article_34_3_percent", "article_25", "review_required"] | None = None
    duty_rate: str | None = None
    cess_rule: str | None = None
    rule_version: str | None = None
    effective_date: str | None = None
    market_value: int | None = None
    stamp_duty_paid: int | None = None
    metro_cess_paid: int | None = None
    total_stamp_duty_paid: int | None = None
    challan_number: str | None = None
    status: Literal["compliant", "short_paid", "not_available", "not_applicable", "requires_review"] | None = None
    review_status: Literal["unreviewed", "confirmed", "corrected", "requires_review"] | None = None
    reviewer_notes: str | None = None


def _process_ingestion_with_comparison(ingestion_id: str, comparison_id: str) -> None:
    process_ingestion(ingestion_id, SessionLocal)
    run_comparison(comparison_id, SessionLocal)


def _comparison_summary(comparison: ComparisonRun) -> dict:
    return {
        "id": comparison.id,
        "ingestion_id": comparison.ingestion_id,
        "mode": comparison.mode,
        "source_sha256": comparison.source_sha256,
        "status": comparison.status,
        "error": comparison.error,
        "created_at": comparison.created_at.isoformat(),
        "engines": [
            {
                "engine": result.engine,
                "status": result.status,
                "package_version": result.package_version,
                "page_count": result.page_count,
                "processing_seconds": result.processing_seconds,
                "error": result.error,
            }
            for result in comparison.engines
        ],
    }


def _stored_artifact(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    storage_root = Path(settings.storage_dir).resolve()
    artifact = (storage_root / relative_path).resolve()
    if storage_root not in artifact.parents or not artifact.is_file():
        return None
    return artifact


def _read_json_artifact(relative_path: str | None) -> dict | list | None:
    if not relative_path:
        return None
    artifact = _stored_artifact(relative_path)
    if artifact is None:
        raise HTTPException(500, "A stored comparison artifact is unavailable.")
    try:
        return json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read a stored document-comparison JSON artifact")
        raise HTTPException(500, "A stored comparison artifact could not be read.")


def _audit(db: Session, entity_type: str, entity_id: str, action: str, before: str | None, after: str | None):
    db.add(AuditEvent(entity_type=entity_type, entity_id=entity_id, action=action, before_value=before, after_value=after))


def _delete_orphan_entities(db: Session) -> None:
    db.execute(delete(Person).where(~select(DocumentPerson.id).where(DocumentPerson.person_id == Person.id).exists()))
    db.execute(delete(PropertyRecord).where(~select(DocumentProperty.id).where(DocumentProperty.property_id == PropertyRecord.id).exists()))


@app.get("/api/health")
def health():
    from .pdf_utils import fitz
    ocr_status = get_ocr_status()
    versions = {}
    for package in ("paddleocr", "docling"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "status": "ok",
        "ocr_configured": ocr_status["configured"],
        "ocr_engines_available": ocr_status["engines"],
        "ocr_languages_available": ocr_status["languages"],
        "ocr_error": ocr_status["error"],
        "pdf_renderer_available": fitz is not None,
        "embedded_scan_image_fallback": True,
        "direct_image_upload": True,
        "document_engines": {
            "pp_structure": {"available": versions["paddleocr"] is not None, "version": versions["paddleocr"]},
            "docling": {"available": versions["docling"] is not None, "version": versions["docling"]},
            "processing_local": True,
            "model_weights_may_download_on_first_use": True,
        },
    }


@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    return {
        "ingestions": db.scalar(select(func.count()).select_from(Ingestion)) or 0,
        "pages": db.scalar(select(func.count()).select_from(Page)) or 0,
        "documents": db.scalar(select(func.count()).select_from(LogicalDocument)) or 0,
        "cases": db.scalar(select(func.count()).select_from(CaseRecord)) or 0,
        "documents_requiring_review": db.scalar(select(func.count()).select_from(LogicalDocument).where(LogicalDocument.review_status == "needs_review")) or 0,
        "unclassified_pages": db.scalar(select(func.count()).select_from(Page).where(Page.category == "unknown")) or 0,
        "failed_jobs": db.scalar(select(func.count()).select_from(ProcessingJob).where(ProcessingJob.status == "failed")) or 0,
    }


@app.get("/api/ingestions")
def list_ingestions(db: Session = Depends(get_db)):
    rows = db.scalars(select(Ingestion).order_by(Ingestion.created_at.desc()).limit(100)).all()
    jobs = {
        job.ingestion_id: job
        for job in db.scalars(
            select(ProcessingJob).where(ProcessingJob.ingestion_id.in_([row.id for row in rows]))
        ).all()
    }
    return [
        {"id": row.id, "filename": row.filename, "source_type": row.source_type, "sha256": row.sha256, "file_size": row.file_size, "page_count": row.page_count, "status": row.status, "progress": jobs[row.id].progress if row.id in jobs else row.progress, "stage": jobs[row.id].stage if row.id in jobs else "unknown", "error": row.error}
        for row in rows
    ]


@app.post("/api/ingestions", status_code=202)
async def upload_pdf(
    file: UploadFile = File(...),
    processing_mode: Literal["existing", "pp_structure", "docling", "compare"] = Form(default="existing"),
    db: Session = Depends(get_db),
):
    name = Path(file.filename or "").name
    if Path(name).suffix.casefold() != ".pdf":
        raise HTTPException(400, "Only .pdf files are accepted.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(400, "The uploaded file must have PDF content type.")
    storage_dir = Path(settings.storage_dir).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)
    ingestion_id = str(uuid4())
    target_dir = (storage_dir / ingestion_id).resolve()
    if storage_dir not in target_dir.parents:
        raise HTTPException(400, "Invalid storage path.")
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / "source.pdf"
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"PDF exceeds the {settings.max_upload_mb} MB upload limit.")
                digest.update(chunk)
                stream.write(chunk)
        with target.open("rb") as stream:
            signature = stream.read(5)
        if size < 8 or signature != b"%PDF-":
            raise HTTPException(400, "File signature is not a valid PDF.")
        file_hash = digest.hexdigest()
        duplicate = db.scalar(select(Ingestion).where(Ingestion.sha256 == file_hash))
        if duplicate:
            target.unlink(missing_ok=True)
            target_dir.rmdir()
            comparison = None
            if processing_mode != "existing":
                comparison = create_comparison(db, duplicate, processing_mode)
                db.commit()
                db.refresh(comparison)
                executor.submit(run_comparison, comparison.id, SessionLocal)
            return JSONResponse(
                status_code=200,
                content={
                    "id": duplicate.id,
                    "status": duplicate.status,
                    "duplicate": True,
                    "comparison_run_id": comparison.id if comparison else None,
                    "message": "This exact PDF was already ingested.",
                },
            )
        try:
            with LocalPdf(target) as pdf:
                page_count = pdf.page_count
                if page_count == 0 or page_count > settings.max_pdf_pages:
                    raise HTTPException(400, f"PDF page count must be between 1 and {settings.max_pdf_pages}.")
        except (PdfReadError, ValueError, OSError) as exc:
            raise HTTPException(400, "PDF could not be opened or is password-protected.") from exc
        ingestion = Ingestion(id=ingestion_id, filename=name[:255], sha256=file_hash, file_size=size, page_count=page_count, status="queued", storage_path=str(target))
        db.add(ingestion)
        job = ProcessingJob(ingestion_id=ingestion_id)
        db.add(job)
        comparison = None
        if processing_mode != "existing":
            comparison = create_comparison(db, ingestion, processing_mode)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            duplicate = db.scalar(select(Ingestion).where(Ingestion.sha256 == file_hash))
            if not duplicate:
                raise
            target.unlink(missing_ok=True)
            target_dir.rmdir()
            comparison = None
            if processing_mode != "existing":
                comparison = create_comparison(db, duplicate, processing_mode)
                db.commit()
                db.refresh(comparison)
                executor.submit(run_comparison, comparison.id, SessionLocal)
            return JSONResponse(
                status_code=200,
                content={
                    "id": duplicate.id,
                    "status": duplicate.status,
                    "duplicate": True,
                    "comparison_run_id": comparison.id if comparison else None,
                    "message": "This exact PDF was already ingested.",
                },
            )
        db.close()
        if comparison:
            executor.submit(_process_ingestion_with_comparison, ingestion_id, comparison.id)
        else:
            executor.submit(process_ingestion, ingestion_id, SessionLocal)
        return {
            "id": ingestion_id,
            "status": "queued",
            "page_count": page_count,
            "job_id": job.id,
            "comparison_run_id": comparison.id if comparison else None,
        }
    except Exception:
        try:
            if target.exists():
                target.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete failed upload %s while cleaning up a rejected PDF.", target)
        try:
            if target_dir.exists():
                target_dir.rmdir()
        except OSError:
            pass
        raise


@app.post("/api/ingestions/images", status_code=202)
async def upload_images(
    files: list[UploadFile] = File(...),
    bundle_name: str = Form(default=""),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(400, "Select at least one PNG, JPG, or JPEG image.")
    if len(files) > settings.max_pdf_pages:
        raise HTTPException(400, f"Image batch cannot exceed {settings.max_pdf_pages} pages.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    image_batch: list[tuple[str, str, bytes]] = []
    digest = hashlib.sha256(b"casework-image-bundle-v1")
    total_size = 0
    try:
        for index, file in enumerate(files, 1):
            name = Path(file.filename or "").name
            suffix = Path(name).suffix.casefold()
            if suffix not in {".png", ".jpg", ".jpeg"}:
                raise HTTPException(400, f"Image {index}: only PNG, JPG, and JPEG files are accepted.")
            if file.content_type not in {"image/png", "image/jpeg", "application/octet-stream"}:
                raise HTTPException(400, f"Image {index}: unsupported content type.")
            data = await file.read(max_bytes - total_size + 1)
            if not data:
                raise HTTPException(400, f"Image {index} is empty.")
            total_size += len(data)
            if total_size > max_bytes:
                raise HTTPException(413, f"Image batch exceeds the {settings.max_upload_mb} MB upload limit.")
            try:
                validated_suffix = validate_image(data, name, file.content_type)
            except ValueError as exc:
                raise HTTPException(400, f"Image {index}: {exc}") from exc
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
            image_name = f"page-{index}-image-0{validated_suffix}"
            image_batch.append((image_name, validated_suffix, data))

        file_hash = digest.hexdigest()
        duplicate = db.scalar(select(Ingestion).where(Ingestion.sha256 == file_hash))
        if duplicate:
            return JSONResponse(
                status_code=200,
                content={
                    "id": duplicate.id,
                    "status": duplicate.status,
                    "duplicate": True,
                    "page_count": duplicate.page_count,
                    "source_type": duplicate.source_type,
                    "message": "This exact ordered image batch was already ingested.",
                },
            )

        storage_dir = Path(settings.storage_dir).resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
        ingestion_id = str(uuid4())
        target_dir = (storage_dir / ingestion_id).resolve()
        if storage_dir not in target_dir.parents:
            raise HTTPException(400, "Invalid storage path.")
        page_image_dir = target_dir / "page_images"
        page_image_dir.mkdir(parents=True, exist_ok=False)
        for image_name, _, data in image_batch:
            (page_image_dir / image_name).write_bytes(data)
        manifest_path = target_dir / "image-manifest.json"
        manifest_path.write_text(
            json.dumps([image_name for image_name, _, _ in image_batch]),
            encoding="utf-8",
        )
        display_name = bundle_name.strip()
        if not display_name:
            display_name = f"Image bundle ({len(image_batch)} pages)"
        ingestion = Ingestion(
            id=ingestion_id,
            filename=Path(display_name).name[:255],
            sha256=file_hash,
            file_size=total_size,
            page_count=len(image_batch),
            status="queued",
            storage_path=str(manifest_path),
            source_type="image",
        )
        db.add(ingestion)
        job = ProcessingJob(ingestion_id=ingestion_id)
        db.add(job)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            duplicate = db.scalar(select(Ingestion).where(Ingestion.sha256 == file_hash))
            if not duplicate:
                raise
            shutil.rmtree(target_dir, ignore_errors=True)
            return JSONResponse(
                status_code=200,
                content={
                    "id": duplicate.id,
                    "status": duplicate.status,
                    "duplicate": True,
                    "page_count": duplicate.page_count,
                    "source_type": duplicate.source_type,
                    "message": "This exact ordered image batch was already ingested.",
                },
            )
        db.close()
        executor.submit(process_ingestion, ingestion_id, SessionLocal)
        return {
            "id": ingestion_id,
            "status": "queued",
            "page_count": len(image_batch),
            "source_type": "image",
            "job_id": job.id,
        }
    except HTTPException:
        if "target_dir" in locals():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except Exception:
        if "target_dir" in locals():
            shutil.rmtree(target_dir, ignore_errors=True)
        logger.exception("Image batch ingestion could not be created")
        raise
    finally:
        for file in files:
            await file.close()


@app.get("/api/ingestions/{ingestion_id}")
def ingestion_detail(ingestion_id: str, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if not ingestion:
        raise HTTPException(404, "Ingestion not found.")
    job = db.scalar(select(ProcessingJob).where(ProcessingJob.ingestion_id == ingestion_id))
    return {"id": ingestion.id, "filename": ingestion.filename, "source_type": ingestion.source_type, "page_count": ingestion.page_count, "status": ingestion.status, "progress": job.progress if job else 0, "stage": job.stage if job else "unknown", "error": ingestion.error}


@app.post("/api/ingestions/{ingestion_id}/comparisons", status_code=202)
def create_ingestion_comparison(ingestion_id: str, payload: ComparisonRequest, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if not ingestion:
        raise HTTPException(404, "Ingestion not found.")
    if ingestion.source_type != "pdf":
        raise HTTPException(400, "Document-engine comparisons require an uploaded PDF.")
    try:
        comparison = create_comparison(db, ingestion, payload.mode)
        db.commit()
        db.refresh(comparison)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(400, str(exc)) from exc
    executor.submit(run_comparison, comparison.id, SessionLocal)
    return {"id": comparison.id, "status": comparison.status, "mode": comparison.mode}


@app.get("/api/comparisons")
def list_comparisons(db: Session = Depends(get_db)):
    rows = db.scalars(
        select(ComparisonRun)
        .options(selectinload(ComparisonRun.engines))
        .order_by(ComparisonRun.created_at.desc())
        .limit(100)
    ).all()
    return [_comparison_summary(row) for row in rows]


@app.get("/api/comparisons/{comparison_id}")
def comparison_detail(comparison_id: str, db: Session = Depends(get_db)):
    comparison = db.scalar(
        select(ComparisonRun)
        .options(selectinload(ComparisonRun.engines))
        .where(ComparisonRun.id == comparison_id)
    )
    if not comparison:
        raise HTTPException(404, "Comparison run not found.")
    engines = []
    for result in comparison.engines:
        engines.append(
            {
                "engine": result.engine,
                "status": result.status,
                "package_version": result.package_version,
                "config": json.loads(result.config_json) if result.config_json else {},
                "confidence": json.loads(result.confidence_json) if result.confidence_json else {
                    "availability": "not_available",
                    "reason": "The engine has not reported confidence metadata.",
                },
                "processing_seconds": result.processing_seconds,
                "page_count": result.page_count,
                "error": result.error,
                "normalized": _read_json_artifact(result.normalized_json_path),
                "native_output_url": f"/comparisons/{comparison.id}/artifacts/{result.engine}/native.json" if result.output_json_path else None,
                "markdown_url": f"/comparisons/{comparison.id}/artifacts/{result.engine}/document.md" if result.markdown_path else None,
            }
        )
    return {
        **_comparison_summary(comparison),
        "page_comparison": _read_json_artifact(comparison.page_comparison_json_path),
        "page_comparison_url": f"/comparisons/{comparison.id}/artifacts/comparison/page-comparison.json" if comparison.page_comparison_json_path else None,
        "engines": engines,
    }


@app.get("/api/comparisons/{comparison_id}/artifacts/{engine}/{artifact_name}")
def comparison_artifact(
    comparison_id: str,
    engine: Literal["pp_structure", "docling", "comparison"],
    artifact_name: Literal["native.json", "normalized.json", "document.md", "page-comparison.json"],
    db: Session = Depends(get_db),
):
    comparison = db.get(ComparisonRun, comparison_id)
    if not comparison:
        raise HTTPException(404, "Comparison run not found.")
    if engine == "comparison":
        if artifact_name != "page-comparison.json":
            raise HTTPException(404, "Comparison artifact not found.")
        relative_path = comparison.page_comparison_json_path
    else:
        result = db.scalar(
            select(EngineProcessingResult).where(
                EngineProcessingResult.comparison_id == comparison_id,
                EngineProcessingResult.engine == engine,
            )
        )
        if not result:
            raise HTTPException(404, "Engine result not found.")
        relative_path = {
            "native.json": result.output_json_path,
            "normalized.json": result.normalized_json_path,
            "document.md": result.markdown_path,
            "page-comparison.json": None,
        }[artifact_name]
    artifact = _stored_artifact(relative_path)
    if artifact is None:
        raise HTTPException(404, "Comparison artifact is not available.")
    media_type = "text/markdown" if artifact.suffix == ".md" else "application/json"
    return FileResponse(artifact, media_type=media_type, filename=artifact.name)


@app.get("/api/ingestions/{ingestion_id}/source")
def ingestion_source(ingestion_id: str, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if not ingestion:
        raise HTTPException(404, "Ingestion not found.")
    source = Path(ingestion.storage_path).resolve()
    if ingestion.source_type == "image":
        if source.parent.parent != Path(settings.storage_dir).resolve():
            raise HTTPException(404, "Uploaded source images are unavailable.")
        try:
            image_bundle = LocalImageBundle(source)
            first_image = image_bundle.source_image_path(0)
        except (OSError, ValueError):
            raise HTTPException(404, "Uploaded source images are unavailable.")
        return FileResponse(
            first_image,
            media_type=mimetypes.guess_type(first_image.name)[0] or "application/octet-stream",
            filename=first_image.name,
            headers={"Cache-Control": "no-store"},
        )
    if not source.is_file() or source.parent.parent != Path(settings.storage_dir).resolve():
        raise HTTPException(404, "Source PDF is unavailable.")
    return FileResponse(source, media_type="application/pdf", filename="source.pdf", content_disposition_type="inline", headers={"Cache-Control": "no-store"})


@app.get("/api/ingestions/{ingestion_id}/report.pdf")
def ingestion_report_pdf(ingestion_id: str, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if ingestion is None:
        raise HTTPException(404, "Ingestion not found.")
    try:
        report = build_ingestion_report(db, ingestion_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except IngestionNotCompleteError as exc:
        raise HTTPException(409, str(exc)) from exc
    except SourceChangedError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except OSError:
        logger.exception("Could not read the source document for its formatted report")
        raise HTTPException(500, "Could not read the original source to build its formatted report.")
    except (PdfReadError, RuntimeError):
        logger.exception("Could not build formatted PDF report", extra={"ingestion_id": ingestion_id})
        raise HTTPException(500, "Could not build the formatted PDF report; see the local server log.")
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(ingestion.filename).stem).strip(".-") or ingestion_id
    return Response(
        content=report,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}-formatted-report.pdf"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/ingestions/{ingestion_id}/retry", status_code=202)
def retry_ingestion(ingestion_id: str, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if not ingestion:
        raise HTTPException(404, "Ingestion not found.")
    claimed = db.execute(
        update(Ingestion)
        .where(Ingestion.id == ingestion_id, Ingestion.status.in_(("failed", "completed_with_warnings")))
        .values(status="queued", progress=0, error=None)
    )
    if claimed.rowcount != 1:
        raise HTTPException(409, "Only failed or warning-status ingestions can be retried.")
    case_ids = set(db.scalars(select(LogicalDocument.case_id).where(LogicalDocument.ingestion_id == ingestion_id, LogicalDocument.case_id.is_not(None))).all())
    db.execute(delete(ValidationFinding).where(ValidationFinding.document_id.in_(select(LogicalDocument.id).where(LogicalDocument.ingestion_id == ingestion_id))))
    db.execute(delete(LogicalDocument).where(LogicalDocument.ingestion_id == ingestion_id))
    db.execute(delete(Page).where(Page.ingestion_id == ingestion_id))
    for case_id in case_ids:
        case = db.get(CaseRecord, case_id)
        if case and not db.scalar(select(LogicalDocument.id).where(LogicalDocument.case_id == case_id).limit(1)):
            db.delete(case)
    db.flush()
    _delete_orphan_entities(db)
    job = db.scalar(select(ProcessingJob).where(ProcessingJob.ingestion_id == ingestion_id))
    if not job:
        job = ProcessingJob(ingestion_id=ingestion_id)
        db.add(job)
    job.status = ingestion.status = "queued"
    job.stage = "queued"
    job.progress = ingestion.progress = 0
    job.error = ingestion.error = None
    db.commit()
    db.close()
    executor.submit(process_ingestion, ingestion_id, SessionLocal)
    return {"id": ingestion_id, "status": "queued", "job_id": job.id}


@app.post("/api/ingestions/{ingestion_id}/reprocess-fields")
def reprocess_ingestion_fields(ingestion_id: str, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if not ingestion:
        raise HTTPException(404, "Ingestion not found.")
    if ingestion.status in {"queued", "processing"}:
        raise HTTPException(409, "Wait until ingestion processing is complete before refreshing its extracted fields.")
    summary = refresh_ingestion_extraction(db, ingestion_id)
    db.commit()
    return {"id": ingestion_id, "status": ingestion.status, **summary}


def _parse_identification_documents(raw: Any) -> dict[str, Any]:
    empty_grouped: dict[str, list[dict[str, Any]]] = {"pan_card": [], "aadhaar_card": [], "other_id": []}
    data: Any = raw
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
    if not isinstance(data, dict):
        return {
            "status": "not_found",
            "summary_counts": {"pan_card": 0, "aadhaar_card": 0, "other_id": 0, "total": 0},
            "grouped_documents": empty_grouped,
            "pan_card": [],
            "aadhaar_card": [],
            "other_id": [],
        }
    grouped_src = data.get("grouped_documents") if isinstance(data.get("grouped_documents"), dict) else data
    pan_list = list(grouped_src.get("pan_card") or data.get("pan_card") or [])
    aadhaar_list = list(grouped_src.get("aadhaar_card") or data.get("aadhaar_card") or [])
    other_list = list(grouped_src.get("other_id") or data.get("other_id") or [])
    total = len(pan_list) + len(aadhaar_list) + len(other_list)
    status = data.get("status")
    if status not in ("found", "not_found", "uncertain"):
        if total == 0:
            status = "not_found"
        else:
            all_items = [*pan_list, *aadhaar_list, *other_list]
            has_definite = any(
                (item.get("id_number") and item.get("id_number") != "[Aadhaar Redacted]")
                or item.get("confidence") in ("high", "medium")
                for item in all_items
            )
            status = "found" if has_definite else "uncertain"
    return {
        "status": status,
        "summary_counts": {
            "pan_card": len(pan_list),
            "aadhaar_card": len(aadhaar_list),
            "other_id": len(other_list),
            "total": total,
        },
        "grouped_documents": {
            "pan_card": pan_list,
            "aadhaar_card": aadhaar_list,
            "other_id": other_list,
        },
        "pan_card": pan_list,
        "aadhaar_card": aadhaar_list,
        "other_id": other_list,
    }


def _serialize_stamp_duty_assessment(assessment: StampDutyAssessment | None) -> dict | None:
    if not assessment:
        return None
    source_pages: list[int] = []
    if assessment.source_pages:
        for p in str(assessment.source_pages).replace(";", ",").split(","):
            p_str = p.strip()
            if p_str.isdigit():
                source_pages.append(int(p_str))

    evidence_val: Any = assessment.evidence
    if isinstance(evidence_val, str) and evidence_val.strip().startswith("{") and evidence_val.strip().endswith("}"):
        try:
            evidence_val = json.loads(evidence_val)
        except Exception:
            evidence_val = {"summary": assessment.evidence}
    elif isinstance(evidence_val, str) and evidence_val:
        evidence_val = {"summary": evidence_val}

    calc_components: Any = assessment.calculation_components
    if isinstance(calc_components, str) and calc_components.strip().startswith("{"):
        try:
            calc_components = json.loads(calc_components)
        except Exception:
            pass

    source_ev: Any = assessment.source_evidence
    if isinstance(source_ev, str) and source_ev.strip().startswith("{"):
        try:
            source_ev = json.loads(source_ev)
        except Exception:
            pass

    return {
        "id": assessment.id,
        "document_id": assessment.document_id,
        "donor_name": assessment.donor_name,
        "donee_name": assessment.donee_name,
        "donor_role": assessment.donor_role,
        "donee_role": assessment.donee_role,
        "donee_relation_to_donor": assessment.donee_relation_to_donor,
        "property_type": assessment.property_type or "unknown",
        "property_type_uncertain": bool(assessment.property_type_uncertain),
        "cess_uncertain": bool(assessment.cess_uncertain),
        "location_band": assessment.location_band,
        "property_description": assessment.property_description,
        "property_location": assessment.property_location,
        "survey_number": assessment.survey_number,
        "area": assessment.area,
        "relationship": assessment.relationship,
        "relationship_normalized": assessment.relationship_normalized,
        "relationship_eligibility": assessment.relationship_eligibility or "unknown",
        "jurisdiction": assessment.jurisdiction or "unknown",
        "selected_rule": assessment.selected_rule or "review_required",
        "duty_rate": assessment.duty_rate,
        "cess_rule": assessment.cess_rule,
        "rule_version": assessment.rule_version or assessment.legal_rule_version or "MH-ART34-2024.1",
        "effective_date": assessment.effective_date or "2017-05-16",
        "calculation_components": calc_components,
        "source_evidence": source_ev,
        "identification_documents": _parse_identification_documents(getattr(assessment, "identification_documents", None)),
        "market_value": assessment.market_value,
        "base_stamp_duty": assessment.base_duty,
        "metro_cess_rate": assessment.metro_cess_rate,
        "metro_cess": assessment.metro_cess_amount,
        "expected_total": assessment.expected_total,
        "stamp_duty_paid": assessment.stamp_duty_paid,
        "metro_cess_paid": assessment.metro_cess_paid,
        "total_stamp_duty_paid": assessment.total_stamp_duty_paid,
        "challan_number": assessment.challan_number,
        "difference": assessment.difference,
        "status": assessment.status or "requires_review",
        "legal_rule_version": assessment.legal_rule_version or "MH-ART34-2024.1",
        "evidence": evidence_val or {},
        "source_pages": source_pages,
        "review_status": assessment.review_status or "unreviewed",
        "reviewer_notes": assessment.reviewer_notes,
    }


def recalculate_and_sync_stamp_duty_assessment(
    db: Session,
    assessment: StampDutyAssessment,
    doc: LogicalDocument,
    explicit_status: str | None = None,
    skip_field_id: str | None = None,
) -> None:
    rel_elig = assessment.relationship_eligibility or "unknown"
    prop_type = assessment.property_type or "unknown"
    prop_uncertain = bool(assessment.property_type_uncertain)
    mv = assessment.market_value
    donee_rel = assessment.donee_relation_to_donor or assessment.relationship_normalized

    # Determine location band and cess applicability strictly without loose assumptions
    location_band = assessment.location_band
    jurisdiction = assessment.jurisdiction
    cess_uncertain = assessment.cess_uncertain

    if not location_band or cess_uncertain or not jurisdiction or jurisdiction == "unknown":
        resolved_band, resolved_jur, resolved_uncertain, _ = resolve_authoritative_location_band(
            prop_loc=assessment.property_location,
            schedule_text=assessment.property_description,
            reg_summary=None,
        )
        if not location_band:
            location_band = resolved_band
        if not jurisdiction or jurisdiction == "unknown":
            jurisdiction = resolved_jur
        cess_uncertain = resolved_uncertain

    assessment.location_band = location_band
    assessment.jurisdiction = jurisdiction
    assessment.cess_uncertain = cess_uncertain

    if assessment.selected_rule in ("article_34_200", "article_34_3_percent", "article_25", "review_required"):
        selected_rule = assessment.selected_rule
    else:
        selected_rule = select_stamp_duty_rule(
            donee_relation_to_donor=donee_rel,
            property_type=prop_type,
            property_type_uncertain=prop_uncertain,
            is_relationship_ambiguous=(rel_elig == "ambiguous"),
        )
        assessment.selected_rule = selected_rule

    if assessment.total_stamp_duty_paid is None and (assessment.stamp_duty_paid is not None or assessment.metro_cess_paid is not None):
        assessment.total_stamp_duty_paid = (assessment.stamp_duty_paid or 0) + (assessment.metro_cess_paid or 0)
    total_paid = assessment.total_stamp_duty_paid

    base_duty: int | None = None
    duty_rate: str | None = None
    cess_rule: str | None = None
    metro_cess_rate: float | None = None
    metro_cess_amount: int | None = None
    expected_total: int | None = None

    # Evaluate cess strictly on verified location; no automatic exemption for agricultural property
    if cess_uncertain:
        metro_cess_rate = None
        metro_cess_amount = None
        cess_rule = "jurisdiction_uncertain"
    elif jurisdiction == "metro":
        metro_cess_rate = float(STAMP_DUTY_METRO_CESS_RATE)
        metro_cess_amount = (
            int((Decimal(mv) * Decimal(str(metro_cess_rate))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            if mv is not None
            else None
        )
        cess_rule = "metro_cess_1_percent"
    else:  # non_metro
        metro_cess_rate = 0.0
        metro_cess_amount = 0
        cess_rule = "non_metro_exempt"

    if selected_rule == "article_34_200":
        duty_rate = "200_fixed"
        base_duty = int(STAMP_DUTY_BASE_AMOUNT)
        if cess_uncertain:
            expected_total = None
        else:
            expected_total = base_duty + (metro_cess_amount or 0)

    elif selected_rule == "article_34_3_percent":
        duty_rate = "3%"
        if mv is not None:
            base_duty = int((Decimal(mv) * Decimal("0.03")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            if cess_uncertain:
                expected_total = None
            else:
                expected_total = base_duty + (metro_cess_amount or 0)
        else:
            base_duty = None
            expected_total = None

    elif selected_rule == "article_25":
        if location_band == "municipal_corporation_mmr_influence":
            duty_rate = "5%"
            if mv is not None:
                base_duty = int((Decimal(mv) * Decimal("0.05")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            else:
                base_duty = None
        elif location_band == "gram_panchayat_other":
            duty_rate = "4%"
            if mv is not None:
                base_duty = int((Decimal(mv) * Decimal("0.04")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            else:
                base_duty = None
        else:
            duty_rate = "review_required"
            base_duty = None

        if base_duty is not None and not cess_uncertain:
            expected_total = base_duty + (metro_cess_amount or 0)
        else:
            expected_total = None

    else:  # review_required
        duty_rate = "review_required"
        if rel_elig in ("article_34_200", "eligible"):
            base_duty = int(STAMP_DUTY_BASE_AMOUNT)
        if not cess_uncertain and mv is not None and metro_cess_amount is not None:
            if base_duty:
                expected_total = base_duty + metro_cess_amount

    assessment.selected_rule = selected_rule
    assessment.duty_rate = duty_rate
    assessment.cess_rule = cess_rule
    assessment.base_duty = base_duty
    assessment.metro_cess_rate = metro_cess_rate
    assessment.metro_cess_amount = metro_cess_amount
    assessment.expected_total = expected_total

    if explicit_status:
        assessment.status = explicit_status
        if total_paid is not None and expected_total is not None:
            assessment.difference = total_paid - expected_total
        else:
            assessment.difference = None
    elif selected_rule == "review_required":
        if rel_elig == "unknown":
            assessment.status = "not_available"
        else:
            assessment.status = "requires_review"
        assessment.difference = None
    elif selected_rule == "article_25":
        art25_reasons = []
        if not location_band:
            art25_reasons.append("uncertain location band")
        if cess_uncertain:
            art25_reasons.append("cess uncertain")
        if mv is None:
            art25_reasons.append("missing market value")
        if prop_uncertain or prop_type in ("unknown", "ambiguous"):
            art25_reasons.append("uncertain property type")
        if rel_elig in ("unknown", "ambiguous"):
            art25_reasons.append("ambiguous relationship")
        if art25_reasons:
            assessment.status = "requires_review"
            assessment.difference = None
        else:
            assessment.status = "not_applicable"
            if total_paid is not None and expected_total is not None:
                assessment.difference = total_paid - expected_total
            else:
                assessment.difference = None
    elif cess_uncertain:
        assessment.status = "requires_review"
        assessment.difference = None
    elif mv is None or expected_total is None:
        assessment.status = "not_available"
        assessment.difference = None
    elif total_paid is None:
        assessment.status = "not_available"
        assessment.difference = None
    else:
        diff = total_paid - expected_total
        assessment.difference = diff
        assessment.status = "compliant" if diff >= 0 else "short_paid"

    calc_components = {
        "selected_rule": assessment.selected_rule,
        "duty_rate": assessment.duty_rate,
        "base_duty_amount": assessment.base_duty,
        "market_value": assessment.market_value,
        "location_band": assessment.location_band,
        "cess_uncertain": assessment.cess_uncertain,
        "cess_rule": assessment.cess_rule,
        "metro_cess_rate": assessment.metro_cess_rate,
        "metro_cess_amount": assessment.metro_cess_amount,
        "expected_total": assessment.expected_total,
        "total_stamp_duty_paid": total_paid,
        "difference": assessment.difference,
    }
    assessment.calculation_components = json.dumps(calc_components)

    def sync_field(name: str, val: str | None):
        f = db.scalar(
            select(ExtractedField).where(
                ExtractedField.document_id == doc.id, ExtractedField.field_name == name
            )
        )
        if not f:
            if val is not None:
                f = ExtractedField(
                    document_id=doc.id,
                    field_name=name,
                    extracted_value=val,
                    verified_value=val,
                    normalized_value=val,
                    source_page=doc.start_page,
                    evidence="Synchronized from Article 34 Stamp Duty Assessment.",
                    extraction_method="reviewer_sync",
                    verification_status="verified",
                )
                db.add(f)
        else:
            if f.id != skip_field_id:
                f.verified_value = val
                f.verification_status = "verified"
                f.normalized_value = val

    sync_field("market_value", str(assessment.market_value) if assessment.market_value is not None else None)
    sync_field("property_type", assessment.property_type if assessment.property_type != "unknown" else None)
    sync_field("property_description", assessment.property_description)
    sync_field("property_location", assessment.property_location)
    sync_field("survey_number", assessment.survey_number)
    sync_field("area", assessment.area)
    sync_field("stamp_duty_paid", str(assessment.stamp_duty_paid) if assessment.stamp_duty_paid is not None else None)
    sync_field("metro_cess_paid", str(assessment.metro_cess_paid) if assessment.metro_cess_paid is not None else None)
    sync_field("total_stamp_duty_paid", str(assessment.total_stamp_duty_paid) if assessment.total_stamp_duty_paid is not None else None)
    sync_field("expected_stamp_duty", str(assessment.expected_total) if assessment.expected_total is not None else None)
    sync_field("stamp_duty_difference", str(assessment.difference) if assessment.difference is not None else None)
    sync_field("stamp_duty_status", assessment.status)
    sync_field("relationship_eligibility", assessment.relationship_eligibility)
    if assessment.challan_number:
        sync_field("challan_number", assessment.challan_number)
    if assessment.relationship:
        sync_field("donor_donee_relationship", assessment.relationship)


def _doc_dict(row: LogicalDocument):
    serialized_assessment = _serialize_stamp_duty_assessment(getattr(row, "stamp_duty_assessment", None))
    return {
        "id": row.id,
        "ingestion_id": row.ingestion_id,
        "source_type": row.ingestion.source_type,
        "case_id": row.case_id,
        "start_page": row.start_page,
        "end_page": row.end_page,
        "category": row.category,
        "predicted_category": row.predicted_category,
        "confidence_proxy": row.confidence_proxy,
        "classification_evidence": row.classification_evidence,
        "review_status": row.review_status,
        "case_link_status": row.case_link_status,
        "case_link_evidence": row.case_link_evidence,
        "fields": [_field_dict(field) for field in row.fields],
        "stamp_duty_assessment": serialized_assessment,
        "identification_documents": (
            serialized_assessment["identification_documents"]
            if serialized_assessment
            else _parse_identification_documents(None)
        ),
    }


def _field_dict(field: ExtractedField, mask_sensitive: bool = True):
    value = field.verified_value if field.verified_value is not None else field.extracted_value
    availability = "not_available" if field.verification_status == "not_available" else "available"
    display_value = "Not available" if availability == "not_available" else value
    if mask_sensitive and field.sensitive and value:
        value = f"••••{value[-4:]}" if len(value) > 4 else "••••"
        display_value = value
    extracted_value = field.extracted_value
    verified_value = field.verified_value
    if mask_sensitive and field.sensitive:
        extracted_value = f"••••{extracted_value[-4:]}" if extracted_value and len(extracted_value) > 4 else ("••••" if extracted_value else None)
        verified_value = f"••••{verified_value[-4:]}" if verified_value and len(verified_value) > 4 else ("••••" if verified_value else None)
    return {"id": field.id, "field_name": field.field_name, "value": value, "display_value": display_value, "availability": availability, "extracted_value": extracted_value, "verified_value": verified_value, "source_page": field.source_page, "evidence": "[REDACTED]" if mask_sensitive and field.sensitive else field.evidence, "verification_status": field.verification_status, "sensitive": field.sensitive}


@app.get("/api/documents")
def list_documents(
    q: str = Query(default="", max_length=100),
    category: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    filters = []
    if category:
        filters.append(LogicalDocument.category == category)
    if q:
        match = f"%{q}%"
        filters.append(
            or_(
                LogicalDocument.id.ilike(match),
                LogicalDocument.category.ilike(match),
                LogicalDocument.classification_evidence.ilike(match),
                LogicalDocument.id.in_(
                    select(ExtractedField.document_id).where(
                        or_(ExtractedField.extracted_value.ilike(match), ExtractedField.evidence.ilike(match))
                    )
                ),
                LogicalDocument.ingestion_id.in_(
                    select(Page.ingestion_id).where(
                        or_(Page.ocr_text_original.ilike(match), Page.ocr_text_corrected.ilike(match))
                    )
                ),
            )
        )
    base = select(LogicalDocument)
    if filters:
        base = base.where(*filters)
    stmt = (
        base.options(selectinload(LogicalDocument.fields), selectinload(LogicalDocument.stamp_duty_assessment))
        .order_by(LogicalDocument.start_page)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = db.scalars(stmt).all()
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    return {"items": [_doc_dict(row) for row in rows], "total": total, "page": page, "page_size": page_size}


@app.get("/api/documents/{document_id}")
def document_detail(document_id: str, db: Session = Depends(get_db)):
    row = db.scalar(
        select(LogicalDocument)
        .options(selectinload(LogicalDocument.fields), selectinload(LogicalDocument.stamp_duty_assessment))
        .where(LogicalDocument.id == document_id)
    )
    if not row:
        raise HTTPException(404, "Document not found.")
    pages = db.scalars(select(Page).where(Page.ingestion_id == row.ingestion_id, Page.page_number >= row.start_page, Page.page_number <= row.end_page).order_by(Page.page_number)).all()
    findings = db.scalars(
        select(ValidationFinding)
        .where(ValidationFinding.document_id == document_id)
        .order_by(ValidationFinding.severity.desc())
    ).all()
    return {
        **_doc_dict(row),
        "findings": [
            {
                "id": f.id,
                "case_id": f.case_id,
                "document_id": f.document_id,
                "rule_id": f.rule_id,
                "severity": f.severity,
                "kind": f.kind,
                "explanation": f.explanation,
                "evidence": f.evidence,
                "source_pages": f.source_pages,
                "review_status": f.review_status,
            }
            for f in findings
        ],
        "pages": [
            {
                "id": page.id,
                "page_number": page.page_number,
                "ocr_text": page.ocr_text_corrected or page.ocr_text_original or page.digital_text or "",
                "ocr_text_original": page.ocr_text_original or page.digital_text or "",
                "ocr_error": page.ocr_error,
                "ocr_engine": page.ocr_engine,
                "ocr_languages": page.ocr_languages,
                "ocr_confidence_proxy": page.ocr_confidence_proxy,
                "ocr_word_boxes": page.ocr_word_boxes,
                "confidence_description": "Mean engine-reported OCR word confidence on its 0-100 scale; not calibrated across engines or a probability of correctness." if page.ocr_confidence_proxy is not None else None,
                "processing_version": page.processing_version,
                "category": page.category,
            }
            for page in pages
        ],
    }


@app.get("/api/documents/{document_id}/audit")
def document_audit(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(LogicalDocument, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    field_ids = db.scalars(select(ExtractedField.id).where(ExtractedField.document_id == doc.id)).all()
    page_ids = db.scalars(select(Page.id).where(Page.ingestion_id == doc.ingestion_id, Page.page_number.between(doc.start_page, doc.end_page))).all()
    entity_ids = [doc.id, *field_ids, *page_ids]
    rows = db.scalars(
        select(AuditEvent)
        .where(AuditEvent.entity_id.in_(entity_ids))
        .order_by(AuditEvent.created_at.desc())
        .limit(100)
    ).all()
    return [
        {
            "id": row.id,
            "entity_type": row.entity_type,
            "action": row.action,
            "before_value": row.before_value,
            "after_value": row.after_value,
            "actor": row.actor,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


@app.get("/api/documents/{document_id}/pages/{page_number}/image")
def page_image(document_id: str, page_number: int, db: Session = Depends(get_db)):
    doc = db.get(LogicalDocument, document_id)
    if not doc or page_number < doc.start_page or page_number > doc.end_page:
        raise HTTPException(404, "Page not found for this document.")
    page = db.scalar(select(Page).where(Page.ingestion_id == doc.ingestion_id, Page.page_number == page_number))
    if not page:
        raise HTTPException(404, "Page not found.")
    image_path = Path(page.image_path).resolve() if page.image_path else None
    storage_root = Path(settings.storage_dir).resolve()
    if image_path and storage_root not in image_path.parents:
        image_path = None
    if image_path and image_path.is_file():
        return Response(image_path.read_bytes(), media_type=mimetypes.guess_type(image_path.name)[0] or "application/octet-stream")
    ingestion = db.get(Ingestion, doc.ingestion_id)
    if ingestion.source_type == "image":
        raise HTTPException(404, "The original uploaded page image is unavailable.")
    try:
        with LocalPdf(ingestion.storage_path) as pdf:
            image = pdf.render_png(page_number - 1, dpi=140)
            if image is None:
                raise HTTPException(503, "PDF rendering is unavailable; open the source PDF in the browser viewer.")
            return Response(image, media_type="image/png")
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "Could not render source page.") from exc


@app.get("/api/documents/{document_id}/pages/{page_number}/table-lines")
def page_table_lines(document_id: str, page_number: int, db: Session = Depends(get_db)):
    doc = db.get(LogicalDocument, document_id)
    if not doc or page_number < doc.start_page or page_number > doc.end_page:
        raise HTTPException(404, "Page not found for this document.")
    page = db.scalar(
        select(Page).where(Page.ingestion_id == doc.ingestion_id, Page.page_number == page_number)
    )
    if not page:
        raise HTTPException(404, "Source page is unavailable for table-line detection.")
    storage_root = Path(settings.storage_dir).resolve()
    try:
        image_path = Path(page.image_path).resolve() if page.image_path else None
        if image_path and storage_root in image_path.parents and image_path.is_file():
            lines = detect_table_lines(image_path)
        else:
            ingestion = db.get(Ingestion, doc.ingestion_id)
            if not ingestion or ingestion.source_type != "pdf":
                raise HTTPException(404, "Source page image is unavailable for table-line detection.")
            source_pdf = Path(ingestion.storage_path).resolve()
            if storage_root not in source_pdf.parents or not source_pdf.is_file() or source_pdf.suffix.casefold() != ".pdf":
                raise HTTPException(404, "Source PDF is unavailable for table-line detection.")
            with LocalPdf(source_pdf) as pdf:
                rendered_page = pdf.render_png(page_number - 1, dpi=140)
            if rendered_page is None:
                raise HTTPException(503, "PDF page rendering is unavailable for table-line detection.")
            lines = detect_table_lines(rendered_page)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Source page image could not be read for table-line detection.") from exc
    except RuntimeError as exc:
        logger.exception("Could not detect source-page table lines", extra={"page_id": page.id})
        raise HTTPException(503, "Could not detect table lines from the source page.") from exc
    except (OSError, ValueError, PdfReadError) as exc:
        logger.exception("Could not render source page for table-line detection", extra={"page_id": page.id})
        raise HTTPException(500, "Could not render the source page for table-line detection.") from exc
    return {"table_lines": lines}


@app.patch("/api/documents/{document_id}/category")
def correct_category(document_id: str, payload: CategoryCorrection, db: Session = Depends(get_db)):
    doc = db.get(LogicalDocument, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    old = doc.category
    doc.category = payload.category
    doc.review_status = "reviewed"
    _audit(db, "document", doc.id, "category_corrected", old, payload.category)
    db.add(ReviewDecision(entity_type="document", entity_id=doc.id, decision="category_corrected"))
    db.commit()
    return _doc_dict(doc)


@app.patch("/api/documents/{document_id}/case-link")
def review_case_link(document_id: str, payload: CaseLinkDecision, db: Session = Depends(get_db)):
    doc = db.get(LogicalDocument, document_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    previous_case = doc.case_id
    if payload.decision == "approved":
        if not payload.case_id:
            raise HTTPException(400, "A case_id is required to approve a case link.")
        if not db.get(CaseRecord, payload.case_id):
            raise HTTPException(404, "Case not found.")
        doc.case_id = payload.case_id
        doc.case_link_status = "approved"
        doc.case_link_evidence = payload.note or doc.case_link_evidence or "Approved by reviewer."
    else:
        doc.case_link_status = "rejected"
        doc.case_link_evidence = payload.note or doc.case_link_evidence
        isolated_case = CaseRecord(title=f"Kept separate, pages {doc.start_page}-{doc.end_page}", status="needs_review")
        db.add(isolated_case)
        db.flush()
        doc.case_id = isolated_case.id
    if previous_case and previous_case != doc.case_id:
        old_case = db.get(CaseRecord, previous_case)
        if old_case and not db.scalar(select(LogicalDocument.id).where(LogicalDocument.case_id == previous_case).limit(1)):
            db.delete(old_case)
        else:
            sync_case_entities(db, previous_case)
    sync_case_entities(db, doc.case_id) if doc.case_id else None
    _audit(db, "document", doc.id, f"case_link_{payload.decision}", previous_case, doc.case_id)
    db.add(ReviewDecision(entity_type="document", entity_id=doc.id, decision=f"case_link_{payload.decision}", note=payload.note))
    db.commit()
    return {"id": doc.id, "case_id": doc.case_id, "case_link_status": doc.case_link_status}


@app.patch("/api/documents/{document_id}/stamp-duty-assessment", response_model=StampDutyAssessmentResponse)
def update_stamp_duty_assessment(
    document_id: str,
    payload: StampDutyAssessmentUpdateRequest,
    db: Session = Depends(get_db),
):
    doc = db.scalar(
        select(LogicalDocument)
        .options(selectinload(LogicalDocument.fields), selectinload(LogicalDocument.stamp_duty_assessment))
        .where(LogicalDocument.id == document_id)
    )
    if not doc:
        raise HTTPException(404, "Document not found.")

    assessment = doc.stamp_duty_assessment
    if not assessment:
        assessment = StampDutyAssessment(document_id=doc.id)
        db.add(assessment)
        doc.stamp_duty_assessment = assessment

    recalc_keys = {
        "market_value",
        "stamp_duty_paid",
        "metro_cess_paid",
        "total_stamp_duty_paid",
        "property_type",
        "property_type_uncertain",
        "donor_role",
        "donee_role",
        "donee_relation_to_donor",
        "relationship",
        "relationship_eligibility",
        "jurisdiction",
        "location_band",
        "cess_uncertain",
        "selected_rule",
    }
    update_data = payload.model_dump(exclude_unset=True)
    needs_recalc = any(k in update_data for k in recalc_keys)

    old_status = assessment.status
    for field_name, value in update_data.items():
        if field_name == "status":
            continue
        setattr(assessment, field_name, value)

    if "relationship" in update_data and "relationship_eligibility" not in update_data:
        if update_data["relationship"]:
            rel_res = evaluate_article_34_relationship(update_data["relationship"])
            assessment.relationship_normalized = rel_res[1]
            assessment.relationship_eligibility = RelationshipEligibility(rel_res[2])
            if getattr(rel_res, "donee_relation_to_donor", None):
                assessment.donee_relation_to_donor = rel_res.donee_relation_to_donor
            if getattr(rel_res, "donor_role", None):
                assessment.donor_role = rel_res.donor_role
            if getattr(rel_res, "donee_role", None):
                assessment.donee_role = rel_res.donee_role
            needs_recalc = True

    explicit_status = update_data.get("status")
    if needs_recalc or explicit_status:
        recalculate_and_sync_stamp_duty_assessment(db, assessment, doc, explicit_status=explicit_status)
    elif "status" in update_data:
        assessment.status = update_data["status"]

    if payload.review_status:
        assessment.review_status = payload.review_status
    elif needs_recalc and assessment.review_status == "unreviewed":
        assessment.review_status = "corrected"

    _audit(
        db,
        "stamp_duty_assessment",
        str(assessment.id or doc.id),
        "assessment_updated",
        old_status,
        assessment.status,
    )
    db.commit()
    db.refresh(assessment)
    return _serialize_stamp_duty_assessment(assessment)


@app.post("/api/documents/{document_id}/split", status_code=201)
def split_document(document_id: str, payload: SplitDocumentRequest, db: Session = Depends(get_db)):
    doc = db.scalar(select(LogicalDocument).options(selectinload(LogicalDocument.fields)).where(LogicalDocument.id == document_id))
    if not doc:
        raise HTTPException(404, "Document not found.")
    split_page = payload.split_before_page
    if split_page <= doc.start_page or split_page > doc.end_page:
        raise HTTPException(400, "The split must be before a page inside the current document, not at either outer edge.")
    before = doc.end_page
    new_doc = LogicalDocument(
        ingestion_id=doc.ingestion_id,
        case_id=doc.case_id,
        start_page=split_page,
        end_page=doc.end_page,
        category=doc.category,
        predicted_category=doc.predicted_category,
        classification_method=doc.classification_method,
        classification_evidence=doc.classification_evidence,
        confidence_proxy=doc.confidence_proxy,
        review_status="needs_review",
        case_link_status=doc.case_link_status,
        case_link_evidence=doc.case_link_evidence,
    )
    db.add(new_doc)
    db.flush()
    doc.end_page = split_page - 1
    for field in list(doc.fields):
        if field.source_page >= split_page:
            field.document_id = new_doc.id
    _audit(db, "document", doc.id, "split", str(before), f"{doc.end_page}; new document {new_doc.id} starts at {split_page}")
    db.add(ReviewDecision(entity_type="document", entity_id=doc.id, decision="split", note=f"New document {new_doc.id} starts on page {split_page}."))
    db.commit()
    return {"original": _doc_dict(doc), "created": _doc_dict(new_doc)}


@app.post("/api/documents/{document_id}/merge")
def merge_documents(document_id: str, payload: MergeDocumentRequest, db: Session = Depends(get_db)):
    doc = db.scalar(select(LogicalDocument).options(selectinload(LogicalDocument.fields)).where(LogicalDocument.id == document_id))
    other = db.scalar(select(LogicalDocument).options(selectinload(LogicalDocument.fields)).where(LogicalDocument.id == payload.other_document_id))
    if not doc or not other:
        raise HTTPException(404, "Both documents must exist.")
    if doc.id == other.id or doc.ingestion_id != other.ingestion_id:
        raise HTTPException(400, "Choose two different documents from the same PDF.")
    if doc.end_page + 1 != other.start_page and other.end_page + 1 != doc.start_page:
        raise HTTPException(400, "Only adjacent page ranges can be merged.")
    if doc.case_id and other.case_id and doc.case_id != other.case_id:
        raise HTTPException(409, "Documents linked to different cases cannot be merged until their case links are reviewed.")
    keep, remove = (doc, other) if doc.start_page < other.start_page else (other, doc)
    keep.start_page = min(doc.start_page, other.start_page)
    keep.end_page = max(doc.end_page, other.end_page)
    if keep.category != remove.category:
        keep.category = "unknown"
    for field in list(remove.fields):
        field.document_id = keep.id
    for relation in db.scalars(select(DocumentPerson).where(DocumentPerson.document_id == remove.id)).all():
        relation.document_id = keep.id
    for relation in db.scalars(select(DocumentProperty).where(DocumentProperty.document_id == remove.id)).all():
        relation.document_id = keep.id
    if not keep.case_id:
        keep.case_id = remove.case_id
        keep.case_link_status = remove.case_link_status
        keep.case_link_evidence = remove.case_link_evidence
    removed_id = remove.id
    db.delete(remove)
    _audit(db, "document", keep.id, "merged", removed_id, keep.id)
    db.add(ReviewDecision(entity_type="document", entity_id=keep.id, decision="merged", note=f"Merged adjacent document {removed_id}."))
    if keep.case_id:
        sync_case_entities(db, keep.case_id)
    db.commit()
    return _doc_dict(keep)


@app.patch("/api/pages/{page_id}/ocr")
def correct_ocr(page_id: str, payload: PageCorrection, db: Session = Depends(get_db)):
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found.")
    old = page.ocr_text_corrected or page.ocr_text_original or ""
    page.ocr_text_corrected = payload.corrected_text
    _audit(db, "page", page.id, "ocr_corrected", f"{len(old)} characters", f"{len(payload.corrected_text)} characters")
    db.commit()
    return {"id": page.id, "corrected": True}


@app.patch("/api/fields/{field_id}")
def correct_field(field_id: str, payload: FieldCorrection, db: Session = Depends(get_db)):
    field = db.get(ExtractedField, field_id)
    if not field:
        raise HTTPException(404, "Field not found.")
    old = field.verified_value if field.verified_value is not None else field.extracted_value
    field.verified_value = payload.value
    field.verification_status = payload.verification_status
    if field.sensitive:
        _audit(db, "field", field.id, "field_corrected", "[REDACTED]", "[REDACTED]")
    else:
        _audit(db, "field", field.id, "field_corrected", old, payload.value)
    db.add(ReviewDecision(entity_type="field", entity_id=field.id, decision=payload.verification_status))

    doc = db.get(LogicalDocument, field.document_id)
    if doc and (doc.category == "gift_deed" or doc.stamp_duty_assessment is not None):
        assessment = doc.stamp_duty_assessment
        if not assessment and doc.category == "gift_deed":
            assessment = StampDutyAssessment(document_id=doc.id)
            db.add(assessment)
            doc.stamp_duty_assessment = assessment

        if assessment:
            fname = field.field_name
            val = payload.value
            relevant = False
            if fname == "market_value":
                assessment.market_value = parse_indian_currency_amount(val or "") if val else None
                relevant = True
            elif fname == "property_type":
                assessment.property_type = val or "unknown"
                relevant = True
            elif fname in ("stamp_duty_paid", "paid_stamp_duty"):
                assessment.stamp_duty_paid = parse_indian_currency_amount(val or "") if val else None
                relevant = True
            elif fname == "metro_cess_paid":
                assessment.metro_cess_paid = parse_indian_currency_amount(val or "") if val else None
                relevant = True
            elif fname in ("total_stamp_duty_paid", "total_paid"):
                assessment.total_stamp_duty_paid = parse_indian_currency_amount(val or "") if val else None
                relevant = True
            elif fname in ("donor_donee_relationship", "relationship"):
                assessment.relationship = val
                if val:
                    _, norm_rel, elig, _, _, _ = evaluate_article_34_relationship(val)
                    assessment.relationship_normalized = norm_rel
                    assessment.relationship_eligibility = elig
                else:
                    assessment.relationship_normalized = None
                    assessment.relationship_eligibility = "unknown"
                relevant = True
            elif fname == "relationship_eligibility":
                assessment.relationship_eligibility = val or "unknown"
                relevant = True
            elif fname == "survey_number":
                assessment.survey_number = val
                relevant = True
            elif fname == "property_location":
                assessment.property_location = val
                relevant = True
            elif fname == "property_description":
                assessment.property_description = val
                relevant = True
            elif fname == "area":
                assessment.area = val
                relevant = True
            elif fname == "challan_number":
                assessment.challan_number = val
                relevant = True

            if relevant:
                assessment.review_status = "corrected"
                recalculate_and_sync_stamp_duty_assessment(db, assessment, doc, skip_field_id=field.id)

    db.commit()
    return _field_dict(field)


@app.get("/api/cases")
def list_cases(db: Session = Depends(get_db)):
    rows = db.scalars(
        select(CaseRecord).options(
            selectinload(CaseRecord.documents).selectinload(LogicalDocument.fields),
            selectinload(CaseRecord.documents).selectinload(LogicalDocument.stamp_duty_assessment),
        )
    ).all()
    return [
        {
            "id": row.id,
            "title": row.title,
            "reference": row.reference,
            "status": row.status,
            "documents": [_doc_dict(doc) for doc in row.documents],
            "people": [{"id": link.person.id, "name": link.person.display_name, "relationship": link.relationship_type} for link in row.people],
            "properties": [{"id": link.property.id, "identifier": link.property.identifier, "description": link.property.description} for link in row.properties],
        }
        for row in rows
    ]


@app.get("/api/cases/{case_id}/report")
def case_report(case_id: str, format: Literal["json", "csv"] = "json", db: Session = Depends(get_db)):
    case = db.scalar(
        select(CaseRecord)
        .options(
            selectinload(CaseRecord.documents).selectinload(LogicalDocument.fields),
            selectinload(CaseRecord.documents).selectinload(LogicalDocument.stamp_duty_assessment),
        )
        .where(CaseRecord.id == case_id)
    )
    if not case:
        raise HTTPException(404, "Case not found.")
    findings = db.scalars(select(ValidationFinding).where(ValidationFinding.case_id == case.id)).all()
    report = {
        "case": {
            "id": case.id,
            "title": case.title,
            "reference": case.reference,
            "status": case.status,
            "people": [{"name": link.person.display_name, "relationship": link.relationship_type} for link in case.people],
            "properties": [{"identifier": link.property.identifier, "description": link.property.description} for link in case.properties],
        },
        "documents": [{**_doc_dict(doc), "fields": [_field_dict(field) for field in doc.fields]} for doc in case.documents],
        "findings": [{"rule_id": f.rule_id, "severity": f.severity, "kind": f.kind, "explanation": f.explanation, "evidence": f.evidence, "source_pages": f.source_pages, "review_status": f.review_status} for f in findings],
        "information_status_note": "Extracted values are machine-generated unless individually marked verified; inferred values are labelled separately.",
    }
    if format == "json":
        return report
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["case_id", "document_id", "category", "page_start", "page_end", "field", "value", "verification_status", "source_page"])
    for doc in case.documents:
        for field in doc.fields:
            report_field = _field_dict(field)
            writer.writerow([case.id, doc.id, doc.category, doc.start_page, doc.end_page, field.field_name, report_field["display_value"], field.verification_status, field.source_page])
    return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="case-{case.id}.csv"'})


@app.get("/api/findings")
def list_findings(db: Session = Depends(get_db)):
    rows = db.scalars(select(ValidationFinding).order_by(ValidationFinding.severity.desc())).all()
    return [{"id": row.id, "case_id": row.case_id, "document_id": row.document_id, "rule_id": row.rule_id, "severity": row.severity, "kind": row.kind, "explanation": row.explanation, "evidence": row.evidence, "source_pages": row.source_pages, "review_status": row.review_status} for row in rows]


@app.patch("/api/findings/{finding_id}")
def resolve_finding(finding_id: str, payload: FindingDecision, db: Session = Depends(get_db)):
    finding = db.get(ValidationFinding, finding_id)
    if not finding:
        raise HTTPException(404, "Finding not found.")
    old = finding.review_status
    finding.review_status = payload.status
    _audit(db, "finding", finding.id, "finding_reviewed", old, payload.status)
    db.add(ReviewDecision(entity_type="finding", entity_id=finding.id, decision=payload.status, note=payload.note))
    db.commit()
    return {"id": finding.id, "review_status": finding.review_status}


@app.delete("/api/ingestions/{ingestion_id}", status_code=204)
def delete_ingestion(ingestion_id: str, db: Session = Depends(get_db)):
    ingestion = db.get(Ingestion, ingestion_id)
    if not ingestion:
        raise HTTPException(404, "Ingestion not found.")
    storage_path = Path(ingestion.storage_path).parent
    case_ids = set(db.scalars(select(LogicalDocument.case_id).where(LogicalDocument.ingestion_id == ingestion_id, LogicalDocument.case_id.is_not(None))).all())
    _audit(db, "ingestion", ingestion.id, "ingestion_deleted", ingestion.sha256, None)
    db.delete(ingestion)
    db.commit()
    for case_id in case_ids:
        case = db.get(CaseRecord, case_id)
        if case and not db.scalar(select(LogicalDocument.id).where(LogicalDocument.case_id == case_id).limit(1)):
            db.delete(case)
    db.flush()
    _delete_orphan_entities(db)
    db.commit()
    shutil.rmtree(storage_path, ignore_errors=True)
    return Response(status_code=204)
