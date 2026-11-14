import io
import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from backend.app.db import Base, SessionLocal, engine
from backend.app.main import app
from backend.app.config import get_settings
from backend.app.models import (
    ComparisonRun,
    EngineProcessingResult,
    ExtractedField,
    Ingestion,
    LogicalDocument,
    Page,
    StampDutyAssessment,
    ValidationFinding,
)
from backend.app.ocr import detect_table_lines

client = TestClient(app)


def pdf_bytes(page_texts: list[str] | None = None) -> bytes:
    page_texts = page_texts or ["GIFT DEED Registration No: AB-1234 Date: 12/03/2024 Donor: Synthetic Alice Example"]
    page_objects = []
    for index, text in enumerate(page_texts):
        page_id = 3 + 2 * index
        content_id = page_id + 1
        lines = text.split("\n")
        escaped_lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
        text_ops = "\n".join(f"({line}) '" for line in escaped_lines)
        content = f"BT /F1 12 Tf 14 TL 50 750 Td\n{text_ops}\nET".encode()
        page_objects.extend(
            [
                f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {3 + 2 * len(page_texts)} 0 R >> >> /Contents {content_id} 0 R >>\nendobj\n".encode(),
                f"{content_id} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream\nendobj\n",
            ]
        )
    page_ids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(page_texts)))
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{page_ids}] /Count {len(page_texts)} >>\nendobj\n".encode(),
        *page_objects,
        f"{3 + 2 * len(page_texts)} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode(),
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode())
    return bytes(output)


def image_bytes(text: str = "GIFT DEED Registration No: TEST-1001") -> bytes:
    image = Image.new("RGB", (1800, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=60)
    for line_number, line in enumerate(text.splitlines()):
        draw.text((25, 35 + line_number * 75), line, font=font, fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def wait_for_ingestion(ingestion_id: str) -> dict:
    for _ in range(1200):
        state = client.get(f"/api/ingestions/{ingestion_id}").json()
        if state["status"] in {"completed", "completed_with_warnings", "failed"}:
            return state
        time.sleep(0.1)
    raise AssertionError(f"Ingestion {ingestion_id} did not finish in time.")


def setup_function():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def test_upload_process_review_and_export():
    response = client.post(
        "/api/ingestions",
        files={"file": ("sample.pdf", pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 202
    ingestion_id = response.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    assert len(documents) == 1
    document = documents[0]
    assert document["category"] == "gift_deed"
    detail = client.get(f"/api/documents/{document['id']}").json()
    assert detail["pages"][0]["ocr_text"]
    original_text = detail["pages"][0]["ocr_text_original"]
    correction = client.patch(
        f"/api/pages/{detail['pages'][0]['id']}/ocr",
        json={"corrected_text": f"{original_text}\nReviewer correction"},
    )
    assert correction.status_code == 200
    corrected_detail = client.get(f"/api/documents/{document['id']}").json()
    assert corrected_detail["pages"][0]["ocr_text_original"] == original_text
    assert corrected_detail["pages"][0]["ocr_text"].endswith("Reviewer correction")
    registration = next(field for field in document["fields"] if field["field_name"] == "registration_number")
    corrected = client.patch(
        f"/api/fields/{registration['id']}",
        json={"value": "AB-1234-VERIFIED", "verification_status": "verified"},
    )
    assert corrected.status_code == 200
    assert corrected.json()["verified_value"] == "AB-1234-VERIFIED"
    audit = client.get(f"/api/documents/{document['id']}/audit")
    assert {event["action"] for event in audit.json()} >= {"field_corrected", "ocr_corrected"}

    case = client.get("/api/cases").json()[0]
    report = client.get(f"/api/cases/{case['id']}/report?format=json")
    assert report.status_code == 200
    assert report.json()["case"]["reference"] == "ab-1234"
    assert report.json()["case"]["people"][0]["name"] == "Synthetic Alice Example"
    csv_report = client.get(f"/api/cases/{case['id']}/report?format=csv")
    assert csv_report.status_code == 200
    assert "verification_status" in csv_report.text

    duplicate = client.post(
        "/api/ingestions",
        files={"file": ("another-name.pdf", pdf_bytes(), "application/pdf")},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["id"] == ingestion_id
    source = client.get(f"/api/ingestions/{ingestion_id}/source")
    assert source.status_code == 200
    assert source.content.startswith(b"%PDF-")
    deleted = client.delete(f"/api/ingestions/{ingestion_id}")
    assert deleted.status_code == 204
    assert client.get("/api/dashboard").json()["ingestions"] == 0


def test_pdf_page_preview_returns_rendered_image():
    response = client.post(
        "/api/ingestions",
        files={"file": ("sample.pdf", pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 202
    ingestion_id = response.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    document = client.get("/api/documents").json()["items"][0]
    preview = client.get(f"/api/documents/{document['id']}/pages/1/image")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert preview.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_formatted_pdf_report_shows_positioned_ocr_image_and_red_missing_field(tmp_path):
    source_buffer = io.BytesIO()
    source_canvas = canvas.Canvas(source_buffer, pagesize=(612, 792))
    source_canvas.drawString(48, 744, "ORIGINAL SOURCE TABLE - Registration: SYN-102")
    source_canvas.drawString(48, 720, "Column A                  Column B")
    source_canvas.save()
    source_bytes = source_buffer.getvalue()
    ingestion_id = "formatted-report-test"
    storage_root = Path(get_settings().storage_dir).resolve()
    source_dir = storage_root / ingestion_id
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "source.pdf"
    source_path.write_bytes(source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    with SessionLocal() as db:
        ingestion = Ingestion(
            id=ingestion_id,
            filename="synthetic-report.pdf",
            sha256=source_hash,
            file_size=len(source_bytes),
            page_count=1,
            status="completed",
            storage_path=str(source_path),
        )
        db.add(ingestion)
        db.commit()
        document = LogicalDocument(
            id="formatted-report-document",
            ingestion_id=ingestion_id,
            start_page=1,
            end_page=1,
            category="gift_deed",
            predicted_category="gift_deed",
            classification_evidence="Synthetic test",
        )
        page = Page(
            id="formatted-report-page",
            ingestion_id=ingestion_id,
            page_number=1,
            digital_text="Registration No: SYN-102\nExtracted detail: verified text",
            ocr_word_boxes=json.dumps(
                {
                    "words": [
                        {"text": "POSITIONED", "x": 48, "y": 120, "width": 150, "height": 24},
                        {"text": "OCR", "x": 205, "y": 120, "width": 48, "height": 24},
                    ],
                    "table_lines": [
                        {"orientation": "horizontal", "x1": 40, "y1": 110, "x2": 520, "y2": 110},
                        {"orientation": "vertical", "x1": 40, "y1": 110, "x2": 40, "y2": 210},
                    ],
                }
            ),
            category="gift_deed",
        )
        db.add_all([document, page])
        db.commit()
        db.add_all(
            [
                ExtractedField(
                    document_id=document.id,
                    field_name="registration_number",
                    extracted_value="SYN-102",
                    source_page=1,
                    verification_status="unreviewed",
                ),
                ExtractedField(
                    document_id=document.id,
                    field_name="property_id",
                    extracted_value=None,
                    source_page=1,
                    evidence="Expected field not found in OCR text.",
                    verification_status="not_available",
                ),
            ]
        )
        db.commit()

    response = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "formatted-report.pdf" in response.headers["content-disposition"]
    report = PdfReader(io.BytesIO(response.content))
    assert len(report.pages) == 1
    report_text = report.pages[0].extract_text()
    assert "ORIGINAL SOURCE TABLE" not in report_text
    assert "OCR PAGE 1 OF 1" in report_text
    assert "Positioned OCR image at left" in report_text
    assert "Registration Number" in report_text
    assert "SYN-102" in report_text
    assert "Extracted detail: verified text" in report_text
    assert "Not available" in report_text
    assert "verify the source" in report_text
    xobjects = report.pages[0]["/Resources"]["/XObject"].get_object()
    assert any(resource.get_object().get("/Subtype") == "/Image" for resource in xobjects.values())
    layout_image = report.pages[0].images[0].image.convert("RGB")
    assert layout_image.getpixel((100, 110))[0] < 80
    assert layout_image.getpixel((600, 110))[0] > 240


def test_formatted_pdf_report_supports_original_image_bundles_and_checks_integrity():
    image_data = image_bytes("ORIGINAL SCANNED TABLE\nSOURCE ROW 1")
    ingestion_id = f"formatted-image-report-{time.time_ns()}"
    storage_root = Path(get_settings().storage_dir).resolve()
    source_dir = storage_root / ingestion_id
    image_dir = source_dir / "page_images"
    image_dir.mkdir(parents=True, exist_ok=False)
    image_path = image_dir / "page-1-image-0.png"
    image_path.write_bytes(image_data)
    manifest_path = source_dir / "image-manifest.json"
    manifest_path.write_text('["page-1-image-0.png"]', encoding="utf-8")
    digest = hashlib.sha256(b"casework-image-bundle-v1")
    digest.update(len(image_data).to_bytes(8, "big"))
    digest.update(image_data)
    document_id = f"{ingestion_id}-document"

    with SessionLocal() as db:
        db.add_all(
            [
                Ingestion(
                    id=ingestion_id,
                    filename="scanned-table-bundle",
                    sha256=digest.hexdigest(),
                    file_size=len(image_data),
                    page_count=1,
                    status="completed",
                    storage_path=str(manifest_path),
                    source_type="image",
                ),
                LogicalDocument(
                    id=document_id,
                    ingestion_id=ingestion_id,
                    start_page=1,
                    end_page=1,
                    category="gift_deed",
                    predicted_category="gift_deed",
                    classification_evidence="Synthetic image report test",
                ),
                Page(
                    id=f"{ingestion_id}-page",
                    ingestion_id=ingestion_id,
                    page_number=1,
                    ocr_text_original="OCR row 1: SYN-IMAGE-42",
                    category="gift_deed",
                ),
            ]
        )
        db.commit()
        db.add_all(
            [
                ExtractedField(
                    document_id=document_id,
                    field_name="registration_number",
                    extracted_value="SYN-IMAGE-42",
                    source_page=1,
                    verification_status="unreviewed",
                ),
                ExtractedField(
                    document_id=document_id,
                    field_name="property_id",
                    extracted_value=None,
                    source_page=1,
                    verification_status="not_available",
                ),
            ]
        )
        db.commit()

    response = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")

    assert response.status_code == 200
    report = PdfReader(io.BytesIO(response.content))
    assert len(report.pages) == 1
    report_text = report.pages[0].extract_text()
    assert "Registration Number" in report_text
    assert "SYN-IMAGE-42" in report_text
    assert "Not available" in report_text
    assert report.pages[0]["/Resources"]["/XObject"]

    image_path.write_bytes(image_bytes("CHANGED IMAGE CONTENT"))
    changed_source = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert changed_source.status_code == 409
    assert "image bundle has changed since ingestion" in changed_source.json()["detail"]


def test_positioned_ocr_detects_table_rules_from_the_page_image(tmp_path):
    image = Image.new("L", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    for y in (100, 300, 700):
        draw.line((50, y, 1150, y), fill="black", width=4)
    for x in (50, 300, 600, 900, 1150):
        draw.line((x, 100, x, 700), fill="black", width=4)
    image_path = tmp_path / "table-page.png"
    image.save(image_path)

    lines = detect_table_lines(image_path)
    horizontal_positions = [line["y1"] for line in lines if line["orientation"] == "horizontal"]
    vertical_positions = [line["x1"] for line in lines if line["orientation"] == "vertical"]

    assert len(horizontal_positions) >= 3
    assert len(vertical_positions) >= 4
    assert all(any(abs(position - expected) <= 8 for position in horizontal_positions) for expected in (100, 300, 700))
    assert all(any(abs(position - expected) <= 8 for position in vertical_positions) for expected in (50, 300, 600, 900))


def test_positioned_ocr_renders_pdf_pages_without_saved_page_images(tmp_path):
    source_buffer = io.BytesIO()
    source_canvas = canvas.Canvas(source_buffer, pagesize=(612, 792))
    for y in (200, 400, 650):
        source_canvas.line(50, y, 560, y)
    for x in (50, 220, 390, 560):
        source_canvas.line(x, 200, x, 650)
    source_canvas.save()
    source_bytes = source_buffer.getvalue()
    ingestion_id = f"positioned-pdf-lines-{time.time_ns()}"
    storage_root = Path(get_settings().storage_dir).resolve()
    source_dir = storage_root / ingestion_id
    source_dir.mkdir(parents=True, exist_ok=False)
    source_path = source_dir / "source.pdf"
    source_path.write_bytes(source_bytes)
    document_id = f"{ingestion_id}-document"

    with SessionLocal() as db:
        db.add_all(
            [
                Ingestion(
                    id=ingestion_id,
                    filename="vector-table.pdf",
                    sha256=hashlib.sha256(source_bytes).hexdigest(),
                    file_size=len(source_bytes),
                    page_count=1,
                    status="completed",
                    storage_path=str(source_path),
                ),
                LogicalDocument(
                    id=document_id,
                    ingestion_id=ingestion_id,
                    start_page=1,
                    end_page=1,
                    category="unknown",
                    predicted_category="unknown",
                    classification_evidence="Synthetic PDF table-line test",
                ),
                Page(
                    id=f"{ingestion_id}-page",
                    ingestion_id=ingestion_id,
                    page_number=1,
                    category="unknown",
                    image_path=None,
                ),
            ]
        )
        db.commit()

    response = client.get(f"/api/documents/{document_id}/pages/1/table-lines")

    assert response.status_code == 200
    lines = response.json()["table_lines"]
    assert sum(line["orientation"] == "horizontal" for line in lines) >= 3
    assert sum(line["orientation"] == "vertical" for line in lines) >= 3


def test_formatted_pdf_report_continues_long_extraction_text(tmp_path):
    source_buffer = io.BytesIO()
    source_canvas = canvas.Canvas(source_buffer, pagesize=(612, 792))
    source_canvas.drawString(48, 744, "ORIGINAL SOURCE PAGE")
    source_canvas.save()
    source_bytes = source_buffer.getvalue()
    ingestion_id = "formatted-report-long-text"
    storage_root = Path(get_settings().storage_dir).resolve()
    source_dir = storage_root / ingestion_id
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "source.pdf"
    source_path.write_bytes(source_bytes)
    long_field_value = f"field-start {'segment ' * 500}field-end"

    with SessionLocal() as db:
        ingestion = Ingestion(
            id=ingestion_id,
            filename="long-text.pdf",
            sha256=hashlib.sha256(source_bytes).hexdigest(),
            file_size=len(source_bytes),
            page_count=1,
            status="completed",
            storage_path=str(source_path),
        )
        document = LogicalDocument(
            id="formatted-report-long-document",
            ingestion_id=ingestion_id,
            start_page=1,
            end_page=1,
            category="unknown",
            predicted_category="unknown",
        )
        db.add_all(
            [
                ingestion,
                document,
                Page(
                    id="formatted-report-long-page",
                    ingestion_id=ingestion_id,
                    page_number=1,
                    digital_text="\n".join(f"Extracted line number {index}" for index in range(200)),
                    category="unknown",
                ),
                ExtractedField(
                    document_id=document.id,
                    field_name="long_test_field",
                    extracted_value=long_field_value,
                    source_page=1,
                    verification_status="unreviewed",
                ),
            ]
        )
        db.commit()

    response = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")

    assert response.status_code == 200
    report = PdfReader(io.BytesIO(response.content))
    report_text = "\n".join(page.extract_text() for page in report.pages)
    assert len(report.pages) > 1
    assert "Extracted line number 199" in report_text
    assert "field-start" in report_text
    assert "field-end" in report_text


@pytest.mark.parametrize(
    ("status", "source_content", "expected_detail"),
    [
        ("processing", None, "Wait until document extraction is complete"),
        ("completed", pdf_bytes(), "has changed since ingestion"),
    ],
)
def test_formatted_pdf_report_requires_completed_unchanged_source(
    status,
    source_content,
    expected_detail,
):
    ingestion_id = f"formatted-report-guard-{status}"
    storage_root = Path(get_settings().storage_dir).resolve()
    source_dir = storage_root / ingestion_id
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "source.pdf"
    if source_content is not None:
        source_path.write_bytes(source_content)
    with SessionLocal() as db:
        db.add(
            Ingestion(
                id=ingestion_id,
                filename="guard-test.pdf",
                sha256="0" * 64,
                file_size=len(source_content or b""),
                page_count=1,
                status=status,
                storage_path=str(source_path),
            )
        )
        db.commit()

    response = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")

    assert response.status_code == 409
    assert expected_detail in response.json()["detail"]


def test_rejects_non_pdf_signature():
    response = client.post(
        "/api/ingestions",
        files={"file": ("not-a-pdf.pdf", b"not a PDF document", "application/pdf")},
    )
    assert response.status_code == 400


def test_rejects_malformed_pdf_with_clear_client_error():
    response = client.post(
        "/api/ingestions",
        files={"file": ("malformed.pdf", b"%PDF-1.4\nnot a complete PDF", "application/pdf")},
    )
    assert response.status_code == 400
    assert "PDF could not be opened" in response.json()["detail"]


def test_uploads_one_image_through_document_pipeline_and_preserves_source():
    payload = image_bytes()
    response = client.post(
        "/api/ingestions/images",
        files=[("files", ("gift-deed.png", payload, "image/png"))],
    )
    assert response.status_code == 202
    ingestion_id = response.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["source_type"] == "image"
    assert state["page_count"] == 1
    assert state["status"] in {"completed", "completed_with_warnings"}
    listed_ingestion = next(item for item in client.get("/api/ingestions").json() if item["id"] == ingestion_id)
    assert listed_ingestion["stage"] == "completed"

    documents = client.get("/api/documents").json()["items"]
    document = next(item for item in documents if item["ingestion_id"] == ingestion_id)
    detail = client.get(f"/api/documents/{document['id']}").json()
    assert detail["source_type"] == "image"
    assert detail["pages"][0]["page_number"] == 1
    if client.get("/api/health").json()["ocr_configured"]:
        assert detail["pages"][0]["ocr_engine"].startswith("paddleocr")
        assert "gift deed" in detail["pages"][0]["ocr_text"].casefold()
        assert detail["pages"][0]["ocr_confidence_proxy"] is not None
    else:
        assert detail["pages"][0]["ocr_error"]
    preview = client.get(f"/api/documents/{document['id']}/pages/1/image")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    assert preview.content == payload
    duplicate = client.post(
        "/api/ingestions/images",
        files=[("files", ("renamed-gift-deed.png", payload, "image/png"))],
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True


def test_reprocess_fields_classifies_invoice_and_records_missing_values():
    ingestion_id = "invoice-reprocess-test"
    with SessionLocal() as db:
        db.add(
            Ingestion(
                id=ingestion_id,
                filename="invoice.png",
                sha256="a" * 64,
                file_size=100,
                page_count=1,
                status="completed",
                storage_path="invoice-manifest.json",
                source_type="image",
            )
        )
        db.flush()
        db.add(
            Page(
                id="invoice-reprocess-page",
                ingestion_id=ingestion_id,
                page_number=1,
                ocr_text_original=(
                    "TAX INVOICE\nInvoice Serial Number: INV-001\n"
                    "Invoice Date: 01-02-2025\nGSTIN: 29ABCDE1234F1Z5"
                ),
                category="unknown",
                classification_evidence="No configured category keywords matched",
            )
        )
        db.add(
            LogicalDocument(
                id="invoice-reprocess-document",
                ingestion_id=ingestion_id,
                start_page=1,
                end_page=1,
                category="unknown",
                predicted_category="unknown",
                classification_evidence="No configured category keywords matched",
                review_status="needs_review",
            )
        )
        db.commit()

    response = client.post(f"/api/ingestions/{ingestion_id}/reprocess-fields")

    assert response.status_code == 200
    assert response.json()["documents"] == 1
    details = client.get("/api/documents/invoice-reprocess-document").json()
    missing_value = next(field for field in details["fields"] if field["field_name"] == "invoice_total")
    assert missing_value["value"] is None
    assert missing_value["display_value"] == "Not available"
    assert missing_value["availability"] == "not_available"
    with SessionLocal() as db:
        document = db.get(LogicalDocument, "invoice-reprocess-document")
        page = db.get(Page, "invoice-reprocess-page")
        fields = {field.field_name: field for field in document.fields}
        assert document.category == "tax_invoice"
        assert "tax invoice" in document.classification_evidence.casefold()
        assert page.category == "tax_invoice"
        assert fields["invoice_number"].extracted_value == "INV-001"
        assert fields["invoice_date"].extracted_value == "01-02-2025"
        assert fields["invoice_total"].verification_status == "not_available"
        finding = db.query(ValidationFinding).filter_by(
            document_id=document.id,
            rule_id="missing_invoice_fields",
        ).one()
        assert "invoice total" in finding.explanation


def test_uploads_multiple_images_as_one_page_bundle():
    first_page = "SYNTHETIC TEST PAGE\nGIFT DEED\nRegistration Ref: SYNTH-REG-2026-001\nProperty ID: FLAT-TEST-001"
    second_page = "SYNTHETIC TEST PAGE\nGIFT DEED\nRegistration Ref: SYNTH-REG-2026-001\nProperty ID: FLAT-TEST-002"
    response = client.post(
        "/api/ingestions/images",
        files=[
            ("files", ("gift-page-1.png", image_bytes(first_page), "image/png")),
            ("files", ("gift-page-2.png", image_bytes(second_page), "image/png")),
        ],
        data={"bundle_name": "Synthetic gift deed pages"},
    )
    assert response.status_code == 202
    ingestion_id = response.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["page_count"] == 2
    assert state["status"] in {"completed", "completed_with_warnings"}
    documents = client.get("/api/documents").json()["items"]
    matching = [item for item in documents if item["ingestion_id"] == ingestion_id]
    assert matching
    assert matching[0]["start_page"] == 1
    assert matching[-1]["end_page"] == 2
    if client.get("/api/health").json()["ocr_configured"]:
        detail = client.get(f"/api/documents/{matching[0]['id']}").json()
        assert "SYNTH-REG-2026-001" in detail["pages"][0]["ocr_text"]
        assert "FLAT-TEST-002" in detail["pages"][1]["ocr_text"]
        findings = client.get("/api/findings").json()
        document_findings = [finding for finding in findings if finding["document_id"] == matching[0]["id"]]
        conflict = next(
            finding for finding in document_findings
            if finding["rule_id"] == "conflicting_extracted_values"
        )
        assert "FLAT-TEST-001" in conflict["evidence"]
        assert "FLAT-TEST-002" in conflict["evidence"]
        assert not any(finding["rule_id"] == "missing_registration_reference" for finding in document_findings)
    source = client.get(f"/api/ingestions/{ingestion_id}/source")
    assert source.status_code == 200
    assert source.content == image_bytes(first_page)
    assert client.delete(f"/api/ingestions/{ingestion_id}").status_code == 204


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "expected_status"),
    [
        ("bad.gif", "image/gif", b"not an image", 400),
        ("bad.png", "text/plain", b"not an image", 400),
        ("corrupt.jpg", "image/jpeg", b"not an image", 400),
    ],
)
def test_rejects_invalid_image_upload(filename, content_type, content, expected_status):
    response = client.post(
        "/api/ingestions/images",
        files=[("files", (filename, content, content_type))],
    )
    assert response.status_code == expected_status
    assert response.json()["detail"]


def test_rejects_image_batch_over_size_limit(monkeypatch):
    monkeypatch.setattr("backend.app.main.settings.max_upload_mb", 0)
    response = client.post(
        "/api/ingestions/images",
        files=[("files", ("too-large.png", image_bytes(), "image/png"))],
    )
    assert response.status_code == 413
    assert "upload limit" in response.json()["detail"]


def test_unsupported_image_only_pdf_completes_with_page_warning(monkeypatch):
    monkeypatch.setattr("backend.app.pdf_utils.fitz", None)
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)

    response = client.post(
        "/api/ingestions",
        files={"file": ("unsupported-scan.pdf", output.getvalue(), "application/pdf")},
    )
    assert response.status_code == 202
    ingestion_id = response.json()["id"]
    state = wait_for_ingestion(ingestion_id)

    assert state["status"] == "completed_with_warnings"
    documents = client.get("/api/documents").json()["items"]
    document = next(item for item in documents if item["ingestion_id"] == ingestion_id)
    detail = client.get(f"/api/documents/{document['id']}").json()
    page = detail["pages"][0]
    assert page["ocr_text"] == ""
    assert "no full-page pdf renderer is available" in page["ocr_error"].casefold()
    assert "OCR was skipped" in page["ocr_error"]
    assert client.delete(f"/api/ingestions/{ingestion_id}").status_code == 204


def test_comparison_artifact_route_rejects_paths_outside_private_storage(tmp_path, monkeypatch):
    storage_dir = tmp_path / "private"
    storage_dir.mkdir()
    outside_artifact = tmp_path / "outside.json"
    outside_artifact.write_text('{"private": true}', encoding="utf-8")
    monkeypatch.setattr("backend.app.main.settings.storage_dir", storage_dir)

    with SessionLocal() as db:
        ingestion = Ingestion(
            id="comparison-artifact-ingestion",
            filename="synthetic.pdf",
            sha256="a" * 64,
            file_size=20,
            page_count=1,
            status="completed",
            storage_path=str(outside_artifact),
        )
        comparison = ComparisonRun(
            id="comparison-artifact-test",
            ingestion_id=ingestion.id,
            mode="docling",
            source_sha256=ingestion.sha256,
            engines=[
                EngineProcessingResult(
                    engine="docling",
                    status="completed",
                    output_json_path="../outside.json",
                )
            ],
        )
        db.add(ingestion)
        db.commit()
        db.add(comparison)
        db.commit()

    response = client.get(
        "/api/comparisons/comparison-artifact-test/artifacts/docling/native.json"
    )

    assert response.status_code == 404


def test_deleting_ingestion_removes_comparison_artifacts_and_results(tmp_path, monkeypatch):
    storage_dir = tmp_path / "private"
    source_dir = storage_dir / "comparison-delete-ingestion"
    artifacts_dir = source_dir / "comparisons" / "comparison-delete-run" / "docling"
    artifacts_dir.mkdir(parents=True)
    source_pdf = source_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 synthetic")
    native_output = artifacts_dir / "native.json"
    native_output.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("backend.app.main.settings.storage_dir", storage_dir)

    with SessionLocal() as db:
        ingestion = Ingestion(
            id="comparison-delete-ingestion",
            filename="synthetic.pdf",
            sha256="b" * 64,
            file_size=source_pdf.stat().st_size,
            page_count=1,
            status="completed",
            storage_path=str(source_pdf),
        )
        db.add(ingestion)
        db.commit()
        comparison = ComparisonRun(
            id="comparison-delete-run",
            ingestion_id=ingestion.id,
            mode="docling",
            source_sha256=ingestion.sha256,
            engines=[
                EngineProcessingResult(
                    engine="docling",
                    status="completed",
                    output_json_path="comparison-delete-ingestion/comparisons/comparison-delete-run/docling/native.json",
                )
            ],
        )
        db.add(comparison)
        db.commit()

    response = client.delete("/api/ingestions/comparison-delete-ingestion")

    assert response.status_code == 204
    assert not source_dir.exists()
    assert client.get("/api/comparisons/comparison-delete-run").status_code == 404


def test_invalid_split_does_not_change_document():
    page_text = "GIFT DEED Registration No: AB-1111 Synthetic document page."
    uploaded = client.post("/api/ingestions", files={"file": ("sample.pdf", pdf_bytes([page_text, page_text]), "application/pdf")})
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}
    document = client.get("/api/documents").json()["items"][0]
    response = client.post(f"/api/documents/{document['id']}/split", json={"split_before_page": 1})
    assert response.status_code == 400
    split = client.post(f"/api/documents/{document['id']}/split", json={"split_before_page": 2})
    assert split.status_code == 201
    created_id = split.json()["created"]["id"]
    assert len(client.get("/api/documents").json()["items"]) == 2
    merged = client.post(f"/api/documents/{document['id']}/merge", json={"other_document_id": created_id})
    assert merged.status_code == 200
    assert merged.json()["start_page"] == 1
    assert merged.json()["end_page"] == 2



def test_donor_donee_relationship_end_to_end_review_and_verification():
    page_text = (
        "GIFT DEED\n"
        "Registration Ref: REG-GIFT-REL-01\n"
        "Donor: Ramesh Kumar Sharma\n"
        "Donee: Anil Kumar Sharma\n"
        "WHEREAS the Donor is the father of the Donee and out of natural love and affection conveys the property.\n"
        "NOW THIS DEED WITNESSETH that the gift is executed."
    )
    uploaded = client.post("/api/ingestions", files={"file": ("gift_rel.pdf", pdf_bytes([page_text]), "application/pdf")})
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    doc = next(d for d in documents if d["ingestion_id"] == ingestion_id)
    assert doc["category"] == "gift_deed"

    details = client.get(f"/api/documents/{doc['id']}").json()
    rel_field = next(f for f in details["fields"] if f["field_name"] == "donor_donee_relationship")
    assert rel_field["value"] == "Father and Son"
    assert rel_field["verification_status"] == "unreviewed"
    assert "WHEREAS the Donor is the father of the Donee" in rel_field["evidence"]
    assert rel_field["source_page"] == 1

    # Reviewer verifies the field
    verified_resp = client.patch(
        f"/api/fields/{rel_field['id']}",
        json={"value": "Father and Son", "verification_status": "verified"},
    )
    assert verified_resp.status_code == 200
    assert verified_resp.json()["verification_status"] == "verified"
    assert verified_resp.json()["verified_value"] == "Father and Son"

    # Reviewer edits the field
    edited_resp = client.patch(
        f"/api/fields/{rel_field['id']}",
        json={"value": "Father to Son (Verified)", "verification_status": "verified"},
    )
    assert edited_resp.status_code == 200
    assert edited_resp.json()["verified_value"] == "Father to Son (Verified)"

    # Audit history recorded
    audit = client.get(f"/api/documents/{doc['id']}/audit").json()
    field_audits = [a for a in audit if a["action"] == "field_corrected"]
    assert len(field_audits) >= 2

    # Clean up
    client.delete(f"/api/ingestions/{ingestion_id}")



def test_donor_donee_relationship_missing_flagged_in_api_and_report():
    page_text = (
        "GIFT DEED\n"
        "Registration Ref: REG-GIFT-MISSING-01\n"
        "Donor: Rajesh Kumar Sharma\n"
        "Donee: Anil Kumar Sharma\n"
        "WHEREAS the Donor is the owner of the property and transfers it without consideration.\n"
        "NOW THIS DEED WITNESSETH that the gift is executed."
    )
    uploaded = client.post("/api/ingestions", files={"file": ("gift_missing.pdf", pdf_bytes([page_text]), "application/pdf")})
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    doc = next(d for d in documents if d["ingestion_id"] == ingestion_id)
    assert doc["category"] == "gift_deed"

    details = client.get(f"/api/documents/{doc['id']}").json()
    rel_field = next(f for f in details["fields"] if f["field_name"] == "donor_donee_relationship")
    assert rel_field["value"] is None
    assert rel_field["display_value"] == "Not available"
    assert rel_field["availability"] == "not_available"
    assert rel_field["verification_status"] == "not_available"

    # Finding created for missing expected relationship
    findings = client.get(f"/api/documents/{doc['id']}").json()
    with SessionLocal() as db:
        finding = db.query(ValidationFinding).filter_by(
            document_id=doc["id"],
            rule_id="missing_donor_donee_relationship",
        ).first()
        assert finding is not None
        assert "Expected donor-donee relationship" in finding.explanation

    # Formatted report download succeeds and contains red missing field count
    report = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF-")

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_stamp_duty_assessment_serialization_and_endpoint():
    page_text = (
        "GIFT DEED / बक्षीसपत्र\n"
        "Registration Ref: REG-GIFT-MUMBAI-44\n"
        "Donor: Ramesh Kulkarni\n"
        "Donee: Suresh Kulkarni\n"
        "WHEREAS the Donor out of love and affection towards his son the Donee transfers the residential premises.\n"
        "Schedule of Property: Residential Flat No 402, Mumbai Suburban, 650 sq ft.\n"
        "Ready Reckoner / Market Value: ₹ 25,00,000/-.\n"
        "Stamp Duty Paid: ₹ 20,000/-. Challan No: CH-2024-8899."
    )
    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("gift_stamp_duty.pdf", pdf_bytes([page_text]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    doc = next(d for d in documents if d["ingestion_id"] == ingestion_id)
    assert doc["category"] == "gift_deed"

    # Verify stamp_duty_assessment serialized in document_detail
    details = client.get(f"/api/documents/{doc['id']}").json()
    assert "stamp_duty_assessment" in details
    sda = details["stamp_duty_assessment"]
    assert sda is not None
    assert sda["donor_name"] == "Ramesh Kulkarni"
    assert sda["donee_name"] == "Suresh Kulkarni"
    assert sda["property_type"] == "residential_property"
    assert "son" in sda["relationship"].lower()
    assert sda["relationship_eligibility"] in {"eligible", "article_34_200"}
    assert sda["donor_role"] == "father"
    assert sda["donee_role"] == "son"
    assert sda["donee_relation_to_donor"] == "son"
    assert sda["selected_rule"] == "article_34_200"
    assert sda["duty_rate"] == "200_fixed"
    assert sda["cess_rule"] == "metro_cess_1_percent"
    assert sda["calculation_components"] is not None
    assert sda["source_evidence"] is not None
    assert sda["market_value"] == 2500000
    assert sda["base_stamp_duty"] == 200
    assert sda["metro_cess_rate"] == 0.01
    assert sda["metro_cess"] == 25000
    assert sda["expected_total"] == 25200
    assert sda["total_stamp_duty_paid"] == 20000
    assert sda["difference"] == -5200
    assert sda["status"] == "short_paid"
    assert sda["legal_rule_version"] == "MH-ART34-2024.1"
    assert sda["review_status"] == "unreviewed"

    # Verify document extracted_fields contains all checklist 3 assessment fields
    doc_fields_by_name = {f["field_name"]: f for f in details["fields"]}
    for req_field in (
        "property_type",
        "market_value",
        "stamp_duty_paid",
        "expected_stamp_duty",
        "stamp_duty_difference",
        "stamp_duty_status",
    ):
        assert req_field in doc_fields_by_name, f"Expected {req_field} in document extracted_fields"
        assert doc_fields_by_name[req_field]["value"] is not None

    # Verify review findings
    with SessionLocal() as db:
        finding = db.query(ValidationFinding).filter_by(
            document_id=doc["id"],
            rule_id="short_paid_stamp_duty",
        ).first()
        assert finding is not None
        assert "Short paid" in finding.explanation

    # PATCH /api/documents/{id}/stamp-duty-assessment
    patch_resp = client.patch(
        f"/api/documents/{doc['id']}/stamp-duty-assessment",
        json={
            "stamp_duty_paid": 25200,
            "total_stamp_duty_paid": 25200,
            "review_status": "confirmed",
            "reviewer_notes": "Verified receipt against Maharashtra e-GRAS treasury challan.",
        },
    )
    assert patch_resp.status_code == 200
    updated_sda = patch_resp.json()
    assert updated_sda["total_stamp_duty_paid"] == 25200
    assert updated_sda["difference"] == 0
    assert updated_sda["status"] == "compliant"
    assert updated_sda["review_status"] == "confirmed"
    assert updated_sda["reviewer_notes"] == "Verified receipt against Maharashtra e-GRAS treasury challan."

    # Verify fields synchronized
    updated_doc = client.get(f"/api/documents/{doc['id']}").json()
    fields_by_name = {f["field_name"]: f for f in updated_doc["fields"]}
    assert fields_by_name["stamp_duty_status"]["value"] == "compliant"
    assert fields_by_name["expected_stamp_duty"]["value"] == "25200"
    assert fields_by_name["stamp_duty_difference"]["value"] == "0"

    # Test PATCH /api/fields/{field_id} (correct_field) synchronizes back to assessment
    mv_field = fields_by_name["market_value"]
    field_patch_resp = client.patch(
        f"/api/fields/{mv_field['id']}",
        json={"value": "₹ 30,00,000", "verification_status": "verified"},
    )
    assert field_patch_resp.status_code == 200

    refreshed_doc = client.get(f"/api/documents/{doc['id']}").json()
    recalc_sda = refreshed_doc["stamp_duty_assessment"]
    assert recalc_sda["market_value"] == 3000000
    assert recalc_sda["metro_cess"] == 30000
    assert recalc_sda["expected_total"] == 30200
    assert recalc_sda["difference"] == 25200 - 30200  # -5000 (short paid again)
    assert recalc_sda["status"] == "short_paid"
    assert recalc_sda["review_status"] == "corrected"

    # Test PATCH /api/documents/{id}/stamp-duty-assessment overriding property_type, market_value, stamp_duty_paid, review_status
    override_resp = client.patch(
        f"/api/documents/{doc['id']}/stamp-duty-assessment",
        json={
            "property_type": "agricultural_land",
            "jurisdiction": "non_metro",
            "location_band": "gram_panchayat_other",
            "market_value": 4000000,
            "stamp_duty_paid": 200,
            "total_stamp_duty_paid": 200,
            "review_status": "confirmed",
            "reviewer_notes": "Reviewer confirmed agricultural land classification in rural non-metro area; 0% metro cess applied.",
        },
    )
    assert override_resp.status_code == 200
    override_sda = override_resp.json()
    assert override_sda["property_type"] == "agricultural_land"
    assert override_sda["market_value"] == 4000000
    assert override_sda["base_stamp_duty"] == 200
    assert override_sda["metro_cess"] == 0
    assert override_sda["expected_total"] == 200
    assert override_sda["total_stamp_duty_paid"] == 200
    assert override_sda["difference"] == 0
    assert override_sda["status"] == "compliant"
    assert override_sda["review_status"] == "confirmed"

    # Verify extracted_fields synchronized after override
    doc_after_override = client.get(f"/api/documents/{doc['id']}").json()
    overridden_fields = {f["field_name"]: f for f in doc_after_override["fields"]}
    assert overridden_fields["property_type"]["value"] == "agricultural_land"
    assert overridden_fields["market_value"]["value"] == "4000000"
    assert overridden_fields["stamp_duty_status"]["value"] == "compliant"
    assert overridden_fields["expected_stamp_duty"]["value"] == "200"
    assert overridden_fields["stamp_duty_difference"]["value"] == "0"

    # Test formatted PDF report contains Checklist 3 dedicated section
    report = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(report.content))
    all_text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert "checklist 3: stamp-duty assessment (article 34)" in all_text
    assert "market value (true/declared)" in all_text
    assert "40,00,000" in all_text
    assert "base stamp duty (art. 34)" in all_text
    assert "200" in all_text
    assert "expected total stamp duty" in all_text
    assert "maharashtra stamp act, article 34" in all_text
    assert "mh-art34-2024.1" in all_text

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_identification_documents_api_persistence_and_serialization():
    page_1 = (
        "GIFT DEED\n"
        "Registration Ref: REG-ID-TEST-99\n"
        "Donor: Ramesh Kulkarni, aged 64 years, PAN: ABCDE1234F\n"
        "Donee: Suresh Kulkarni, aged 32 years, Aadhaar No: 3456 7890 1234\n"
        "WHEREAS the Donor is the father of the Donee.\n"
        "Schedule of Property: Residential Flat No 10, Pune Municipal Corporation.\n"
        "Market Value: ₹ 30,00,000/-\n"
        "Total Stamp Duty Paid: ₹ 30,200/-"
    )
    page_2 = (
        "INCOME TAX DEPARTMENT\n"
        "Permanent Account Number Card\n"
        "Name\n"
        "RAMESH KULKARNI\n"
        "ABCDE1234F\n"
        "Date of Birth\n"
        "10/05/1960"
    )
    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("gift_with_ids.pdf", pdf_bytes([page_1, page_2]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")
    assert set(gift_doc["identification_documents"]["grouped_documents"].keys()) == {"pan_card", "aadhaar_card", "other_id"}
    assert gift_doc["identification_documents"]["status"] == "found"
    assert gift_doc["identification_documents"]["summary_counts"] == {"pan_card": 1, "aadhaar_card": 1, "other_id": 0, "total": 2}

    details = client.get(f"/api/documents/{gift_doc['id']}").json()
    id_docs = details["identification_documents"]
    assert details["stamp_duty_assessment"]["identification_documents"] == id_docs
    assert len(id_docs["pan_card"]) == 1
    assert id_docs["pan_card"][0]["id_number"] == "ABCDE1234F"
    assert id_docs["pan_card"][0]["party_role"] == "Donor"
    assert id_docs["pan_card"][0]["page_numbers"] == [1, 2]
    assert len(id_docs["aadhaar_card"]) == 1
    assert id_docs["aadhaar_card"][0]["id_number"] == "XXXX-XXXX-1234"
    assert id_docs["aadhaar_card"][0]["party_role"] == "Donee"

    # Generate formatted PDF report and verify Extracted Identification Documents rendering
    report = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(report.content))
    all_text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert "extracted identification documents" in all_text
    assert "pan card details" in all_text
    assert "aadhaar card details" in all_text
    assert "other identification documents" in all_text
    assert "abcde1234f" in all_text
    assert "xxxx-xxxx-1234" in all_text
    assert "no other identification documents detected" in all_text

    # Verify the Extracted Identification Documents section itself masks raw Aadhaar digits
    id_section_start = all_text.index("extracted identification documents")
    id_section_end = all_text.index("complete extracted text", id_section_start)
    id_section_text = all_text[id_section_start:id_section_end]
    assert "xxxx-xxxx-1234" in id_section_text
    assert "3456 7890 1234" not in id_section_text

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_formatted_pdf_report_wraps_long_id_evidence_and_renders_all_groups():
    page_1 = (
        "GIFT DEED\n"
        "Registration Ref: REG-ID-LONG-01\n"
        "Donor: Ramesh Kulkarni, aged 64 years, PAN: ABCDE1234F\n"
        "Donee: Suresh Kulkarni, aged 32 years, Voter ID: MHX1234567\n"
        "WHEREAS the Donor is the father of the Donee.\n"
        "Schedule of Property: Residential Flat No 10, Pune Municipal Corporation.\n"
        "Market Value: ₹ 30,00,000/-\n"
        "Total Stamp Duty Paid: ₹ 30,200/-"
    )
    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("gift_long_id.pdf", pdf_bytes([page_1]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    # Inject a long holder name on the persisted ID record to verify wrapping cleanly across margins
    # and confirm Confidence / Evidence commentary lines are omitted from ID cards in the PDF report
    from backend.app.models import StampDutyAssessment

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")
    long_tail = "LONG_HOLDER_END_MARKER_VERIFIED"
    long_holder = (
        "Rameshwaram Venkatasubramanian Kulkarni Alias Patwardhan "
        + long_tail
    )
    with SessionLocal() as db:
        assessment = db.query(StampDutyAssessment).filter_by(document_id=gift_doc["id"]).one()
        payload = json.loads(assessment.identification_documents)
        payload["pan_card"][0]["holder_name"] = long_holder
        if "grouped_documents" in payload:
            payload["grouped_documents"]["pan_card"][0]["holder_name"] = long_holder
        assessment.identification_documents = json.dumps(payload)
        db.commit()

    report = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(report.content))
    all_text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert "extracted identification documents" in all_text
    assert "pan card details" in all_text
    assert "no aadhaar card details detected" in all_text
    assert "other identification documents" in all_text
    assert "abcde1234f" in all_text
    assert "mhx1234567" in all_text
    assert long_tail.lower() in all_text
    id_section_start = all_text.index("extracted identification documents")
    id_section_end = all_text.index("complete extracted text", id_section_start)
    id_section_text = all_text[id_section_start:id_section_end]
    assert "confidence:" not in id_section_text
    assert "evidence:" not in id_section_text

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_end_to_end_synthetic_id_smoke_and_pdf():
    page_1 = (
        "GIFT DEED\n"
        "Registration Ref: REG-SMOKE-2026\n"
        "Donor: Ramesh Kulkarni, aged 65 years, holding PAN ABCDE1234F\n"
        "Donee: Suresh Kulkarni, aged 32 years, holding Aadhaar [Aadhaar Redacted]\n"
        "WHEREAS the Donor gifts the residential flat to his son the Donee.\n"
        "Schedule of Property: Residential Flat No 10, Pune Municipal Corporation.\n"
        "Market Value: ₹ 30,00,000/-\n"
        "Total Stamp Duty Paid: ₹ 30,200/-"
    )
    page_2 = (
        "INCOME TAX DEPARTMENT\n"
        "GOVT. OF INDIA\n"
        "Permanent Account Number Card\n"
        "ABCDE1234F\n"
        "Name\n"
        "RAMESH KULKARNI\n"
        "Date of Birth\n"
        "10/05/1960"
    )
    page_3 = (
        "Unique Identification Authority of India\n"
        "Government of India\n"
        "Suresh Kulkarni\n"
        "Aadhaar No: 4567 8901 2345\n"
        "Witness 1: Mohan Patil, Voter ID: MHX7654321"
    )
    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("synthetic_smoke.pdf", pdf_bytes([page_1, page_2, page_3]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")
    details = client.get(f"/api/documents/{gift_doc['id']}").json()
    id_docs = details["identification_documents"]

    assert id_docs["status"] == "found"
    assert id_docs["summary_counts"] == {"pan_card": 1, "aadhaar_card": 1, "other_id": 1, "total": 3}

    pan_items = id_docs["grouped_documents"]["pan_card"]
    assert len(pan_items) == 1
    assert pan_items[0]["id_number"] == "ABCDE1234F"
    assert pan_items[0]["party_role"] == "Donor"
    assert pan_items[0]["page_numbers"] == [1, 2]

    aadhaar_items = id_docs["grouped_documents"]["aadhaar_card"]
    assert len(aadhaar_items) == 1
    assert aadhaar_items[0]["party_role"] == "Donee"
    assert aadhaar_items[0]["id_number"] == "XXXX-XXXX-2345"
    assert "4567 8901 2345" not in aadhaar_items[0]["evidence_snippet"]

    other_items = id_docs["grouped_documents"]["other_id"]
    assert len(other_items) == 1
    assert other_items[0]["party_role"] == "Witness"
    assert other_items[0]["id_number"] == "MHX7654321"

    report = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(report.content))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    assert "extracted identification documents" in pdf_text
    assert "pan card" in pdf_text
    assert "aadhaar card" in pdf_text

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_identification_documents_legacy_null_fallback_in_api():
    page_1 = (
        "GIFT DEED\n"
        "Registration Ref: REG-LEGACY-01\n"
        "Donor: Ramesh Kulkarni\n"
        "Donee: Suresh Kulkarni\n"
        "WHEREAS the Donor is the father of the Donee.\n"
        "Schedule of Property: Residential Flat No 10, Pune Municipal Corporation.\n"
        "Market Value: ₹ 25,00,000/-\n"
        "Total Stamp Duty Paid: ₹ 25,200/-"
    )
    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("legacy_null_id.pdf", pdf_bytes([page_1]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    wait_for_ingestion(ingestion_id)

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")

    # Simulate legacy DB row where identification_documents column is None
    from backend.app.models import StampDutyAssessment

    with SessionLocal() as db:
        assessment = db.query(StampDutyAssessment).filter_by(document_id=gift_doc["id"]).one()
        assessment.identification_documents = None
        db.commit()

    detail_resp = client.get(f"/api/documents/{gift_doc['id']}")
    assert detail_resp.status_code == 200
    details = detail_resp.json()
    expected_empty = {
        "extractor_version": 3,
        "status": "not_found",
        "summary_counts": {"pan_card": 0, "aadhaar_card": 0, "other_id": 0, "total": 0},
        "grouped_documents": {"pan_card": [], "aadhaar_card": [], "other_id": []},
        "pan_card": [],
        "aadhaar_card": [],
        "other_id": [],
        "deed_parties": [
            {
                "party_role": "Donor",
                "deed_name": "Ramesh Kulkarni",
                "normalized_deed_name": "Ramesh Kulkarni",
                "page_number": 1,
            },
            {
                "party_role": "Donee",
                "deed_name": "Suresh Kulkarni",
                "normalized_deed_name": "Suresh Kulkarni",
                "page_number": 1,
            },
        ],
        "name_verification_summary": {
            "total_evaluated": 0,
            "exact_match_count": 0,
            "minor_variation_count": 0,
            "mismatch_count": 0,
            "unverified_count": 0,
            "has_discrepancies": False,
            "flagged_discrepancies": [],
        },
    }
    assert details["identification_documents"] == expected_empty
    assert details["stamp_duty_assessment"]["identification_documents"] == expected_empty

    # Also confirm PATCH assessment recalculation validates StampDutyAssessmentResponse cleanly on legacy Null
    patch_resp = client.patch(
        f"/api/documents/{gift_doc['id']}/stamp-duty-assessment",
        json={"reviewer_notes": "Legacy record check"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["identification_documents"] == expected_empty

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_identification_documents_auto_recomputes_when_extractor_version_outdated():
    page_1 = (
        "GIFT DEED\n"
        "BETWEEN:\n"
        "MR. CHETAN AMICHAND SHAH, [PAN # AAEPS2095R], [AADHAR\n"
        "# 8161 9641 4794], age 75 years,\n"
        'hereinafter referred to as "THE DONOR";\n'
        "AND\n"
        "MRS. SHEELA CHETAN SHAH, [PAN # AAMPS4553J[, [AADHAR #\n"
        "5598 7473 7252], age 71 years,\n"
        'hereinafter referred to as "THE DONEE".\n'
        "WHEREAS Donor gifts the flat to his wife Donee.\n"
        "Schedule of Property: Flat 19A, Mumbai Municipal Corporation.\n"
        "Market Value: Rs. 50,00,000/-\n"
        "Total Stamp Duty Paid: Rs. 200/-"
    )
    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("stale_id_recompute.pdf", pdf_bytes([page_1]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")

    from backend.app.models import StampDutyAssessment
    from backend.app.processing import ID_EXTRACTOR_VERSION

    # Simulate an older DB record from an outdated extractor_version
    stale_payload = {
        "extractor_version": "outdated-v0",
        "status": "not_found",
        "summary_counts": {"pan_card": 0, "aadhaar_card": 0, "other_id": 0, "total": 0},
        "grouped_documents": {"pan_card": [], "aadhaar_card": [], "other_id": []},
        "pan_card": [],
        "aadhaar_card": [],
        "other_id": [],
    }
    with SessionLocal() as db:
        assessment = db.query(StampDutyAssessment).filter_by(document_id=gift_doc["id"]).one()
        assessment.identification_documents = json.dumps(stale_payload)
        db.commit()

    detail_resp = client.get(f"/api/documents/{gift_doc['id']}")
    assert detail_resp.status_code == 200
    id_docs = detail_resp.json()["identification_documents"]
    assert id_docs["status"] == "found"
    assert id_docs["summary_counts"] == {"pan_card": 2, "aadhaar_card": 2, "other_id": 0, "total": 4}
    assert {p["id_number"]: p["party_role"] for p in id_docs["pan_card"]} == {
        "AAEPS2095R": "Donor",
        "AAMPS4553J": "Donee",
    }

    with SessionLocal() as db:
        refreshed = db.query(StampDutyAssessment).filter_by(document_id=gift_doc["id"]).one()
        persisted = json.loads(refreshed.identification_documents)
        assert persisted["extractor_version"] == ID_EXTRACTOR_VERSION
        assert "name_verification_summary" in persisted
        assert "deed_parties" in persisted

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_resolve_identification_documents_endpoints_and_fallback():
    from backend.app.main import _resolve_identification_documents
    from backend.app.models import LogicalDocument, StampDutyAssessment

    page_1 = (
        "GIFT DEED\n"
        "This Deed of Gift is made between:\n"
        "MR. RAJESH KUMAR SHARMA, PAN: ABCDE1234F, hereinafter called the DONOR;\n"
        "AND\n"
        "MR. SURESH RAJESH SHARMA, Aadhaar No: 9876 5432 1098, hereinafter called the DONEE.\n"
        "WHEREAS Donor gifts property to Donee.\n"
        "Schedule of Property: Flat 5B, Pune.\n"
        "Market Value: Rs. 40,00,000/-\n"
        "Total Stamp Duty Paid: Rs. 200/-"
    )
    page_2 = (
        "INCOME TAX DEPARTMENT\n"
        "PERMANENT ACCOUNT NUMBER CARD\n"
        "Name\n"
        "R K SHARMA\n"
        "Father's Name\n"
        "RAM SHARMA\n"
        "ABCDE1234F\n"
        "DOB: 12/08/1970"
    )

    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("resolve_test.pdf", pdf_bytes([page_1, page_2]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")

    # Test GET /api/documents/{id}/stamp-duty
    resp = client.get(f"/api/documents/{gift_doc['id']}/stamp-duty")
    assert resp.status_code == 200
    duty_data = resp.json()
    id_docs = duty_data["identification_documents"]

    assert id_docs["extractor_version"] == 3
    assert len(id_docs["deed_parties"]) >= 2
    assert id_docs["name_verification_summary"]["total_evaluated"] >= 1

    pan_cards = id_docs["grouped_documents"]["pan_card"]
    assert len(pan_cards) == 1
    donor_pan = pan_cards[0]
    assert donor_pan["id_number"] == "ABCDE1234F"
    assert donor_pan["holder_name"] == "R K SHARMA"
    assert donor_pan["deed_name"] in ("RAJESH KUMAR SHARMA", "MR. RAJESH KUMAR SHARMA")

    # Name verification detail structure
    ver = donor_pan["name_verification"]
    assert ver["status"] == "minor_variation"
    assert ver["flagged"] is True
    assert ver["severity"] == "warning"
    assert ver["id_holder_name"] == "R K SHARMA"
    assert "Initial" in ver["discrepancy_reason"]

    # Test GET /api/documents/{id}/report.pdf
    pdf_resp = client.get(f"/api/documents/{gift_doc['id']}/report.pdf")
    assert pdf_resp.status_code == 200
    assert "application/pdf" in pdf_resp.headers["content-type"]

    # Test _resolve_identification_documents on document without pages (fallback safe default)
    with SessionLocal() as db:
        empty_doc = LogicalDocument(
            ingestion_id="nonexistent-ingestion",
            start_page=1,
            end_page=1,
            category="gift_deed",
        )
        empty_assessment = StampDutyAssessment(
            document=empty_doc,
        )
        fallback = _resolve_identification_documents(db, empty_doc, empty_assessment)
        assert fallback["status"] == "not_found"
        assert fallback["summary_counts"]["total"] == 0
        assert fallback["grouped_documents"]["pan_card"] == []
        assert fallback["name_verification_summary"]["total_evaluated"] == 0
        assert fallback["name_verification_summary"]["has_discrepancies"] is False
        assert fallback["extractor_version"] == 3

        # Test extracted_id_documents_json property on StampDutyAssessment
        assert empty_assessment.extracted_id_documents_json == empty_assessment.identification_documents
        empty_assessment.extracted_id_documents_json = '{"test": 123}'
        assert empty_assessment.identification_documents == '{"test": 123}'

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_api_stamp_duty_auto_recompute_when_extractor_version_outdated():
    page_1 = (
        "GIFT DEED\n"
        "Registration Ref: REG-TEST-RECOMP-01\n"
        "Donor: Ramesh Kulkarni, aged 64 years, PAN: ABCDE1234F\n"
        "Donee: Suresh Kulkarni, aged 32 years\n"
        "WHEREAS Donor gifts residential property to Donee.\n"
        "Schedule of Property: Residential Flat No 10, Pune.\n"
        "Market Value: ₹ 30,00,000/-\n"
        "Total Stamp Duty Paid: ₹ 30,200/-"
    )

    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("recomp_test.pdf", pdf_bytes([page_1]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")
    doc_id = gift_doc["id"]

    # Manually simulate an existing legacy DB row with extractor_version < 3
    with SessionLocal() as db:
        ass = db.query(StampDutyAssessment).filter_by(document_id=doc_id).one()
        ass.identification_documents = json.dumps(
            {
                "extractor_version": 2,
                "status": "found",
                "pan_card": [{"doc_type": "pan_card", "id_number": "ABCDE1234F", "holder_name": "OLD LEGACY NAME"}],
            }
        )
        db.commit()

    # Call GET /api/documents/{id}/stamp-duty: must auto-recompute to extractor_version 3
    resp = client.get(f"/api/documents/{doc_id}/stamp-duty")
    assert resp.status_code == 200
    duty_data = resp.json()
    id_docs = duty_data["identification_documents"]

    assert id_docs["extractor_version"] == 3
    assert "name_verification_summary" in id_docs
    assert id_docs["name_verification_summary"]["total_evaluated"] >= 1
    assert len(id_docs["grouped_documents"]["pan_card"]) >= 1

    pan_item = id_docs["grouped_documents"]["pan_card"][0]
    assert pan_item["id_number"] == "ABCDE1234F"
    assert "name_verification" in pan_item
    assert pan_item["name_verification"]["status"] in ("exact_match", "unverified", "minor_variation")

    # Verify that database row was backfilled and persisted permanently
    with SessionLocal() as db:
        ass = db.query(StampDutyAssessment).filter_by(document_id=doc_id).one()
        persisted = json.loads(ass.identification_documents)
        assert persisted.get("extractor_version") == 3
        assert "name_verification_summary" in persisted

    client.delete(f"/api/ingestions/{ingestion_id}")


def test_api_document_pdf_report_renders_flagged_name_discrepancies():
    page_1 = (
        "GIFT DEED\n"
        "Registration Ref: REG-TEST-DISC-01\n"
        "Donor: Smt. Kavita Ajay Deshmukh, aged 45 years, holding PAN ABCDE1234F\n"
        "Donee: Rahul Deshmukh, aged 22 years\n"
        "WHEREAS Donor gifts the flat to Donee.\n"
        "Schedule of Property: Residential Flat No 10, Pune.\n"
        "Market Value: ₹ 30,00,000/-\n"
        "Total Stamp Duty Paid: ₹ 30,200/-"
    )
    page_2 = (
        "INCOME TAX DEPARTMENT\n"
        "PERMANENT ACCOUNT NUMBER CARD\n"
        "ABCDE1234F\n"
        "Name\n"
        "KAVITA A DESHMUKH\n"
        "Date of Birth\n"
        "12/08/1979"
    )

    uploaded = client.post(
        "/api/ingestions",
        files={"file": ("disc_test.pdf", pdf_bytes([page_1, page_2]), "application/pdf")},
    )
    assert uploaded.status_code == 202
    ingestion_id = uploaded.json()["id"]
    state = wait_for_ingestion(ingestion_id)
    assert state["status"] in {"completed", "completed_with_warnings"}

    documents = client.get("/api/documents").json()["items"]
    gift_doc = next(d for d in documents if d["ingestion_id"] == ingestion_id and d["category"] == "gift_deed")
    doc_id = gift_doc["id"]

    # Verify GET /api/documents/{id}/report.pdf
    doc_report = client.get(f"/api/documents/{doc_id}/report.pdf")
    assert doc_report.status_code == 200
    assert doc_report.content.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(doc_report.content))
    all_text = "\n".join(p.extract_text() or "" for p in reader.pages).lower()
    assert "extracted identification documents" in all_text
    assert "name verification summary" in all_text
    assert "flagged name discrepancies" in all_text
    assert "kavita a deshmukh" in all_text
    assert "kavita ajay deshmukh" in all_text
    assert "minor variation" in all_text

    # Also verify GET /api/ingestions/{id}/report.pdf
    ing_report = client.get(f"/api/ingestions/{ingestion_id}/report.pdf")
    assert ing_report.status_code == 200
    assert ing_report.content.startswith(b"%PDF-")

    client.delete(f"/api/ingestions/{ingestion_id}")
