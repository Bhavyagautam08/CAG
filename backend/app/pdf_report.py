import hashlib
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pypdf import PdfReader
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .models import Ingestion, LogicalDocument, Page, StampDutyAssessment
from .ocr import detect_table_lines
from .pdf_utils import LocalImageBundle

logger = logging.getLogger(__name__)


class IngestionNotCompleteError(RuntimeError):
    pass


class SourceChangedError(RuntimeError):
    pass


PAGE_HEIGHT = 780
PAGE_MARGIN = 24
OCR_LAYOUT_MAX_HEIGHT = 690
OCR_LAYOUT_MAX_WIDTH = 760
PANEL_WIDTH = 390
PANEL_BACKGROUND = HexColor("#f5f7fb")
INK = HexColor("#17233b")
MUTED = HexColor("#607086")
ACCENT = HexColor("#2458a6")
RED = HexColor("#b42318")
RED_BACKGROUND = HexColor("#fff0ee")
MAX_FIELD_LINES_PER_SHEET = 50


def _register_unicode_font() -> str:
    candidates = [
        (Path(r"C:\Windows\Fonts\Nirmala.ttc"), 0),
        (Path(r"C:\Windows\Fonts\segoeui.ttf"), 0),
        (Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"), 0),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 0),
    ]
    for path, subfont_index in candidates:
        if not path.is_file():
            continue
        font_name = f"CaseworkReport-{path.stem}-{subfont_index}"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(path), subfontIndex=subfont_index))
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("Could not register PDF report font %s: %s", path, exc)
                continue
        return font_name
    logger.warning("No Unicode PDF report font was found; using built-in Helvetica.")
    return "Helvetica"


def _wrap(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if pdfmetrics.stringWidth(word, font_name, font_size) <= max_width:
            current = word
            continue
        chunk = ""
        for character in word:
            if chunk and pdfmetrics.stringWidth(chunk + character, font_name, font_size) > max_width:
                lines.append(chunk)
                chunk = ""
            chunk += character
        current = chunk
    if current:
        lines.append(current)
    return lines


CURRENCY_FIELD_NAMES = {
    "market_value",
    "stamp_duty_paid",
    "metro_cess_paid",
    "total_stamp_duty_paid",
    "expected_stamp_duty",
    "stamp_duty_difference",
}

_FIELD_LABEL_OVERRIDES = {
    "donor_name": "Donor Name",
    "donee_name": "Donee Name",
    "donor_donee_relationship": "Donor-Donee Relationship",
    "relationship_value": "Relationship (Recital)",
    "relationship_eligibility": "Article 34 Eligibility",
    "property_type": "Property Type",
    "property_description": "Property Description",
    "property_location": "Property Location",
    "survey_number": "Survey / Gat Number",
    "area": "Property Area",
    "property_area": "Property Area",
    "property_id": "Property ID",
    "registration_number": "Registration Number",
    "registration_date": "Registration Date",
    "document_date": "Document Date",
    "market_value": "Market / Ready Reckoner Value",
    "stamp_duty_paid": "Stamp Duty Paid",
    "metro_cess_paid": "Metro Cess Paid",
    "total_stamp_duty_paid": "Total Stamp Duty Paid",
    "challan_number": "Challan / Receipt Number",
    "expected_stamp_duty": "Expected Stamp Duty (Art. 34)",
    "stamp_duty_difference": "Stamp Duty Difference",
    "stamp_duty_status": "Stamp Duty Assessment Status",
}


def _format_rupees(amount: Any) -> str:
    if amount is None or amount == "":
        return "Not available"
    try:
        val = int(amount)
    except (ValueError, TypeError):
        try:
            val = int(float(amount))
        except (ValueError, TypeError):
            return str(amount)
    is_neg = val < 0
    s = str(abs(val))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3
    return f"-₹{formatted}" if is_neg else f"₹{formatted}"


def _humanize(value: str) -> str:
    if value in _FIELD_LABEL_OVERRIDES:
        return _FIELD_LABEL_OVERRIDES[value]
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip().title()


def _display_field_value(field: Any) -> str:
    value = field.verified_value if field.verified_value is not None else field.extracted_value
    if field.verification_status == "not_available":
        return "Not available - not located in extracted text; verify the source."
    if value is None or value == "":
        return "No value extracted"
    if field.sensitive:
        return f"••••{value[-4:]}" if len(value) > 4 else "••••"
    if field.field_name in CURRENCY_FIELD_NAMES:
        formatted = _format_rupees(value)
        if field.field_name == "stamp_duty_difference":
            try:
                diff_val = int(float(value))
                if diff_val < 0:
                    return f"{formatted} (Short Paid)"
                else:
                    return f"{formatted} (Compliant)"
            except (ValueError, TypeError):
                pass
        return formatted
    if field.field_name == "property_type":
        labels = {
            "residential_property": "Residential Property (Eligible)",
            "agricultural_land": "Agricultural Land (Eligible)",
            "other_property": "Commercial / Other (Non-concessional)",
            "unknown": "Unknown / Mixed Indicators",
        }
        return labels.get(value, _humanize(value))
    if field.field_name == "relationship_eligibility":
        labels = {
            "eligible": "Eligible (Article 34 Concession)",
            "not_eligible": "Not Eligible (Standard Conveyance Applies)",
            "ambiguous": "Ambiguous Recital (Review Required)",
            "unknown": "Unknown / Not Explicitly Stated",
        }
        return labels.get(value, _humanize(value))
    if field.field_name == "stamp_duty_status":
        labels = {
            "compliant": "Compliant",
            "short_paid": "Short Paid (Deficit)",
            "requires_review": "Requires Review",
            "not_applicable": "Not Applicable (Concession Ineligible)",
            "not_available": "Not Available",
        }
        return labels.get(value, _humanize(value))
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_bundle_sha256(bundle: LocalImageBundle) -> tuple[str, int]:
    digest = hashlib.sha256(b"casework-image-bundle-v1")
    total_size = 0
    for path in bundle.paths:
        data = path.read_bytes()
        total_size += len(data)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest(), total_size


def _image_page_pdf(path: Path) -> io.BytesIO:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.load()
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, "white")
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            image = flattened
        else:
            image = image.convert("RGB")
    width, height = image.size
    points_per_pixel = 72 / 200
    output = io.BytesIO()
    image_canvas = canvas.Canvas(
        output,
        pagesize=(width * points_per_pixel, height * points_per_pixel),
        pageCompression=1,
    )
    image_canvas.drawImage(
        ImageReader(image),
        0,
        0,
        width=width * points_per_pixel,
        height=height * points_per_pixel,
    )
    image_canvas.save()
    output.seek(0)
    return output


def _report_image_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        (Path(r"C:\Windows\Fonts\Nirmala.ttc"), 0),
        (Path(r"C:\Windows\Fonts\segoeui.ttf"), 0),
        (Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"), 0),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 0),
    ]
    for path, subfont_index in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), max(1, size), index=subfont_index)
            except (OSError, ValueError):
                logger.warning("Could not load OCR report image font %s", path)
    return ImageFont.load_default(size=max(1, size))


def _safe_page_image_path(raw_path: str | None, storage_root: Path) -> Path | None:
    if not raw_path:
        return None
    image_path = Path(raw_path).resolve()
    if storage_root not in image_path.parents:
        logger.warning("Ignoring page image outside private storage: %s", image_path)
        return None
    return image_path if image_path.is_file() else None


def _ocr_layout_image(
    page: Any,
    source_page: Any,
    page_text: str,
    page_number: int,
    storage_root: Path,
    image_bundle: LocalImageBundle | None,
) -> Image.Image:
    words: list[dict[str, Any]] = []
    table_lines: list[dict[str, Any]] = []
    if page and page.ocr_word_boxes:
        try:
            parsed = json.loads(page.ocr_word_boxes)
            if isinstance(parsed, list):
                words = [entry for entry in parsed if isinstance(entry, dict)]
            elif isinstance(parsed, dict):
                raw_words = parsed.get("words", [])
                if isinstance(raw_words, list):
                    words = [entry for entry in raw_words if isinstance(entry, dict)]
                raw_lines = parsed.get("table_lines", [])
                if isinstance(raw_lines, list):
                    table_lines = [entry for entry in raw_lines if isinstance(entry, dict)]
        except json.JSONDecodeError:
            logger.warning("Page %s has invalid OCR word-box data; using text-only layout", page_number)

    image_path = _safe_page_image_path(page.processed_image_path, storage_root) if page else None
    if image_path is None and page:
        image_path = _safe_page_image_path(page.image_path, storage_root)
    if image_path is None and image_bundle and page_number <= len(image_bundle.paths):
        image_path = image_bundle.paths[page_number - 1]

    if image_path:
        try:
            with Image.open(image_path) as source:
                dimensions = ImageOps.exif_transpose(source).size
        except OSError as exc:
            logger.warning("Could not read page image dimensions for OCR report page %s: %s", page_number, exc)
            image_path = None
            dimensions = None
    else:
        dimensions = None

    max_x = max(
        (
            float(word["x"]) + max(0.0, float(word["width"]))
            for word in words
            if all(
                isinstance(word.get(key), (int, float)) and math.isfinite(float(word[key]))
                for key in ("x", "width")
            )
        ),
        default=0.0,
    )
    max_y = max(
        (
            float(word["y"]) + max(0.0, float(word["height"]))
            for word in words
            if all(
                isinstance(word.get(key), (int, float)) and math.isfinite(float(word[key]))
                for key in ("y", "height")
            )
        ),
        default=0.0,
    )
    if dimensions:
        source_width, source_height = dimensions
    else:
        source_width = max(1, round(float(source_page.mediabox.width) * 2), math.ceil(max_x + 32))
        source_height = max(1, round(float(source_page.mediabox.height) * 2), math.ceil(max_y + 32))

    image_scale = min(1.0, 1600 / max(source_width, source_height))
    width = max(1, round(source_width * image_scale))
    height = max(1, round(source_height * image_scale))
    layout = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(layout)

    if not table_lines and image_path:
        table_lines = detect_table_lines(image_path)
    line_width = max(1, round(2 * image_scale))
    for line in table_lines:
        coordinates = [line.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in coordinates
        ):
            continue
        draw.line(
            tuple(round(float(value) * image_scale) for value in coordinates),
            fill=(17, 17, 17),
            width=line_width,
        )

    positioned_words = [
        word for word in words
        if all(
            isinstance(word.get(key), (int, float)) and math.isfinite(float(word[key]))
            for key in ("x", "y", "width", "height")
        )
        and word["width"] > 0
        and word["height"] > 0
        and str(word.get("text", "")).strip()
    ]
    if positioned_words:
        for word in positioned_words:
            text = str(word["text"]).strip()
            font_size = max(
                6,
                round(
                    min(
                        float(word["height"]) * 0.72,
                        float(word["width"]) / max(1, len(text) * 0.58),
                    )
                    * image_scale
                ),
            )
            font = _report_image_font(font_size)
            draw.text(
                (
                    round(float(word["x"]) * image_scale),
                    round((float(word["y"]) + float(word["height"]) * 0.82) * image_scale),
                ),
                text,
                fill=(17, 17, 17),
                font=font,
                anchor="ls",
            )
    elif page_text.strip():
        margin = max(18, round(width * 0.045))
        font_size = max(10, min(26, round(width / 48)))
        font = _report_image_font(font_size)
        line_height = font_size * 1.45
        cursor_y = margin
        max_line_width = max(1, width - margin * 2)
        for paragraph in page_text.splitlines():
            current = ""
            for token in paragraph.split():
                candidate = f"{current} {token}".strip()
                if current and draw.textlength(candidate, font=font) > max_line_width:
                    draw.text((margin, cursor_y), current, fill=(17, 17, 17), font=font)
                    current = token
                    cursor_y += line_height
                else:
                    current = candidate
            if current:
                draw.text((margin, cursor_y), current, fill=(17, 17, 17), font=font)
                cursor_y += line_height
            else:
                cursor_y += line_height
    else:
        font = _report_image_font(max(10, min(24, round(width / 48))))
        draw.text((max(18, round(width * 0.045)), max(18, round(height * 0.045))), "No OCR text is available.", fill=(17, 17, 17), font=font)
    return layout


def _field_rows(fields: list[Any]) -> list[tuple[str, str, bool]]:
    return [
        (
            _humanize(field.field_name),
            _display_field_value(field),
            field.verification_status == "not_available",
        )
        for field in fields
    ]


def _page_header(
    report_canvas: canvas.Canvas,
    page_width: float,
    panel_x: float,
    filename: str,
    page_number: int,
    total_pages: int,
    continuation: bool,
    unavailable_count: int,
) -> None:
    report_canvas.setFillColor(white)
    report_canvas.rect(0, 0, page_width, PAGE_HEIGHT, fill=1, stroke=0)
    report_canvas.setFillColor(PANEL_BACKGROUND)
    report_canvas.rect(panel_x, 0, page_width - panel_x, PAGE_HEIGHT, fill=1, stroke=0)
    report_canvas.setFillColor(INK)
    report_canvas.setFont("Helvetica-Bold", 9)
    report_canvas.drawString(PAGE_MARGIN, PAGE_HEIGHT - 24, "CASEWORK  /  FORMATTED EXTRACTION", shaping=True)
    report_canvas.setFillColor(MUTED)
    report_canvas.setFont("Helvetica", 8)
    report_canvas.drawRightString(page_width - PAGE_MARGIN, PAGE_HEIGHT - 24, filename[:80], shaping=True)

    report_canvas.setFillColor(ACCENT)
    report_canvas.setFont("Helvetica-Bold", 10)
    report_canvas.drawString(
        panel_x + 18,
        PAGE_HEIGHT - 53,
        f"OCR PAGE {page_number} OF {total_pages}",
        shaping=True,
    )
    if continuation:
        report_canvas.setFillColor(MUTED)
        report_canvas.setFont("Helvetica", 8)
        report_canvas.drawRightString(
            page_width - PAGE_MARGIN - 8,
            PAGE_HEIGHT - 53,
            "EXTRACTION CONTINUED",
            shaping=True,
        )
    elif unavailable_count:
        badge_width = min(PANEL_WIDTH - 36, 16 + pdfmetrics.stringWidth(
            f"{unavailable_count} FIELD(S) TO VERIFY", "Helvetica-Bold", 7
        ))
        report_canvas.setFillColor(RED_BACKGROUND)
        report_canvas.roundRect(
            panel_x + 18,
            PAGE_HEIGHT - 81,
            badge_width,
            18,
            5,
            fill=1,
            stroke=0,
        )
        report_canvas.setFillColor(RED)
        report_canvas.setFont("Helvetica-Bold", 7)
        report_canvas.drawString(
            panel_x + 26,
            PAGE_HEIGHT - 75,
            f"{unavailable_count} FIELD(S) TO VERIFY",
            shaping=True,
        )

    report_canvas.setStrokeColor(HexColor("#dce3ed"))
    report_canvas.line(panel_x + 18, PAGE_HEIGHT - 93, page_width - PAGE_MARGIN, PAGE_HEIGHT - 93)
    report_canvas.setFillColor(MUTED)
    report_canvas.setFont("Helvetica", 7)
    report_canvas.drawString(
        PAGE_MARGIN,
        14,
        "Positioned OCR image at left  |  Machine-generated text and flags; verify against the original.",
        shaping=True,
    )


def _draw_section(report_canvas: canvas.Canvas, x: float, y: float, title: str) -> float:
    report_canvas.setFillColor(INK)
    report_canvas.setFont("Helvetica-Bold", 9)
    report_canvas.drawString(x, y, title.upper(), shaping=True)
    return y - 17


def _draw_field_row(
    report_canvas: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    font_name: str,
    label: str,
    value_lines: list[str],
    missing: bool,
) -> float:
    label_width = 112
    font_size = 7.5
    leading = 10
    label_lines = _wrap(label, font_name, font_size, label_width - 12)
    row_lines = max(len(label_lines), len(value_lines))
    row_height = max(24, row_lines * leading + 10)
    report_canvas.setFillColor(RED_BACKGROUND if missing else white)
    report_canvas.roundRect(x, y - row_height + 3, width, row_height, 4, fill=1, stroke=0)
    report_canvas.setFillColor(MUTED)
    report_canvas.setFont(font_name, font_size)
    for index, line in enumerate(label_lines):
        report_canvas.drawString(x + 7, y - 12 - index * leading, line, shaping=True)
    report_canvas.setFillColor(RED if missing else INK)
    report_canvas.setFont(font_name, font_size)
    for index, line in enumerate(value_lines):
        report_canvas.drawString(x + label_width, y - 12 - index * leading, line, shaping=True)
    return y - row_height - 4


def _draw_text_line(
    report_canvas: canvas.Canvas,
    x: float,
    y: float,
    line: str,
    font_name: str,
    font_size: float = 7.2,
) -> None:
    report_canvas.setFillColor(INK)
    report_canvas.setFont(font_name, font_size)
    report_canvas.drawString(x, y, line, shaping=True)


def _draw_stamp_duty_assessment(
    report_canvas: canvas.Canvas,
    assessment: Any,
    x: float,
    y: float,
    max_width: float,
    font_name: str,
    continue_panel_fn: Any,
) -> float:
    if y - 35 < 57:
        y = continue_panel_fn("Checklist 3: Stamp-Duty Assessment (Article 34)")
    y = _draw_section(report_canvas, x, y, "Checklist 3: Stamp-Duty Assessment (Article 34)")

    status = (assessment.status or "requires_review").lower()
    review_status = (assessment.review_status or "unreviewed").lower()

    status_palette = {
        "compliant": (HexColor("#15803d"), HexColor("#dcfce7"), "COMPLIANT"),
        "short_paid": (HexColor("#b91c1c"), HexColor("#fee2e2"), "SHORT PAID"),
        "requires_review": (HexColor("#c2410c"), HexColor("#ffedd5"), "REQUIRES REVIEW"),
        "not_applicable": (HexColor("#4b5563"), HexColor("#f3f4f6"), "NOT APPLICABLE"),
        "not_available": (HexColor("#64748b"), HexColor("#f1f5f9"), "NOT AVAILABLE"),
    }
    badge_fg, badge_bg, status_label = status_palette.get(
        status, (HexColor("#4b5563"), HexColor("#f3f4f6"), status.upper())
    )

    banner_height = 28.0
    if y - banner_height < 57:
        y = continue_panel_fn("Checklist 3: Stamp-Duty Assessment (continued)")
    report_canvas.setFillColor(white)
    report_canvas.roundRect(x, y - banner_height + 3, max_width, banner_height, 4, fill=1, stroke=0)

    badge_text = f"STATUS: {status_label}"
    badge_w = min(140.0, pdfmetrics.stringWidth(badge_text, "Helvetica-Bold", 8) + 14)
    report_canvas.setFillColor(badge_bg)
    report_canvas.roundRect(x + 7, y - 22, badge_w, 18, 4, fill=1, stroke=0)
    report_canvas.setFillColor(badge_fg)
    report_canvas.setFont("Helvetica-Bold", 8)
    report_canvas.drawString(x + 14, y - 16, badge_text, shaping=True)

    review_text = f"REVIEW: {review_status.upper()}"
    report_canvas.setFillColor(MUTED)
    report_canvas.setFont(font_name, 7.5)
    report_canvas.drawRightString(x + max_width - 8, y - 16, review_text, shaping=True)

    y -= banner_height + 4

    rows: list[tuple[str, str, bool]] = []

    donor = assessment.donor_name or "Not extracted"
    donee = assessment.donee_name or "Not extracted"
    rows.append(("Parties", f"Donor: {donor}  |  Donee: {donee}", False))

    rel_raw = assessment.relationship or "Not stated in recital"
    rel_norm = assessment.relationship_normalized or "N/A"
    rows.append(("Relationship (Recital)", f"{rel_raw} (Normalized: {rel_norm})", False))

    elig_val = (assessment.relationship_eligibility or "unknown").replace("_", " ").title()
    is_elig_bad = (assessment.relationship_eligibility or "") in ("not_eligible", "ambiguous")
    rows.append(("Relationship Eligibility", elig_val, is_elig_bad))

    prop_type = (assessment.property_type or "unknown").replace("_", " ").title()
    rows.append(("Property Type", prop_type, assessment.property_type == "other_property"))

    sched_loc = assessment.property_location or "N/A"
    sched_surv = assessment.survey_number or "N/A"
    sched_area = assessment.area or "N/A"
    rows.append(("Schedule & Location", f"Location: {sched_loc} | Survey/Gat: {sched_surv} | Area: {sched_area}", False))

    if assessment.property_description:
        desc_clean = assessment.property_description.strip().replace("\n", " ")
        if len(desc_clean) > 160:
            desc_clean = desc_clean[:157] + "..."
        rows.append(("Property Description", desc_clean, False))

    mv_str = _format_rupees(assessment.market_value)
    rows.append(("Market Value (True/Declared)", mv_str, False))

    base_str = _format_rupees(assessment.base_duty) if assessment.base_duty is not None else "N/A"
    rows.append(("Base Stamp Duty (Art. 34)", f"{base_str} (concessional family rate)", False))

    if assessment.metro_cess_rate:
        cess_rate_pct = f"{int(assessment.metro_cess_rate * 100)}%"
        cess_amt_str = _format_rupees(assessment.metro_cess_amount)
        cess_str = f"{cess_amt_str} ({cess_rate_pct} of Market Value)"
    elif assessment.metro_cess_rate == 0.0:
        cess_str = "Exempt (0% - Non-Metro Jurisdiction)"
    else:
        cess_str = "N/A"
    rows.append(("Metro Cess Rate & Amount", cess_str, False))

    exp_str = _format_rupees(assessment.expected_total) if assessment.expected_total is not None else "N/A"
    rows.append(("Expected Total Stamp Duty", exp_str, False))

    base_p = _format_rupees(assessment.stamp_duty_paid) if assessment.stamp_duty_paid is not None else "N/A"
    metro_p = _format_rupees(assessment.metro_cess_paid) if assessment.metro_cess_paid is not None else "N/A"
    tot_p = _format_rupees(assessment.total_stamp_duty_paid) if assessment.total_stamp_duty_paid is not None else "N/A"
    rows.append(("Stamp Duty Paid", f"Base: {base_p} | Cess: {metro_p} | Total: {tot_p}", False))

    if assessment.challan_number:
        rows.append(("Challan / Receipt Number", str(assessment.challan_number), False))

    if assessment.difference is not None:
        diff_str = _format_rupees(assessment.difference)
        if assessment.difference < 0:
            diff_display = f"{diff_str} (Short Paid / Deficit)"
            diff_bad = True
        else:
            diff_display = f"{diff_str} (Compliant / Surplus)"
            diff_bad = False
    else:
        diff_display = "N/A"
        diff_bad = status in ("short_paid", "requires_review")
    rows.append(("Difference (Paid vs Expected)", diff_display, diff_bad))

    jur_display = (assessment.jurisdiction or "Unknown").replace("_", " ").title()
    rule_ver = assessment.legal_rule_version or "MH-ART34-2024.1"
    rows.append(("Jurisdiction & Legal Rule", f"{jur_display} | Rule Version: {rule_ver}", False))
    rows.append(("Legal Reference", "Maharashtra Stamp Act, Article 34 (Residential/Agricultural Gift to eligible family members: ₹200 base duty + 1% Metro Cess where applicable)", False))

    if assessment.source_pages:
        rows.append(("Assessment Source Pages", f"Page(s) {assessment.source_pages}", False))

    if assessment.evidence:
        ev_clean = str(assessment.evidence).strip().replace("\n", " ")
        if len(ev_clean) > 220:
            ev_clean = ev_clean[:217] + "..."
        rows.append(("Evidence Excerpt", ev_clean, False))

    if assessment.reviewer_notes:
        notes_clean = str(assessment.reviewer_notes).strip().replace("\n", " ")
        if len(notes_clean) > 220:
            notes_clean = notes_clean[:217] + "..."
        rows.append(("Reviewer Notes / Findings", notes_clean, status == "short_paid"))

    for label, val_text, is_missing_or_bad in rows:
        val_lines = _wrap(val_text, font_name, 7.5, max_width - 120)
        row_lines = max(len(_wrap(label, font_name, 7.5, 100)), len(val_lines))
        row_height = max(24, row_lines * 10 + 10)
        if y - row_height < 57:
            y = continue_panel_fn("Checklist 3: Stamp-Duty Assessment (continued)")
        y = _draw_field_row(
            report_canvas,
            x,
            y,
            max_width,
            font_name,
            label,
            val_lines,
            is_missing_or_bad,
        )

    return y


def _draw_report_page(
    report_canvas: canvas.Canvas,
    ocr_layout_image: Image.Image,
    filename: str,
    page_number: int,
    total_pages: int,
    page_text: str,
    fields: list[Any],
    page_category: str,
    page_error: str | None,
    font_name: str,
    assessment: Any = None,
) -> None:
    panel_x = PAGE_MARGIN * 2 + OCR_LAYOUT_MAX_WIDTH
    page_width = panel_x + PANEL_WIDTH + PAGE_MARGIN
    unavailable_count = sum(field.verification_status == "not_available" for field in fields)

    image_scale = min(
        OCR_LAYOUT_MAX_HEIGHT / ocr_layout_image.height,
        OCR_LAYOUT_MAX_WIDTH / ocr_layout_image.width,
    )
    image_width = ocr_layout_image.width * image_scale
    image_height = ocr_layout_image.height * image_scale
    image_x = PAGE_MARGIN + (OCR_LAYOUT_MAX_WIDTH - image_width) / 2
    image_y = (PAGE_HEIGHT - image_height) / 2 - 4

    def start_panel(continuation: bool, section_title: str | None = None) -> float:
        report_canvas.setPageSize((page_width, PAGE_HEIGHT))
        _page_header(
            report_canvas,
            page_width,
            panel_x,
            filename,
            page_number,
            total_pages,
            continuation,
            unavailable_count,
        )
        report_canvas.drawImage(
            ImageReader(ocr_layout_image),
            image_x,
            image_y,
            width=image_width,
            height=image_height,
            preserveAspectRatio=True,
            mask="auto",
        )
        cursor = PAGE_HEIGHT - (107 if unavailable_count and not continuation else 102)
        if continuation:
            cursor = _draw_section(
                report_canvas,
                panel_x + 18,
                cursor,
                section_title or "Extracted text (continued)",
            )
        return cursor

    def continue_panel(section_title: str) -> float:
        report_canvas.showPage()
        return start_panel(True, section_title)

    x = panel_x + 18
    max_width = PANEL_WIDTH - 36
    y = start_panel(False)
    fields_to_draw = _field_rows(fields)
    if page_category and page_category != "unknown":
        report_canvas.setFillColor(MUTED)
        report_canvas.setFont(font_name, 8)
        report_canvas.drawString(x, y, f"Document type: {_humanize(page_category)}", shaping=True)
        y -= 18
    y = _draw_section(report_canvas, x, y, "Extracted fields")
    if not fields_to_draw:
        report_canvas.setFillColor(MUTED)
        report_canvas.setFont(font_name, 7.5)
        report_canvas.drawString(x, y, "No labeled fields were extracted for this page.", shaping=True)
        y -= 22
    else:
        for label, value, missing in fields_to_draw:
            value_lines = _wrap(value, font_name, 7.5, max_width - 120)
            for offset in range(0, len(value_lines), MAX_FIELD_LINES_PER_SHEET):
                value_chunk = value_lines[offset:offset + MAX_FIELD_LINES_PER_SHEET]
                displayed_label = label if offset == 0 else f"{label} (continued)"
                row_lines = max(
                    len(_wrap(displayed_label, font_name, 7.5, 100)),
                    len(value_chunk),
                )
                row_height = max(24, row_lines * 10 + 10)
                if y - row_height < 57:
                    y = continue_panel("Extracted fields (continued)")
                y = _draw_field_row(
                    report_canvas,
                    x,
                    y,
                    max_width,
                    font_name,
                    displayed_label,
                    value_chunk,
                    missing,
                )

    if assessment is not None:
        y = _draw_stamp_duty_assessment(
            report_canvas,
            assessment,
            x,
            y,
            max_width,
            font_name,
            continue_panel,
        )

    if page_error:
        if y - 30 < 57:
            y = continue_panel("Page extraction warning")
        y = _draw_section(report_canvas, x, y, "Page extraction warning")
        for line in _wrap(page_error, font_name, 7.5, max_width):
            if y < 57:
                y = continue_panel("Page extraction warning (continued)")
            _draw_text_line(report_canvas, x, y, line, font_name)
            y -= 11
        y -= 8
    if y - 17 < 57:
        y = continue_panel("Complete extracted text")
    y = _draw_section(report_canvas, x, y, "Complete extracted text")
    text_lines: list[str] = []
    for paragraph in (page_text or "").splitlines():
        if not paragraph.strip():
            text_lines.append("")
        else:
            text_lines.extend(_wrap(paragraph, font_name, 7.2, max_width))
    if not text_lines:
        text_lines = ["No extracted text is available for this page."]
    for line in text_lines:
        if y < 51:
            y = continue_panel("Complete extracted text (continued)")
        _draw_text_line(report_canvas, x, y, line, font_name)
        y -= 10


def build_ingestion_report(db: Session, ingestion_id: str) -> bytes:
    ingestion = db.get(Ingestion, ingestion_id)
    if ingestion is None:
        raise LookupError("Ingestion not found.")
    if ingestion.source_type not in {"pdf", "image"}:
        raise ValueError("Formatted PDF reports are available for PDF and image ingestions.")
    if ingestion.status in {"queued", "processing"}:
        raise IngestionNotCompleteError(
            "Wait until document extraction is complete before downloading its formatted report."
        )

    storage_root = Path(get_settings().storage_dir).resolve()
    source = Path(ingestion.storage_path).resolve()
    image_bundle: LocalImageBundle | None = None
    image_pdf_streams: list[io.BytesIO] = []
    if ingestion.source_type == "pdf":
        if storage_root not in source.parents or not source.is_file() or source.suffix.casefold() != ".pdf":
            raise FileNotFoundError("The original PDF is unavailable inside private storage.")
        if _file_sha256(source) != ingestion.sha256:
            raise SourceChangedError("The original PDF has changed since ingestion; report generation was stopped.")
        reader = PdfReader(str(source), strict=False)
        if reader.is_encrypted:
            raise ValueError("Password-protected source PDFs cannot be formatted.")
        source_readers = [reader]
        source_page_count = len(reader.pages)
    else:
        if storage_root not in source.parents or source.parent.parent != storage_root:
            raise FileNotFoundError("The original image bundle is unavailable inside private storage.")
        image_bundle = LocalImageBundle(source)
        if len(image_bundle.paths) != ingestion.page_count:
            raise FileNotFoundError("The original image bundle page count has changed since ingestion.")
        actual_hash, actual_size = _image_bundle_sha256(image_bundle)
        if actual_hash != ingestion.sha256 or actual_size != ingestion.file_size:
            raise SourceChangedError("The original image bundle has changed since ingestion; report generation was stopped.")
        source_readers = []
        for image_path in image_bundle.paths:
            image_pdf_stream = _image_page_pdf(image_path)
            image_pdf_streams.append(image_pdf_stream)
            source_readers.append(PdfReader(image_pdf_stream, strict=False))
        source_page_count = len(source_readers)

    documents = db.scalars(
        select(LogicalDocument)
        .options(
            selectinload(LogicalDocument.fields),
            selectinload(LogicalDocument.stamp_duty_assessment),
        )
        .where(LogicalDocument.ingestion_id == ingestion_id)
        .order_by(LogicalDocument.start_page)
    ).all()
    pages = db.scalars(
        select(Page)
        .where(Page.ingestion_id == ingestion_id)
        .order_by(Page.page_number)
    ).all()
    pages_by_number = {page.page_number: page for page in pages}
    fields_by_page: dict[int, list[Any]] = {}
    categories_by_page: dict[int, str] = {}
    assessments_by_page: dict[int, Any] = {}
    for document in documents:
        if document.stamp_duty_assessment:
            target_ass_page = document.start_page if 1 <= document.start_page <= source_page_count else 1
            assessments_by_page[target_ass_page] = document.stamp_duty_assessment
        for page_number in range(document.start_page, document.end_page + 1):
            categories_by_page[page_number] = document.category
        for field in document.fields:
            if field.source_page and 1 <= field.source_page <= source_page_count:
                target_page = field.source_page
            else:
                target_page = document.start_page
            if field not in fields_by_page.setdefault(target_page, []):
                fields_by_page[target_page].append(field)
            if field.field_name in (
                "donor_name",
                "donee_name",
                "donor_donee_relationship",
                "property_type",
                "property_description",
                "property_location",
                "survey_number",
                "area",
                "property_area",
                "market_value",
                "relationship_value",
                "relationship_eligibility",
                "stamp_duty_paid",
                "metro_cess_paid",
                "total_stamp_duty_paid",
                "challan_number",
                "expected_stamp_duty",
                "stamp_duty_difference",
                "stamp_duty_status",
            ):
                for doc_page in range(document.start_page, document.end_page + 1):
                    if doc_page != target_page:
                        if field not in fields_by_page.setdefault(doc_page, []):
                            fields_by_page[doc_page].append(field)

    font_name = _register_unicode_font()
    overlay_bytes = io.BytesIO()
    report_canvas = canvas.Canvas(overlay_bytes, pagesize=(900, PAGE_HEIGHT), pageCompression=1)
    for page_number in range(1, source_page_count + 1):
        source_reader = source_readers[0] if ingestion.source_type == "pdf" else source_readers[page_number - 1]
        reader_page = source_reader.pages[page_number - 1] if ingestion.source_type == "pdf" else source_reader.pages[0]
        reader_page.transfer_rotation_to_content()
        page = pages_by_number.get(page_number)
        page_text = ""
        page_error = None
        if page is not None:
            page_text = page.ocr_text_corrected or page.ocr_text_original or page.digital_text or ""
            page_error = page.ocr_error
        fields = sorted(fields_by_page.get(page_number, []), key=lambda field: field.field_name)
        layout_image = _ocr_layout_image(
            page,
            reader_page,
            page_text,
            page_number,
            storage_root,
            image_bundle,
        )
        _draw_report_page(
            report_canvas,
            layout_image,
            Path(ingestion.filename).name,
            page_number,
            source_page_count,
            page_text,
            fields,
            categories_by_page.get(page_number, page.category if page else "unknown"),
            page_error,
            font_name,
            assessment=assessments_by_page.get(page_number),
        )
        if page_number < source_page_count:
            report_canvas.showPage()
    report_canvas.save()
    return overlay_bytes.getvalue()
