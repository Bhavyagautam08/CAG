import difflib
import importlib.metadata
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import get_settings

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", by_alias=True, exclude_none=True))
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _score_signals(value: Any, page_number: int | None = None) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    if isinstance(value, dict):
        current_page = value.get("page_no", value.get("page_number", page_number))
        for key, item in value.items():
            if re.search(r"(?:rec_scores?|confidence|scores?)$", str(key), re.IGNORECASE):
                if isinstance(item, (int, float)):
                    signals.append({"name": str(key), "value": item, "page_number": current_page})
                elif isinstance(item, list):
                    for score in item:
                        if isinstance(score, (int, float)):
                            signals.append({"name": str(key), "value": score, "page_number": current_page})
            else:
                signals.extend(_score_signals(item, current_page if isinstance(current_page, int) else page_number))
    elif isinstance(value, list):
        for item in value:
            signals.extend(_score_signals(item, page_number))
    return signals


def _bbox(value: dict[str, Any]) -> Any:
    for key in ("bbox", "block_bbox", "coordinate", "coordinates", "points", "poly"):
        candidate = value.get(key)
        if isinstance(candidate, (list, dict)):
            return _jsonable(candidate)
    return None


def _paddle_blocks(value: Any, page_number: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    text_keys = ("rec_text", "block_content", "text_content", "text")

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            text = next((item.get(key) for key in text_keys if isinstance(item.get(key), str) and item[key].strip()), None)
            if text is not None:
                label = item.get("block_label", item.get("label", item.get("type", "text")))
                kind = str(label)
                bbox = _bbox(item)
                identity = (text, kind, json.dumps(bbox, ensure_ascii=False, sort_keys=True))
                scores = [
                    item[key]
                    for key in ("rec_score", "score", "confidence")
                    if isinstance(item.get(key), (int, float))
                ]
                if identity not in seen:
                    blocks.append(
                        {
                            "text": text,
                            "kind": kind,
                            "bbox": bbox,
                            "confidence": scores[0] if scores else None,
                            "page_number": page_number,
                            "reading_order": item.get("block_order", item.get("order")),
                        }
                    )
                    seen.add(identity)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return blocks


def _run_paddle(source_pdf: Path, output_dir: Path) -> dict[str, Any]:
    from paddleocr import PPStructureV3

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = {
        "input": "original PDF path",
        "pipeline": "PPStructureV3",
        "local_processing": True,
        "paddle_ocr_language": get_settings().paddle_comparison_language,
        "cpu_threads": 2,
        "one_dnn_enabled": False,
        "formula_recognition_enabled": False,
        "remote_service": False,
    }
    pipeline = PPStructureV3(
        lang=get_settings().paddle_comparison_language,
        enable_mkldnn=False,
        cpu_threads=2,
        use_chart_recognition=False,
        use_formula_recognition=False,
    )
    raw_results: list[Any] = []
    markdown_parts: list[str] = []
    pages: list[dict[str, Any]] = []
    for index, result in enumerate(pipeline.predict(input=str(source_pdf)), start=1):
        result_json = getattr(result, "json", None)
        if callable(result_json):
            result_json = result_json()
        if result_json is None and hasattr(result, "to_dict"):
            result_json = result.to_dict()
        native = _jsonable(result_json if result_json is not None else result)
        raw_results.append(native)
        markdown = getattr(result, "markdown", None)
        if callable(markdown):
            markdown = markdown()
        if isinstance(markdown, dict):
            markdown = markdown.get("markdown_text", markdown.get("text", ""))
        if isinstance(markdown, str) and markdown.strip():
            markdown_parts.append(markdown)
        page_index = index
        if isinstance(native, dict):
            result_data = native.get("res", native)
            page_id = result_data.get("page_index") if isinstance(result_data, dict) else None
            if isinstance(page_id, int):
                page_index = page_id + 1
        result_data = native.get("res", native) if isinstance(native, dict) else native
        parsing_blocks = result_data.get("parsing_res_list") if isinstance(result_data, dict) else None
        blocks = _paddle_blocks(parsing_blocks if isinstance(parsing_blocks, list) else native, page_index)
        pages.append(
            {
                "page_number": page_index,
                "text": "\n".join(block["text"] for block in blocks),
                "blocks": blocks,
                "tables": [block for block in blocks if "table" in block["kind"].casefold()],
                "native_confidence_signals": _score_signals(native, page_index),
            }
        )
    markdown_text = "\n\n".join(markdown_parts)
    native_path = output_dir / "native.json"
    markdown_path = output_dir / "document.md"
    normalized_path = output_dir / "normalized.json"
    native_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    normalized = {"engine": "pp_structure", "pages": pages}
    normalized_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    confidence = {
        "source": "PP-StructureV3 native output",
        "signals": [signal for page in pages for signal in page["native_confidence_signals"]],
        "interpretation": "Native model scores; not calibrated accuracy probabilities.",
    }
    config["duration_seconds"] = round(time.perf_counter() - started, 3)
    return {
        "package_version": _version("paddleocr"),
        "config": config,
        "confidence": confidence,
        "native_path": str(native_path),
        "markdown_path": str(markdown_path),
        "normalized_path": str(normalized_path),
        "page_count": len(pages),
        "duration_seconds": time.perf_counter() - started,
    }


def _docling_bbox(provenance: Any) -> Any:
    box = getattr(provenance, "bbox", None)
    if box is None:
        return None
    return _jsonable(box)


def _run_docling(source_pdf: Path, output_dir: Path) -> dict[str, Any]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        EasyOcrOptions,
        PdfPipelineOptions,
        TableStructureOptions,
        TableFormerMode,
    )
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.document_converter import DocumentConverter, PdfFormatOption

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_dir = (Path(get_settings().storage_dir) / ".models" / "docling-easyocr").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=False,
        allow_external_plugins=False,
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        ),
        ocr_options=EasyOcrOptions(
            lang=["en", "hi", "mr"],
            model_storage_directory=str(model_dir),
            download_enabled=True,
        ),
        accelerator_options=AcceleratorOptions(num_threads=2, device="cpu"),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options,
            ),
        }
    )
    conversion = converter.convert(source_pdf)
    document = conversion.document
    native = document.export_to_dict(mode="json", by_alias=True, exclude_none=True)
    markdown = document.export_to_markdown()
    pages_by_number: dict[int, dict[str, Any]] = {}
    for item, level in document.iterate_items():
        provenance_items = getattr(item, "prov", None) or []
        text = getattr(item, "text", None)
        label = getattr(getattr(item, "label", None), "value", getattr(item, "label", "item"))
        table = "table" in str(label).casefold()
        for provenance in provenance_items:
            page_number = getattr(provenance, "page_no", None)
            if not isinstance(page_number, int):
                continue
            page = pages_by_number.setdefault(
                page_number,
                {"page_number": page_number, "text": "", "blocks": [], "tables": []},
            )
            block = {
                "text": text if isinstance(text, str) else "",
                "kind": str(label),
                "bbox": _docling_bbox(provenance),
                "confidence": None,
                "reading_order": len(page["blocks"]),
                "hierarchy_level": level,
            }
            page["blocks"].append(block)
            if block["text"]:
                page["text"] += ("\n" if page["text"] else "") + block["text"]
            if table:
                block["native_table"] = _jsonable(item)
                page["tables"].append(block)
    normalized = {
        "engine": "docling",
        "pages": [pages_by_number[key] for key in sorted(pages_by_number)],
    }
    native_path = output_dir / "native.json"
    markdown_path = output_dir / "document.md"
    normalized_path = output_dir / "normalized.json"
    native_path.write_text(json.dumps(native, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    normalized_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    confidence_report = getattr(document, "confidence_report", None)
    confidence = _jsonable(confidence_report) if confidence_report is not None else {
        "availability": "not_available",
        "reason": "This Docling conversion result exposes no native confidence report.",
    }
    config = {
        "input": "original PDF path",
        "pipeline": "Docling DocumentConverter",
        "pdf_backend": "PyPdfiumDocumentBackend",
        "ocr_engine": "EasyOCR",
        "ocr_languages": ["en", "hi", "mr"],
        "table_mode": "accurate",
        "remote_services": False,
        "external_plugins": False,
        "accelerator": "cpu",
        "threads": 2,
    }
    config["duration_seconds"] = round(time.perf_counter() - started, 3)
    return {
        "package_version": _version("docling"),
        "config": config,
        "confidence": confidence,
        "native_path": str(native_path),
        "markdown_path": str(markdown_path),
        "normalized_path": str(normalized_path),
        "page_count": len(normalized["pages"]),
        "duration_seconds": time.perf_counter() - started,
    }


def _run_engine_direct(engine: str, source_pdf: Path, output_dir: Path) -> dict[str, Any]:
    if engine == "pp_structure":
        return _run_paddle(source_pdf, output_dir)
    if engine == "docling":
        return _run_docling(source_pdf, output_dir)
    raise ValueError(f"Unsupported document engine: {engine}")


def run_engine(engine: str, source_pdf: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "_worker-result.json"
    result_path.unlink(missing_ok=True)
    worker_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "backend.app.comparison_worker",
                engine,
                str(source_pdf.resolve()),
                str(output_dir.resolve()),
            ],
            cwd=worker_root,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{engine} worker exceeded the 60-minute processing limit.") from exc
    if completed.returncode != 0:
        details = completed.stderr.strip()[-2000:]
        message = f"{engine} worker exited with code {completed.returncode}."
        if details:
            message = f"{message} {details}"
        raise RuntimeError(message)
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{engine} worker completed without a valid result file.") from exc
    finally:
        result_path.unlink(missing_ok=True)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _differences(left: str, right: str, mode: str) -> list[dict[str, Any]]:
    if mode == "character":
        left_values, right_values = list(left), list(right)
    elif mode == "line":
        left_values, right_values = [line.strip() for line in left.splitlines() if line.strip()], [line.strip() for line in right.splitlines() if line.strip()]
    elif mode == "paragraph":
        left_values = [part.strip() for part in re.split(r"\n\s*\n", left) if part.strip()]
        right_values = [part.strip() for part in re.split(r"\n\s*\n", right) if part.strip()]
    else:
        left_values, right_values = _normalize_text(left).split(), _normalize_text(right).split()
    matcher = difflib.SequenceMatcher(a=left_values, b=right_values)
    return [
        {
            "operation": operation,
            "left": "".join(left_values[left_start:left_end]) if mode == "character" else " ".join(left_values[left_start:left_end]),
            "right": "".join(right_values[right_start:right_end]) if mode == "character" else " ".join(right_values[right_start:right_end]),
        }
        for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes()
        if operation != "equal"
    ]


def build_page_comparison(pp_normalized: dict[str, Any], docling_normalized: dict[str, Any]) -> dict[str, Any]:
    by_engine: dict[str, dict[int, dict[str, Any]]] = {}
    for engine_name, normalized in (("pp_structure", pp_normalized), ("docling", docling_normalized)):
        by_engine[engine_name] = {
            page["page_number"]: page
            for page in normalized.get("pages", [])
            if isinstance(page.get("page_number"), int)
        }
    page_numbers = sorted(set(by_engine["pp_structure"]) | set(by_engine["docling"]))
    pages = []
    for page_number in page_numbers:
        pp_page = by_engine["pp_structure"].get(page_number, {})
        docling_page = by_engine["docling"].get(page_number, {})
        pp_text = pp_page.get("text", "")
        docling_text = docling_page.get("text", "")
        pages.append(
            {
                "page_number": page_number,
                "pp_structure_text": pp_text,
                "docling_text": docling_text,
                "text_equal_after_whitespace_normalization": _normalize_text(pp_text) == _normalize_text(docling_text),
                "character_differences": _differences(pp_text, docling_text, "character"),
                "word_differences": _differences(pp_text, docling_text, "word"),
                "line_differences": _differences(pp_text, docling_text, "line"),
                "paragraph_differences": _differences(pp_text, docling_text, "paragraph"),
                "pp_structure_blocks": pp_page.get("blocks", []),
                "docling_blocks": docling_page.get("blocks", []),
                "pp_structure_tables": pp_page.get("tables", []),
                "docling_tables": docling_page.get("tables", []),
            }
        )
    return {
        "accuracy": "Not Measured: no verified ground-truth annotations were supplied.",
        "confidence_note": "Native confidence values are engine-specific and are not directly comparable.",
        "pages": pages,
    }
