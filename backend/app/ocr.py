import importlib.util
import json
import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable

from PIL import Image

from .config import get_settings

logger = logging.getLogger(__name__)

_HIGH_CONFIDENCE = 85.0
_MIN_CONFIDENCE_MARGIN = 15.0
_LOW_CONFIDENCE = 55.0
_MIN_UNMATCHED_CONFIDENCE = 80.0
_FULL_PAGE_EASYOCR_THRESHOLD = 80.0
_MIN_WORDS_FOR_TARGETED_EASYOCR = 3
_HIGH_PADDLE_PAGE_CONFIDENCE = 94.0
_LOW_PADDLE_WORD_CONFIDENCE = 85.0


class OcrEngineUnavailableError(RuntimeError):
    pass


def _configured_languages() -> tuple[str, ...]:
    aliases = {
        "eng": "en",
        "en": "en",
        "hin": "hi",
        "hi": "hi",
        "mar": "mr",
        "mr": "mr",
    }
    configured = [
        language.strip().casefold()
        for language in re.split(r"[+,]", get_settings().ocr_languages)
        if language.strip()
    ]
    unsupported = sorted(set(configured) - aliases.keys())
    if unsupported:
        raise ValueError(f"Unsupported OCR language codes: {', '.join(unsupported)}")
    languages = tuple(dict.fromkeys(aliases[language] for language in configured))
    if not languages:
        raise ValueError("At least one OCR language must be configured.")
    return languages


def get_ocr_status() -> dict[str, object]:
    paddle_available = importlib.util.find_spec("paddle") is not None and importlib.util.find_spec("paddleocr") is not None
    easyocr_available = (
        importlib.util.find_spec("easyocr") is not None
        and importlib.util.find_spec("torch") is not None
    )
    engines = []
    if paddle_available:
        engines.append("paddleocr")
    if easyocr_available:
        engines.append("easyocr")
    errors = []
    try:
        languages = list(_configured_languages())
    except ValueError as exc:
        languages = []
        errors.append(str(exc))
    if not paddle_available:
        errors.append("PaddleOCR/PaddlePaddle is not installed.")
    if not easyocr_available:
        errors.append("EasyOCR/PyTorch is not installed.")
    return {
        "configured": paddle_available and easyocr_available and not errors,
        "engines": engines,
        "languages": languages,
        "error": " ".join(errors) or None,
    }


@lru_cache(maxsize=3)
def _paddle_reader(language: str):
    os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", str(min(4, os.cpu_count() or 1)))
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name=(
            "en_PP-OCRv5_mobile_rec" if language == "en" else "devanagari_PP-OCRv5_mobile_rec"
        ),
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


@lru_cache(maxsize=1)
def _easy_reader(languages: tuple[str, ...]):
    import torch

    cpu_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    import easyocr

    return easyocr.Reader(list(languages), gpu=False, verbose=False)


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _box_edges(box) -> tuple[float, float, float, float] | None:
    box = _plain(box)
    if not isinstance(box, (list, tuple)) or not box:
        return None
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        x1, y1, x2, y2 = (float(value) for value in box)
    else:
        points = [_plain(point) for point in box]
        if any(not isinstance(point, (list, tuple)) or len(point) < 2 for point in points):
            return None
        x_values = [float(point[0]) for point in points]
        y_values = [float(point[1]) for point in points]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expand_text_box(
    text: str,
    box,
    confidence: float,
    engine: str,
    page_number: int,
    image_index: int,
    line_number: int,
) -> list[dict]:
    edges = _box_edges(box)
    tokens = re.findall(r"\S+", text.strip())
    if edges is None or not tokens:
        return []
    x1, y1, x2, y2 = edges
    total_length = sum(len(token) for token in tokens)
    cursor = x1
    words = []
    for token in tokens:
        width = (x2 - x1) * len(token) / total_length
        words.append(
            {
                "page": page_number,
                "image": image_index,
                "block": 0,
                "par": 0,
                "line": line_number,
                "x": round(cursor),
                "y": round(y1),
                "width": max(1, round(width)),
                "height": max(1, round(y2 - y1)),
                "text": token,
                "confidence": max(0.0, min(100.0, confidence)),
                "engine": engine,
            }
        )
        cursor += width
    return words


def _paddle_words(results, page_number: int, image_index: int) -> list[dict]:
    words = []
    for result in results:
        payload = result.json if hasattr(result, "json") else result
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            continue
        payload = payload.get("res", payload)
        if not isinstance(payload, dict):
            continue
        texts = _plain(payload.get("rec_texts")) or []
        scores = _plain(payload.get("rec_scores")) or []
        boxes = payload.get("rec_boxes")
        if boxes is None:
            boxes = payload.get("rec_polys")
        boxes = _plain(boxes) or []
        for index, text in enumerate(texts):
            if index >= len(boxes) or not str(text).strip():
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            words.extend(
                _expand_text_box(
                    str(text),
                    boxes[index],
                    score * 100,
                    "paddleocr",
                    page_number,
                    image_index,
                    index,
                )
            )
    return words


def _easyocr_words(results, page_number: int, image_index: int) -> list[dict]:
    words = []
    for line_number, result in enumerate(results):
        if len(result) < 3:
            continue
        box, text, score = result
        words.extend(
            _expand_text_box(
                str(text),
                box,
                float(score) * 100,
                "easyocr",
                page_number,
                image_index,
                line_number,
            )
        )
    return words


def _intersection_over_union(left: dict, right: dict) -> float:
    left_x2 = left["x"] + left["width"]
    left_y2 = left["y"] + left["height"]
    right_x2 = right["x"] + right["width"]
    right_y2 = right["y"] + right["height"]
    intersection = max(0, min(left_x2, right_x2) - max(left["x"], right["x"])) * max(
        0, min(left_y2, right_y2) - max(left["y"], right["y"])
    )
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union else 0.0


def _normalized_token(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _prefer_candidate(candidate: dict, other: dict) -> bool:
    candidate_confidence = candidate["confidence"]
    other_confidence = other["confidence"]
    return candidate_confidence >= _HIGH_CONFIDENCE and (
        candidate_confidence - other_confidence >= _MIN_CONFIDENCE_MARGIN
        or other_confidence <= _LOW_CONFIDENCE
    )


def _alternative(word: dict) -> dict:
    return {
        "text": word["text"],
        "confidence": word["confidence"],
        "engine": word["engine"],
    }


def _easyocr_mode(paddle_words: list[dict]) -> str:
    if len(paddle_words) < _MIN_WORDS_FOR_TARGETED_EASYOCR:
        return "full"
    scores = [word["confidence"] for word in paddle_words]
    average = mean(scores)
    if average < _FULL_PAGE_EASYOCR_THRESHOLD:
        return "full"
    if average < _HIGH_PADDLE_PAGE_CONFIDENCE or min(scores) < _LOW_PADDLE_WORD_CONFIDENCE:
        return "targeted"
    return "skip"


def _uncertain_line_regions(image_path: Path, paddle_words: list[dict]) -> list[list[int]]:
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for word in paddle_words:
        if word["confidence"] < _HIGH_PADDLE_PAGE_CONFIDENCE:
            key = (word["image"], word["block"], word["line"])
            grouped.setdefault(key, []).append(word)
    if not grouped:
        return []

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    regions = []
    for words in grouped.values():
        left = min(word["x"] for word in words)
        top = min(word["y"] for word in words)
        right = max(word["x"] + word["width"] for word in words)
        bottom = max(word["y"] + word["height"] for word in words)
        horizontal_padding = max(3, round((right - left) * 0.06))
        vertical_padding = max(2, round((bottom - top) * 0.15))
        regions.append(
            [
                max(0, left - horizontal_padding),
                min(image_width, right + horizontal_padding),
                max(0, top - vertical_padding),
                min(image_height, bottom + vertical_padding),
            ]
        )
    return regions


def _combine_words(paddle_words: list[dict], easy_words: list[dict]) -> list[dict]:
    combined = [dict(word) for word in paddle_words]
    matched_paddle_indices = set()
    for easy_word in easy_words:
        matches = [
            (index, _intersection_over_union(easy_word, paddle_word))
            for index, paddle_word in enumerate(paddle_words)
        ]
        match = max(matches, key=lambda item: item[1], default=(-1, 0.0))
        if match[1] >= 0.25:
            index = match[0]
            if index in matched_paddle_indices:
                continue
            matched_paddle_indices.add(index)
            paddle_word = combined[index]
            if _normalized_token(paddle_word["text"]) == _normalized_token(easy_word["text"]):
                paddle_word["agreement"] = True
                paddle_word["engine_scores"] = [
                    {"engine": paddle_word["engine"], "confidence": paddle_word["confidence"]},
                    {"engine": easy_word["engine"], "confidence": easy_word["confidence"]},
                ]
            else:
                paddle_word["agreement"] = False
                if _prefer_candidate(easy_word, paddle_word):
                    selected_word = dict(easy_word)
                    selected_word["agreement"] = False
                    selected_word["alternatives"] = [_alternative(paddle_word)]
                    selected_word["selection_reason"] = "higher_engine_confidence"
                    combined[index] = selected_word
                else:
                    paddle_word.setdefault("alternatives", []).append(_alternative(easy_word))
                    paddle_word["selection_reason"] = "confidence_not_decisive"
        elif easy_word["confidence"] >= _MIN_UNMATCHED_CONFIDENCE:
            combined.append(dict(easy_word))
    return sorted(combined, key=lambda word: (word["y"], word["x"]))


def _words_to_text(words: list[dict]) -> str:
    if not words:
        return ""
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda item: (item["y"] + item["height"] / 2, item["x"])):
        center = word["y"] + word["height"] / 2
        line = next(
            (
                line
                for line in lines
                if abs(center - mean(item["y"] + item["height"] / 2 for item in line))
                <= max(8, word["height"] * 0.6)
            ),
            None,
        )
        if line is None:
            lines.append([word])
        else:
            line.append(word)
    lines.sort(key=lambda line: min(word["y"] for word in line))
    return "\n".join(
        " ".join(word["text"] for word in sorted(line, key=lambda item: item["x"]))
        for line in lines
    )


def detect_table_lines(image_source: Path | bytes) -> list[dict[str, int | float | str]]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV is unavailable for table-line detection.") from exc

    try:
        if isinstance(image_source, bytes):
            image = cv2.imdecode(np.frombuffer(image_source, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError("Could not read page image for table-rule detection.")
        height, width = image.shape
        scale = min(1.0, 1800 / max(width, height))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = image.shape
        block_size = min(31, min(width, height) if min(width, height) % 2 else min(width, height) - 1)
        if block_size < 3 or min(width, height) < 80:
            return []

        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            12,
        )
        candidates: dict[str, list[tuple[float, int, int]]] = {"horizontal": [], "vertical": []}
        for orientation in ("horizontal", "vertical"):
            is_horizontal = orientation == "horizontal"
            dimension = width if is_horizontal else height
            kernel_size = max(20, dimension // 30)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, 1) if is_horizontal else (1, kernel_size),
            )
            rule_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            lines = cv2.HoughLinesP(
                rule_mask,
                1,
                math.pi / 180,
                threshold=max(20, dimension // 30),
                minLineLength=max(50, dimension // 12),
                maxLineGap=max(8, dimension // 50),
            )
            if lines is None:
                continue
            for x1, y1, x2, y2 in lines[:, 0, :]:
                dx = int(x2) - int(x1)
                dy = int(y2) - int(y1)
                length = (dx * dx + dy * dy) ** 0.5
                if is_horizontal:
                    if length < width * 0.12 or abs(dy) > max(12, length * 0.04):
                        continue
                    position = (int(y1) + int(y2)) / 2
                    start, end = sorted((int(x1), int(x2)))
                else:
                    if length < height * 0.18 or abs(dx) > max(10, length * 0.04):
                        continue
                    position = (int(x1) + int(x2)) / 2
                    start, end = sorted((int(y1), int(y2)))
                candidates[orientation].append((position, start, end))

        detected: list[dict[str, int | float | str]] = []
        for orientation, segments in candidates.items():
            is_horizontal = orientation == "horizontal"
            dimension = width if is_horizontal else height
            position_tolerance = max(7, dimension * 0.008)
            interval_gap = max(12, dimension * 0.035)
            minimum_coverage = dimension * (0.42 if is_horizontal else 0.20)
            groups: list[list[tuple[float, int, int]]] = []
            for segment in sorted(segments):
                if groups and segment[0] - groups[-1][-1][0] <= position_tolerance:
                    groups[-1].append(segment)
                else:
                    groups.append([segment])

            for group in groups:
                position = sum(segment[0] for segment in group) / len(group)
                intervals: list[list[int]] = []
                for _, start, end in sorted(group, key=lambda segment: segment[1]):
                    if intervals and start <= intervals[-1][1] + interval_gap:
                        intervals[-1][1] = max(intervals[-1][1], end)
                    else:
                        intervals.append([start, end])
                for start, end in intervals:
                    if end - start < minimum_coverage:
                        continue
                    if is_horizontal:
                        if position < height * 0.015 or position > height * 0.985:
                            continue
                        line = {
                            "orientation": orientation,
                            "x1": round(start / scale),
                            "y1": round(position / scale),
                            "x2": round(end / scale),
                            "y2": round(position / scale),
                        }
                    else:
                        if position < width * 0.015 or position > width * 0.985:
                            continue
                        line = {
                            "orientation": orientation,
                            "x1": round(position / scale),
                            "y1": round(start / scale),
                            "x2": round(position / scale),
                            "y2": round(end / scale),
                        }
                    detected.append(line)
        return detected
    except cv2.error as exc:
        raise RuntimeError(f"Could not analyze page image for table-rule detection: {image_path}") from exc


def run_ocr(
    image_path: Path,
    page_number: int,
    image_index: int,
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[str, list[dict], float | None, str, list[str]]:
    languages = _configured_languages()
    paddle_language = "en" if all(language == "en" for language in languages) else "devanagari"
    if progress_callback:
        progress_callback("loading_paddleocr", 0.05)
    paddle_started = perf_counter()
    try:
        paddle_results = _paddle_reader(paddle_language).predict(str(image_path))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise OcrEngineUnavailableError(f"PaddleOCR failed: {str(exc)[:200]}") from exc
    paddle_words = _paddle_words(paddle_results, page_number, image_index)
    logger.info(
        "PaddleOCR completed for page %s image %s in %.2f seconds (%s words)",
        page_number,
        image_index,
        perf_counter() - paddle_started,
        len(paddle_words),
    )
    errors = []
    easy_words = []
    easyocr_succeeded = False
    verification_mode = _easyocr_mode(paddle_words)
    if verification_mode != "skip":
        if progress_callback:
            progress_callback("loading_easyocr", 0.5)
        easy_started = perf_counter()
        try:
            reader = _easy_reader(languages)
            if progress_callback:
                progress_callback(
                    "easyocr_targeted" if verification_mode == "targeted" else "easyocr",
                    0.6,
                )
            if verification_mode == "targeted":
                regions = _uncertain_line_regions(image_path, paddle_words)
                easy_results = reader.recognize(
                    str(image_path),
                    horizontal_list=regions,
                    free_list=[],
                    detail=1,
                    paragraph=False,
                )
            else:
                easy_results = reader.readtext(str(image_path), detail=1, paragraph=False)
            easy_words = _easyocr_words(easy_results, page_number, image_index)
            easyocr_succeeded = True
            logger.info(
                "EasyOCR %s verification completed for page %s image %s in %.2f seconds (%s words)",
                verification_mode,
                page_number,
                image_index,
                perf_counter() - easy_started,
                len(easy_words),
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("EasyOCR failed for page %s image %s: %s", page_number, image_index, exc)
            errors.append(f"EasyOCR failed: {str(exc)[:200]}")
    elif progress_callback:
        progress_callback("ocr_complete", 0.9)
    words = _combine_words(paddle_words, easy_words)
    text = _words_to_text(words)
    confidence = mean(word["confidence"] for word in words) if words else None
    engine = "paddleocr+easyocr" if easyocr_succeeded else "paddleocr"
    if progress_callback and easyocr_succeeded:
        progress_callback("ocr_complete", 0.9)
    return text, words, confidence, engine, errors
