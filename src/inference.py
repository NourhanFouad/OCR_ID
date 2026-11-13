"""Conservative PaddleOCR inference adapter; never invent semantic fields."""

from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.extraction import extract_fields
from src.extraction import split_mixed_date_id_candidate
from src.preprocessing import crop_egyptian_back_zones, crop_egyptian_front_zones, crop_occupation_retry, field_preprocessing_candidates, load_rgb
from src.validation import derive_birth_date_from_national_id, digits_for_validation, format_date_candidate, inspect_date_format, reconcile_birth_date_candidates, validate_national_id_format


def _prediction_payload(item: Any) -> dict[str, Any]:
    value = getattr(item, "json", item)
    if callable(value):
        value = value()
    if isinstance(value, dict) and "res" in value and isinstance(value["res"], dict):
        value = value["res"]
    if not isinstance(value, dict):
        raise TypeError(f"Unexpected PaddleOCR result type: {type(value).__name__}")
    return value


def create_ocr_engine(language: str = "ar", model_dir: str | Path | None = None, device: str = "cpu", detector_size: str = "mobile", textline_orientation: bool = False):
    """Create one reusable PaddleOCR engine. Set PADDLE_PDX_CACHE_HOME before import/use."""
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(project_root / "models" / "pretrained" / "paddlex_cache"))
    if device == "cpu":
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            f"PaddleOCR import failed ({type(exc).__name__}): {exc}. "
            "Check the installed PaddleOCR dependencies for this Python environment."
        ) from exc

    kwargs: dict[str, Any] = {
        "lang": language,
        "ocr_version": "PP-OCRv5",
        "device": device,
        "text_detection_model_name": f"PP-OCRv5_{detector_size}_det",
        "text_recognition_model_name": "arabic_PP-OCRv5_mobile_rec" if language == "ar" else "latin_PP-OCRv5_mobile_rec",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": textline_orientation,
    }
    if model_dir is not None:
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Configured recognition model directory does not exist: {path}")
        kwargs["text_recognition_model_dir"] = str(path)
    return PaddleOCR(**kwargs)


def recognize_lines(image_path: str | Path | Any, engine: Any, bbox_offset: tuple[int, int] = (0, 0)) -> list[dict[str, Any]]:
    """Return unmodified OCR strings, scores, and detected boxes for one image."""
    model_input = str(image_path) if isinstance(image_path, (str, Path)) else np.asarray(image_path.convert("RGB") if isinstance(image_path, Image.Image) else image_path)
    predictions = list(engine.predict(model_input))
    lines: list[dict[str, Any]] = []
    for item in predictions:
        payload = _prediction_payload(item)
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        boxes = payload.get("rec_boxes")
        if boxes is None:
            boxes = payload.get("rec_polys")
        if boxes is None:
            boxes = []
        for index, text in enumerate(texts):
            score = float(scores[index]) if index < len(scores) and scores[index] is not None else None
            box = boxes[index] if index < len(boxes) else None
            if hasattr(box, "tolist"):
                box = box.tolist()
            if box is not None and bbox_offset != (0, 0):
                dx, dy = bbox_offset
                if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
                    box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
                else:
                    box = [[point[0] + dx, point[1] + dy] for point in box]
            lines.append({"raw_text": str(text), "confidence": score, "bbox": box})
    return lines


def recognize_lines_batch(images: list[Any], engine: Any, bbox_offsets: list[tuple[int, int]] | None = None, bbox_scales: list[float] | None = None) -> list[list[dict[str, Any]]]:
    """Process crops sequentially to limit peak memory on small cloud instances."""
    if bbox_offsets is None:
        bbox_offsets = [(0, 0)] * len(images)
    if len(images) != len(bbox_offsets):
        raise ValueError("Each input image needs one bounding-box offset")
    if bbox_scales is None:
        bbox_scales = [1.0] * len(images)
    if len(images) != len(bbox_scales):
        raise ValueError("Each input image needs one bounding-box scale")
    results = []
    for image, offset, scale in zip(images, bbox_offsets, bbox_scales):
        model_image = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        predictions = list(engine.predict(model_image))
        if len(predictions) != 1:
            raise RuntimeError(f"PaddleOCR returned {len(predictions)} results for one crop")
        results.append(recognize_lines_from_payload(_prediction_payload(predictions[0]), offset, scale))
    return results


def recognize_lines_from_payload(payload: dict[str, Any], bbox_offset: tuple[int, int] = (0, 0), bbox_scale: float = 1.0) -> list[dict[str, Any]]:
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    boxes = payload.get("rec_boxes")
    if boxes is None:
        boxes = payload.get("rec_polys")
    if boxes is None:
        boxes = []
    lines: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else None
        box = boxes[index] if index < len(boxes) else None
        if hasattr(box, "tolist"):
            box = box.tolist()
        if box is not None and (bbox_offset != (0, 0) or bbox_scale != 1.0):
            dx, dy = bbox_offset
            if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
                box = [box[0] / bbox_scale + dx, box[1] / bbox_scale + dy, box[2] / bbox_scale + dx, box[3] / bbox_scale + dy]
            else:
                box = [[point[0] / bbox_scale + dx, point[1] / bbox_scale + dy] for point in box]
        lines.append({"raw_text": str(text), "confidence": score, "bbox": box})
    return lines


def _join_region_lines(lines: list[dict[str, Any]]) -> str | None:
    """Group detections into visual rows and read Arabic rows from right to left."""
    entries = []
    for line in lines:
        text = (line.get("raw_text") or "").strip()
        box = line.get("bbox")
        if not text or not box:
            continue
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            x1, y1, x2, y2 = map(float, box)
        else:
            points = [(float(point[0]), float(point[1])) for point in box]
            if not points:
                continue
            x1, y1 = min(p[0] for p in points), min(p[1] for p in points)
            x2, y2 = max(p[0] for p in points), max(p[1] for p in points)
        entries.append({"text": text, "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "height": y2 - y1})
    if not entries:
        return None
    entries.sort(key=lambda item: item["y"])
    median_height = sorted(max(entry["height"], 1) for entry in entries)[len(entries) // 2]
    row_gap = max(16.0, median_height * 0.48)
    rows: list[list[dict[str, Any]]] = []
    for entry in entries:
        if not rows or abs(entry["y"] - sum(item["y"] for item in rows[-1]) / len(rows[-1])) > row_gap:
            rows.append([entry])
        else:
            rows[-1].append(entry)
    return " ".join(" ".join(item["text"] for item in sorted(row, key=lambda item: item["x"], reverse=True)) for row in rows)


def _line_center_y(line: dict[str, Any]) -> float | None:
    box = line.get("bbox")
    if not box:
        return None
    try:
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            return (float(box[1]) + float(box[3])) / 2
        points = [(float(point[0]), float(point[1])) for point in box]
        return sum(point[1] for point in points) / len(points) if points else None
    except (TypeError, ValueError, IndexError):
        return None


def _line_rect(line: dict[str, Any]) -> list[float] | None:
    box = line.get("bbox")
    if not box:
        return None
    try:
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            x1, y1, x2, y2 = map(float, box)
        else:
            points = [(float(point[0]), float(point[1])) for point in box]
            x1, y1 = min(p[0] for p in points), min(p[1] for p in points)
            x2, y2 = max(p[0] for p in points), max(p[1] for p in points)
        return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None
    except (TypeError, ValueError, IndexError):
        return None


def _front_fields_from_detected_lines(lines: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, dict[str, Any]]:
    """Group full-image detections by the common front-card layout, not OCR order."""
    width, height = image_size
    regions = {
        "name": (0.50, 0.22, 1.00, 0.44),
        "address": (0.48, 0.44, 1.00, 0.66),
        "birth_date": (0.00, 0.63, 0.42, 0.83),
        "national_id": (0.42, 0.68, 1.00, 0.86),
        "serial_number": (0.00, 0.83, 0.42, 0.99),
    }
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in regions}
    for line in lines:
        rect = _line_rect(line)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        matches = [name for name, (left, top, right, bottom) in regions.items() if left <= cx <= right and top <= cy <= bottom]
        if len(matches) == 1:
            assigned[matches[0]].append(line)

    fields: dict[str, dict[str, Any]] = {}
    for name, candidates in assigned.items():
        if not candidates:
            continue
        raw = _join_region_lines(candidates)
        if not raw:
            continue
        rects = [_line_rect(line) for line in candidates]
        rects = [rect for rect in rects if rect is not None]
        bbox = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)]
        confidence = min((line["confidence"] for line in candidates if line.get("confidence") is not None), default=None)
        if name in {"birth_date", "issue_date", "expiry_date"}:
            formatted = format_date_candidate(raw, allow_year_month=name == "issue_date")
            value, status = _safe_region_value(name, raw, formatted)
            date_status = inspect_date_format(raw, allow_year_month=name == "issue_date")
        else:
            formatted, date_status = None, None
            value, status = _safe_region_value(name, raw, None)
        fields[name] = {
            "raw_ocr_output": raw,
            "field_value_candidate": value,
            "field_candidate_status": status,
            "formatted_display_candidate": formatted,
            "date_format_status": date_status,
            "confidence": confidence,
            "bbox": [round(v) for v in bbox],
            "raw_lines": candidates,
            "review_status": "needs_human_review",
            "candidate_provenance": "full_image_text_boxes_position_grouping",
            "source": "full-image detected boxes grouped by normalized front layout; candidate only",
        }
        if name == "national_id":
            fields[name]["validation_result"] = validate_national_id_format(value or raw)
    return fields


def _safe_region_value(name: str, raw: str | None, formatted_date: str | None) -> tuple[str | None, str]:
    """Do not display a semantic value unless its script/format is plausible."""
    if not raw:
        return None, "missing_or_ambiguous"
    if name == "national_id":
        digits = "".join(char for char in raw if char.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")).isdigit())
        check = validate_national_id_format(digits)
        valid = bool(check["checks"] and check["checks"][0]["passed"])
        return (digits, "needs_human_review") if valid else (None, "incomplete_or_invalid_length")
    if name in {"birth_date", "issue_date", "expiry_date"}:
        return (formatted_date, "needs_human_review") if formatted_date else (None, "unformatted_date_needs_review")
    known_values = {
        "marital_status": {"أعزب", "عزباء", "آنسة", "أنسة", "متزوج", "متزوجة", "أرمل", "أرملة", "مطلق", "مطلقة"},
    }
    if name in {"gender", "religion"}:
        normalized = _normalize_arabic_label(raw)
        choices = {
            "gender": {
                "ذكر": ("ذكر", "الذكر", "زكر"),
                "أنثى": ("أنثى", "انثى", "الأنثى", "انثي"),
            },
            "religion": {
                "مسلم": ("مسلم", "مسلمة", "الإسلام", "اسلام", "إسلامي"),
                "مسيحي": ("مسيحي", "مسيحية", "مسيحى"),
                "يهودي": ("يهودي", "يهودية", "يهودى"),
            },
        }[name]
        scores = sorted(
            ((max(SequenceMatcher(None, normalized, _normalize_arabic_label(alias)).ratio() for alias in aliases), canonical)
             for canonical, aliases in choices.items()),
            reverse=True,
        )
        if scores[0][0] >= 0.68 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.12):
            canonical = scores[0][1]
            status = "needs_human_review" if raw.strip() == canonical else "normalized_ocr_needs_human_review"
            return canonical, status
        return None, "text_did_not_match_known_value"
    if name in known_values:
        return (raw, "needs_human_review") if raw.strip() in known_values[name] else (None, "text_did_not_match_known_value")
    if name == "occupation":
        if digits_for_validation(raw):
            return None, "numeric_contamination_needs_review"
        return (raw, "needs_human_review") if len(raw.strip()) <= 45 else (None, "text_too_long_or_misaligned")
    return raw, "needs_human_review"


def _normalize_arabic_label(value: str) -> str:
    value = "".join(char for char in unicodedata.normalize("NFKD", value) if unicodedata.category(char) != "Mn")
    value = value.replace("ـ", "")
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    value = "".join(char for char in value if "\u0621" <= char <= "\u064a")
    if value.startswith("ال"):
        value = value[2:]
    return value


def _known_front_header_suggestion(lines: list[dict[str, Any]], side: str, image_size: tuple[int, int]) -> dict[str, Any] | None:
    """Offer a review-only phrase candidate for the common Egyptian ID header."""
    if side != "front" or not image_size[1]:
        return None
    top_lines = []
    for line in lines:
        text = line.get("raw_text") or ""
        box = line.get("bbox")
        if not text or not box:
            continue
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            center_y = (float(box[1]) + float(box[3])) / 2
        else:
            points = [(float(point[0]), float(point[1])) for point in box]
            center_y = sum(point[1] for point in points) / len(points) if points else image_size[1]
        if center_y <= image_size[1] * 0.27:
            top_lines.append(text)
    if not any("بطاقة" in text for text in top_lines):
        return None
    if any("جمهوريةمصرالعربية" in text.replace(" ", "") for text in top_lines):
        return None
    return {
        "suggested_text": "جمهورية مصر العربية",
        "source": "common Egyptian ID header prior requested by the user",
        "status": "suggestion_needs_visual_confirmation",
        "raw_top_lines": top_lines,
        "note": "Suggestion only. Original OCR output is preserved and is not replaced.",
    }


def run_ocr(image_path: str | Path, language: str = "ar", side: str = "unknown", side_source: str | None = None, template_mapping_confirmed: bool = False, model_dir: str | Path | None = None, device: str = "cpu", front_region_profile: bool = False, back_region_profile: bool = False, engine: Any | None = None, region_engine: Any | None = None) -> dict[str, Any]:
    """Run one language candidate and return raw lines with boxes and scores.

    PaddleOCR downloads its configured pretrained weights on first use if they
    are not already cached. No field assignment or text correction happens here.
    """
    image = load_rgb(image_path)
    image_size = image.size
    engine = engine or create_ocr_engine(language, model_dir, device, textline_orientation=True)
    lines = recognize_lines(image, engine)

    template_fields = extract_fields(lines, side, image_size) if template_mapping_confirmed else None
    header_suggestion = _known_front_header_suggestion(lines, side, image_size)
    region_fields = None
    active_profile = "egyptian_front_approx_v1" if front_region_profile and side == "front" else "egyptian_back_approx_v1" if back_region_profile and side == "back" else None
    if active_profile:
        region_engine = region_engine or create_ocr_engine(language, model_dir, device, textline_orientation=False)
        zones = crop_egyptian_front_zones(image, scale=2.0) if active_profile.startswith("egyptian_front") else crop_egyptian_back_zones(image, scale=3.0)
        names = list(zones)
        region_lines = recognize_lines_batch(
            [zones[name]["image"] for name in names], region_engine,
            bbox_offsets=[tuple(zones[name]["bbox_xyxy"][:2]) for name in names],
            bbox_scales=[zones[name]["scale"] for name in names],
        )
        context_lines = recognize_lines(image, region_engine) if active_profile.startswith("egyptian_back") else []
        region_fields = {}
        for name, candidates in zip(names, region_lines):
            raw_candidate = _join_region_lines(candidates)
            numeric_field = name in {"national_id", "birth_date", "issue_date", "expiry_date"}
            ordered_candidate = raw_candidate
            confidence = min((line["confidence"] for line in candidates if line.get("confidence") is not None), default=None)
            retry_candidates: list[dict[str, Any]] = []
            occupation_contaminated = name == "occupation" and bool(digits_for_validation(ordered_candidate or ""))
            issue_date_needs_retry = name == "issue_date" and format_date_candidate(ordered_candidate, allow_year_month=True) is None
            if occupation_contaminated:
                retry_zone = crop_occupation_retry(image)
                retry_lines = recognize_lines(retry_zone["image"], region_engine)
                retry_text = _join_region_lines(retry_lines)
                if retry_text:
                    retry_candidates.append({
                        "preprocessing": "occupation_tight_crop",
                        "raw_text": retry_text,
                        "confidence": min((line["confidence"] for line in retry_lines if line.get("confidence") is not None), default=None),
                        "bbox": retry_zone["bbox_xyxy"],
                    })
            elif name in {"national_id", "birth_date", "issue_date", "expiry_date"} and (confidence is None or confidence < 0.82 or not ordered_candidate or issue_date_needs_retry):
                try:
                    variants = field_preprocessing_candidates(zones[name]["image"])
                    for variant_name, variant_image in variants.items():
                        variant_lines = recognize_lines(variant_image, region_engine)
                        variant_raw = _join_region_lines(variant_lines)
                        if variant_raw:
                            retry_candidates.append({
                                "preprocessing": variant_name,
                                "raw_text": variant_raw,
                                "confidence": min((line["confidence"] for line in variant_lines if line.get("confidence") is not None), default=None),
                            })
                except RuntimeError:
                    retry_candidates = []
            context_line = None
            alternate_context_line = None
            context_date_sources: list[str] = []
            date_sources = [ordered_candidate] if ordered_candidate else []
            date_sources.extend(item["raw_text"] for item in retry_candidates)
            if name == "issue_date":
                date_lines = [
                    line.get("raw_text") for line in lines
                    if "/" in (line.get("raw_text") or "")
                    and any(char.isdigit() for char in line.get("raw_text", ""))
                    and (_line_center_y(line) is not None and _line_center_y(line) <= image_size[1] * 0.20)
                ]
                context_line = date_lines[0] if date_lines else None
                context_date_sources = date_lines
                date_sources = context_date_sources + date_sources
            if name == "expiry_date":
                primary_expiry = next((line.get("raw_text") for line in lines if any(token in (line.get("raw_text") or "") for token in ("سارية", "حتى"))), None)
                alternate_context_line = next((line.get("raw_text") for line in context_lines if any(token in (line.get("raw_text") or "") for token in ("سارية", "حتى"))), None)
                context_line = primary_expiry
                for candidate_line in (primary_expiry, alternate_context_line):
                    if candidate_line:
                        date_match = re.search(r"[٠-٩0-9]{6,12}", candidate_line)
                        if date_match:
                            date_sources.append(date_match.group(0))
                        else:
                            date_sources.append(candidate_line)
            formatted_date = next((value for source in date_sources if (value := format_date_candidate(source, allow_year_month=name == "issue_date")) is not None), None) if name in {"birth_date", "issue_date", "expiry_date"} else None
            date_status = inspect_date_format(next((source for source in date_sources if format_date_candidate(source, allow_year_month=name == "issue_date")), date_sources[0] if date_sources else None), allow_year_month=name == "issue_date") if name in {"birth_date", "issue_date", "expiry_date"} else None
            display_source = ordered_candidate
            candidate_conflict = False
            selected_retry: str | None = None
            if name == "national_id" and retry_candidates:
                valid_ids = []
                sources = ([{"raw_text": ordered_candidate, "preprocessing": "original"}] if ordered_candidate else []) + retry_candidates
                for item in sources:
                    digits = "".join(char for char in item["raw_text"] if char.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")).isdigit())
                    if len(digits) == 14 and all(check["passed"] is not False for check in validate_national_id_format(digits)["checks"]):
                        valid_ids.append(digits)
                unique_ids = set(valid_ids)
                if len(unique_ids) == 1:
                    display_source = next(iter(unique_ids))
                elif len(unique_ids) > 1:
                    display_source, candidate_conflict = None, True
            elif name in {"birth_date", "issue_date", "expiry_date"} and retry_candidates:
                date_attempts = context_date_sources + ([ordered_candidate] if ordered_candidate else []) + [item["raw_text"] for item in retry_candidates]
                unique_dates = {candidate for source in date_attempts if (candidate := format_date_candidate(source, allow_year_month=name == "issue_date"))}
                if len(unique_dates) == 1:
                    formatted_date = next(iter(unique_dates))
                    display_source = next((source for source in date_attempts if format_date_candidate(source, allow_year_month=name == "issue_date") == formatted_date), display_source)
                elif len(unique_dates) > 1:
                    formatted_date, display_source, candidate_conflict = None, None, True
            elif name == "occupation" and retry_candidates:
                clean_candidates = [item for item in retry_candidates if not digits_for_validation(item["raw_text"]) and len(item["raw_text"].strip()) <= 45]
                unique_texts = {item["raw_text"].strip() for item in clean_candidates}
                if len(unique_texts) == 1:
                    selected_retry = next(item["preprocessing"] for item in clean_candidates if item["raw_text"].strip() in unique_texts)
                    display_source = next(iter(unique_texts))
                elif len(unique_texts) > 1:
                    display_source, candidate_conflict = None, True
            display_value, display_status = _safe_region_value(name, display_source, formatted_date)
            if candidate_conflict:
                display_status = "ambiguous_preprocessing_candidates"
            field = {
                "raw_ocr_output": raw_candidate,
                "reading_order_candidate": ordered_candidate if numeric_field else None,
                "field_value_candidate": display_value,
                "field_candidate_status": display_status,
                "formatted_display_candidate": formatted_date,
                "date_format_status": date_status,
                "whole_image_context_line": context_line,
                "alternate_full_image_line": alternate_context_line,
                "raw_lines": candidates,
                "bbox": zones[name]["bbox_xyxy"],
                "confidence": confidence,
                "preprocessing_retry_candidates": retry_candidates,
                "candidate_provenance": selected_retry or "original_crop",
                "review_status": "needs_human_review",
                "source": f"optional {active_profile} normalized image-region profile; candidate only",
            }
            if name == "national_id":
                field["validation_result"] = validate_national_id_format(display_source or ordered_candidate or "")
            region_fields[name] = field
        if active_profile.startswith("egyptian_front"):
            detected_fields = _front_fields_from_detected_lines(lines, image_size)
            for name, detected in detected_fields.items():
                old = region_fields.get(name)
                if old:
                    full_image_candidate = {
                        "raw_ocr_output": detected.get("raw_ocr_output"),
                        "field_value_candidate": detected.get("field_value_candidate"),
                        "confidence": detected.get("confidence"),
                        "bbox": detected.get("bbox"),
                        "source": detected.get("source"),
                    }
                    detected["alternate_region_candidate"] = {
                        "raw_ocr_output": old.get("raw_ocr_output"),
                        "field_value_candidate": old.get("field_value_candidate"),
                        "confidence": old.get("confidence"),
                        "bbox": old.get("bbox"),
                        "source": old.get("source"),
                    }
                    if (old.get("field_value_candidate") and isinstance(old.get("confidence"), (int, float))
                            and isinstance(detected.get("confidence"), (int, float))
                            and old["confidence"] > detected["confidence"]):
                        detected["alternate_full_image_candidate"] = full_image_candidate
                        for key in ("raw_ocr_output", "field_value_candidate", "field_candidate_status",
                                    "formatted_display_candidate", "date_format_status", "confidence", "bbox",
                                    "raw_lines", "candidate_provenance", "source", "validation_result"):
                            if key in old:
                                detected[key] = old[key]
                region_fields[name] = detected
    mixed_numeric_candidates = [candidate for line in lines if (candidate := split_mixed_date_id_candidate(line.get("raw_text", ""), line.get("confidence"), line.get("bbox"))) is not None]
    if active_profile and active_profile.startswith("egyptian_back") and region_fields and len(mixed_numeric_candidates) == 1:
        mixed = mixed_numeric_candidates[0]
        if mixed.get("national_id_candidate"):
            if "national_id" in region_fields:
                region_fields["national_id"]["field_value_candidate"] = mixed["national_id_candidate"]
                region_fields["national_id"]["field_candidate_status"] = "needs_human_review_mixed_line_split"
                region_fields["national_id"]["mixed_line_candidate"] = mixed
                region_fields["national_id"]["validation_result"] = validate_national_id_format(mixed["national_id_candidate"])
            if "issue_date" in region_fields:
                region_fields["issue_date"]["field_value_candidate"] = mixed["suggested_date_prefix"]
                region_fields["issue_date"]["formatted_display_candidate"] = mixed["suggested_date_prefix"]
                region_fields["issue_date"]["field_candidate_status"] = "needs_human_review_mixed_line_split"
                region_fields["issue_date"]["mixed_line_candidate"] = mixed
    else:
        mixed_numeric_candidates = mixed_numeric_candidates
    if active_profile and active_profile.startswith("egyptian_front") and region_fields:
        birth = region_fields.get("birth_date")
        national_id = region_fields.get("national_id")
        if birth is not None and not birth.get("field_value_candidate") and national_id is not None:
            derived = derive_birth_date_from_national_id(national_id.get("field_value_candidate"))
            birth["national_id_date_candidate"] = derived
            if derived.get("value"):
                birth["field_value_candidate"] = derived["value"]
                birth["field_candidate_status"] = "derived_from_national_id_needs_visual_review"
                birth["candidate_source"] = derived["source"]
                birth["candidate_warning"] = derived["reason"]
        elif birth is not None and national_id is not None and birth.get("formatted_display_candidate"):
            derived = derive_birth_date_from_national_id(national_id.get("field_value_candidate"))
            birth["national_id_date_candidate"] = derived
            if derived.get("value"):
                reconciliation = reconcile_birth_date_candidates(
                    birth["formatted_display_candidate"], derived["value"],
                    national_id.get("confidence"), birth.get("confidence"),
                )
                birth["date_reconciliation"] = reconciliation
                if reconciliation["conflict"]:
                    birth["printed_date_candidate"] = birth["formatted_display_candidate"]
                    if reconciliation["id_preferred"]:
                        id_confidence = national_id.get("confidence")
                        date_confidence = birth.get("confidence")
                        birth["field_value_candidate"] = derived["value"]
                        birth["field_candidate_status"] = "derived_from_national_id_higher_confidence_needs_visual_review"
                        birth["candidate_source"] = "national_id_encoded_date_preferred_by_higher_ocr_score"
                        birth["candidate_warning"] = (
                            f"تاريخ الصورة {birth['formatted_display_candidate']} يختلف عن التاريخ المشفّر في الرقم القومي {derived['value']}. "
                            f"عُرض التاريخ المستنتج لأن درجة الرقم القومي ({id_confidence:.3f}) أعلى من درجة التاريخ ({date_confidence:.3f})؛ راجعيهما بصريًا."
                        )
                    else:
                        birth["field_candidate_status"] = "conflicting_printed_and_id_dates_needs_visual_review"
                        birth["candidate_warning"] = (
                            f"تاريخ الصورة {birth['formatted_display_candidate']} يختلف عن التاريخ المشفّر في الرقم القومي {derived['value']}. "
                            "لم نستبدل تاريخ الصورة لأن درجة الرقم القومي ليست أعلى؛ راجعي الاثنين بصريًا."
                        )
    return {
        "engine": "PaddleOCR",
        "ocr_version": "PP-OCRv5",
        "language_candidate": language,
        "image_size": list(image_size),
        "side": side,
        "side_source": side_source or ("user_selected" if side in {"front", "back"} else "unknown"),
        "raw_lines": lines,
        "fields": template_fields,
        "region_candidates": region_fields,
        "region_profile": active_profile,
        "header_text_suggestion": header_suggestion,
        "mixed_numeric_candidates": mixed_numeric_candidates,
        "status": "baseline_prediction_unverified",
        "note": "All values are raw pretrained OCR predictions and require visual review. The optional Egyptian-front region profile is a layout heuristic; it never edits or accepts text automatically.",
    }
