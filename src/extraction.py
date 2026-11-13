"""Conservative source-template field extraction."""

from __future__ import annotations

from typing import Any
import re

from src.detection import assign_template_fields
from src.validation import DIGIT_TRANSLATION, format_date_candidate, validate_field, validate_national_id_format


def extract_fields(lines: list[dict[str, Any]], side: str, image_size: tuple[int, int]) -> dict[str, Any] | None:
    candidates = assign_template_fields(lines, side, image_size)
    if candidates is None:
        return None
    fields: dict[str, Any] = {}
    for name, candidate in candidates.items():
        raw = candidate["raw_value"]
        fields[name] = {
            "raw_ocr_output": raw,
            "processed_value": None,
            "validation_result": validate_field(name, raw),
            "confidence": candidate["confidence"],
            "bbox": candidate["bbox"],
            "review_status": "needs_human_review" if raw is not None else "missing_or_ambiguous",
            "candidate_count": candidate["candidate_count"],
            "source": candidate["source"],
        }
    return fields


def split_mixed_date_id_candidate(raw_text: str, confidence: float | None = None, bbox: Any = None) -> dict[str, Any] | None:
    """Suggest a date/ID split only when the digit count gives one exact split.

    Issue dates (YYYYMM) and full dates (YYYYMMDD) are allowed, but a national
    ID candidate is shown only when exactly 14 digits remain. Ambiguous lines
    are reported as unassigned; no digits are moved, dropped, or completed.
    """
    digits = "".join(char.translate(DIGIT_TRANSLATION) for char in raw_text if char.translate(DIGIT_TRANSLATION).isdigit())
    if len(digits) < 20:
        return None
    valid_splits = []
    for date_length in (8, 6):
        if len(digits) != date_length + 14:
            continue
        for date_at_start in (True, False):
            date_digits = digits[:date_length] if date_at_start else digits[-date_length:]
            id_digits = digits[date_length:] if date_at_start else digits[:-date_length]
            formatted = format_date_candidate(date_digits, allow_year_month=(date_length == 6))
            if not formatted:
                continue
            if any("٠" <= char <= "٩" for char in raw_text):
                formatted = formatted.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
            id_candidate = id_digits.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")) if any("٠" <= c <= "٩" for c in raw_text) else id_digits
            valid_splits.append({
            "raw_text": raw_text,
            "confidence": confidence,
            "bbox": bbox,
            "suggested_date_prefix": formatted,
            "date_position": "prefix" if date_at_start else "suffix",
            "id_digit_count": len(id_digits),
            "national_id_candidate": id_candidate,
            "national_id_status": "14_digit_candidate_needs_visual_review",
            "validation_result": validate_national_id_format(id_candidate),
            "review_status": "needs_human_review",
            "note": "Unique exact-length split candidate only. Check the original image; neither field is accepted automatically.",
            })
    if len(valid_splits) == 1:
        return valid_splits[0]
    if len(digits) in {20, 22}:
        return None
    return {
        "raw_text": raw_text,
        "confidence": confidence,
        "bbox": bbox,
        "suggested_date_prefix": None,
        "id_digit_count": None,
        "national_id_candidate": None,
        "national_id_status": "ambiguous_merged_numeric_line",
        "validation_result": None,
        "review_status": "needs_human_review",
        "note": f"One numeric OCR line contains {len(digits)} digits; it cannot be split into a valid date plus a 14-digit ID without guessing. Review the separate image regions.",
    }
