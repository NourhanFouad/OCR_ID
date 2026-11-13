"""Source-layout field regions for the fixed synthetic EGID v1 template.

These are broad field ROIs from audited generator anchor coordinates, not
per-image tight text boxes. Assignment is attempted only for manually selected
side and exact 1080x1080 inputs; every candidate still requires human review.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


TEMPLATE_SIZE = (1080, 1080)
FIELD_ROIS: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "front": {
        "first_name": (620, 430, 870, 469),
        "paternal_names": (390, 470, 870, 515),
        "address": (485, 516, 870, 555),
        "governorate": (540, 556, 870, 595),
        "birth_date": (205, 605, 440, 660),
        "national_id": (575, 625, 875, 685),
        "serial_number": (280, 685, 475, 745),
    },
    "back": {
        "national_id": (500, 325, 785, 382),
        "issue_date": (285, 325, 505, 382),
        "occupation": (625, 378, 785, 423),
        "gender": (595, 428, 785, 490),
        "expiry_date": (405, 495, 615, 560),
    },
}


def classify_template_side(image: Image.Image) -> dict[str, Any]:
    """Conservative v1 side check using the back-only barcode strip.

    Thresholds were checked against the 400 source-labeled 1080x1080 cards.
    It returns unknown outside that exact canvas or in the gray zone. This is
    not a classifier validated for photographs of real identity cards.
    """
    if image.size != TEMPLATE_SIZE:
        return {"side": "unknown", "source": "template_barcode_heuristic", "score": None, "status": "unsupported_canvas"}
    gray = np.asarray(image.convert("L"), dtype=np.int16)
    barcode_region = gray[585:725, 250:900]
    transitions = np.abs(np.diff(barcode_region, axis=1))
    score = float((transitions > 50).mean())
    if score <= 0.03:
        side, status = "front", "classified_with_template_heuristic"
    elif score >= 0.30:
        side, status = "back", "classified_with_template_heuristic"
    else:
        side, status = "unknown", "ambiguous_barcode_score"
    return {"side": side, "source": "template_barcode_heuristic", "score": score, "status": status}


def _center(box: Any) -> tuple[float, float] | None:
    if hasattr(box, "tolist"):
        box = box.tolist()
    try:
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            x1, y1, x2, y2 = map(float, box)
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        points = [(float(point[0]), float(point[1])) for point in box]
        if not points:
            return None
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    except (TypeError, ValueError, IndexError):
        return None


def assign_template_fields(lines: list[dict[str, Any]], side: str, image_size: tuple[int, int]) -> dict[str, Any] | None:
    """Map a sole recognized detection to a source-defined ROI, never auto-accept it."""
    if side not in FIELD_ROIS or tuple(image_size) != TEMPLATE_SIZE:
        return None
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in FIELD_ROIS[side]}
    for line in lines:
        center = _center(line.get("bbox"))
        if center is None:
            continue
        x, y = center
        matches = []
        for field, (x1, y1, x2, y2) in FIELD_ROIS[side].items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                matches.append(field)
        if len(matches) == 1:
            assigned[matches[0]].append(line)

    fields: dict[str, Any] = {}
    for field, candidates in assigned.items():
        one = candidates[0] if len(candidates) == 1 else None
        fields[field] = {
            "raw_value": one.get("raw_text") if one else None,
            "confidence": one.get("confidence") if one else None,
            "bbox": one.get("bbox") if one else None,
            "candidate_count": len(candidates),
            "status": "needs_human_review" if one else "missing_or_ambiguous",
            "source": "EGID v1 fixed-template ROI; side selected by user",
        }
    return fields
