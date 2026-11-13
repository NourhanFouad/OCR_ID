"""Conservative image loading/preprocessing helpers.

Input pixels are preserved by default. Perspective correction and contrast
changes are not applied blindly; PaddleOCR document preprocessing is configured
in the baseline adapter and will require validation before disabling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


EGYPTIAN_FRONT_ZONES = {
    "name": (0.54, 0.20, 0.93, 0.45),
    "address": (0.40, 0.46, 0.94, 0.65),
    "birth_date": (0.06, 0.69, 0.39, 0.85),
    "national_id": (0.39, 0.68, 0.97, 0.86),
    "serial_number": (0.07, 0.86, 0.39, 0.98),
}
EGYPTIAN_BACK_ZONES = {
    "national_id": (0.46, 0.035, 0.82, 0.17),
    "issue_date": (0.23, 0.035, 0.455, 0.17),
    "occupation": (0.30, 0.076, 0.98, 0.24),
    "gender": (0.67, 0.22, 0.80, 0.41),
    "religion": (0.49, 0.22, 0.67, 0.41),
    "marital_status": (0.32, 0.22, 0.47, 0.41),
    "expiry_date": (0.29, 0.38, 0.80, 0.55),
}


def load_rgb(path: str | Path) -> Image.Image:
    """Load EXIF-oriented RGB pixels without cropping or resizing."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image.copy()


def field_preprocessing_candidates(image: Image.Image) -> dict[str, Image.Image]:
    """Return conservative OCR retry variants; caller always retains the source."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OCR retry preprocessing requires opencv-python-headless and numpy.") from exc
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enlarged = cv2.resize(rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray_large = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray_large)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharpened = cv2.addWeighted(clahe, 1.35, blurred, -0.35, 0)
    adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 9)
    return {
        "upscaled_grayscale": Image.fromarray(gray_large),
        "upscaled_clahe_sharpen": Image.fromarray(sharpened),
        "adaptive_threshold": Image.fromarray(adaptive),
    }


def _crop_normalized_zones(image: Image.Image, zones: dict[str, tuple[float, float, float, float]], scale: float) -> dict[str, dict[str, Any]]:
    width, height = image.size
    result: dict[str, dict[str, Any]] = {}
    for name, (left, top, right, bottom) in zones.items():
        x1, y1 = int(round(width * left)), int(round(height * top))
        x2, y2 = int(round(width * right)), int(round(height * bottom))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.convert("RGB").crop((x1, y1, x2, y2))
        if scale != 1:
            crop = crop.resize((max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))), Image.Resampling.LANCZOS)
        result[name] = {"image": crop, "bbox_xyxy": [x1, y1, x2, y2], "scale": scale}
    return result


def crop_egyptian_front_zones(image: Image.Image, scale: float = 2.0) -> dict[str, dict[str, Any]]:
    """Return enlarged front-side region candidates and their source boxes.

    Intended for a front card that nearly fills the input image. The source is
    never changed; the profile is only an aid to recognition and human review.
    """
    if scale < 1:
        raise ValueError("scale must be at least 1")
    return _crop_normalized_zones(image, EGYPTIAN_FRONT_ZONES, scale)


def crop_egyptian_back_zones(image: Image.Image, scale: float = 3.0) -> dict[str, dict[str, Any]]:
    """Return enlarged approximate region candidates for a common ID back layout."""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    return _crop_normalized_zones(image, EGYPTIAN_BACK_ZONES, scale)


def crop_occupation_retry(image: Image.Image, scale: float = 3.0) -> dict[str, Any]:
    """Tight occupation-only crop that excludes the numeric row above it."""
    if scale < 1:
        raise ValueError("scale must be at least 1")
    return _crop_normalized_zones(image, {"occupation": (0.30, 0.085, 0.98, 0.24)}, scale)["occupation"]
