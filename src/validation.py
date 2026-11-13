"""Non-destructive validation helpers for synthetic EGID OCR outputs.

Raw OCR is never rewritten. Normalization is returned as a separate candidate
used only by syntactic checks; no check silently repairs a transcription.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any


DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def digits_for_validation(raw: str) -> str:
    """Return a digit-mapped view for checks; keep `raw` unchanged elsewhere."""
    return "".join(ch for ch in raw.translate(DIGIT_TRANSLATION) if ch.isdigit())


def format_date_candidate(raw: str | None, allow_year_month: bool = False) -> str | None:
    """Return a separate YYYY/MM[/DD] display candidate only when unambiguous.

    The original transcription is never changed. Eight digits are treated as
    YYYYMMDD only when they form a real calendar date; six digits are allowed
    as YYYYMM only for explicitly partial issue-date fields.
    """
    if not raw:
        return None
    digit_chars = [char for char in raw if char.translate(DIGIT_TRANSLATION).isdigit()]
    digits = "".join(char.translate(DIGIT_TRANSLATION) for char in digit_chars)
    if allow_year_month:
        year_month = re.fullmatch(r"\s*([٠-٩0-9]{4})\s*[/.-]\s*([٠-٩0-9]{2})\s*", raw)
        month_year = re.fullmatch(r"\s*([٠-٩0-9]{2})\s*[/.-]\s*([٠-٩0-9]{4})\s*", raw)
        if year_month:
            year_text, month_text = year_month.groups()
            year, month = int(digits_for_validation(year_text)), int(digits_for_validation(month_text))
            if 1900 <= year <= 2100 and 1 <= month <= 12:
                return f"{month_text}/{year_text}"
        if month_year:
            month_text, year_text = month_year.groups()
            year, month = int(digits_for_validation(year_text)), int(digits_for_validation(month_text))
            if 1900 <= year <= 2100 and 1 <= month <= 12:
                return f"{month_text}/{year_text}"
    if len(digits) == 8:
        year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
        try:
            date(year, month, day)
        except ValueError:
            return None
        return f"{''.join(digit_chars[:4])}/{''.join(digit_chars[4:6])}/{''.join(digit_chars[6:8])}"
    if allow_year_month and len(digits) == 6:
        year, month = int(digits[:4]), int(digits[4:6])
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return f"{''.join(digit_chars[4:6])}/{''.join(digit_chars[:4])}"
    return None


def inspect_date_format(raw: str | None, allow_year_month: bool = False) -> dict[str, Any]:
    """Describe whether a date can be formatted without changing its digits."""
    result = {"status": "missing" if not raw else "ambiguous", "reason": "No unambiguous date format was found; the raw OCR is preserved."}
    if not raw:
        return result
    chunks = re.findall(r"[٠-٩0-9]+", raw)
    ascii_chunks = [digits_for_validation(chunk) for chunk in chunks]
    if len(ascii_chunks) == 2 and len(ascii_chunks[0]) == 4:
        year, month = int(ascii_chunks[0]), int(ascii_chunks[1])
        if allow_year_month and len(ascii_chunks[1]) == 1:
            return {"status": "incomplete_month", "reason": "Issue-date month has one OCR digit; no missing digit was inferred."}
        if not 1 <= month <= 12:
            return {"status": "invalid_month", "reason": f"Month {chunks[1]} is outside 1–12; no correction was applied."}
        if allow_year_month and 1900 <= year <= 2100:
            return {"status": "valid_year_month", "reason": "Valid year/month candidate; raw OCR is unchanged."}
    if allow_year_month and len(ascii_chunks) == 2 and len(ascii_chunks[0]) == 2 and len(ascii_chunks[1]) == 4:
        month, year = int(ascii_chunks[0]), int(ascii_chunks[1])
        if 1900 <= year <= 2100 and 1 <= month <= 12:
            return {"status": "valid_year_month", "reason": "Valid month/year candidate; raw OCR is unchanged."}
    if len(ascii_chunks) == 3 and len(ascii_chunks[0]) == 4:
        year, month, day = map(int, ascii_chunks)
        try:
            date(year, month, day)
        except ValueError:
            if not 1 <= month <= 12:
                return {"status": "invalid_month", "reason": f"Month {chunks[1]} is outside 1–12; no correction was applied."}
            return {"status": "invalid_calendar_date", "reason": "This is not a valid calendar date; raw OCR is unchanged."}
        return {"status": "valid_date", "reason": "Valid calendar-date candidate; raw OCR is unchanged."}
    display = format_date_candidate(raw, allow_year_month=allow_year_month)
    if display:
        return {"status": "valid_formatted_candidate", "reason": "Digits are unambiguous and the display format was added separately."}
    digit_count = len(digits_for_validation(raw))
    if any(len(chunk) > 2 for chunk in ascii_chunks[1:]):
        bad = next(chunk for chunk in chunks[1:] if len(digits_for_validation(chunk)) > 2)
        return {"status": "invalid_month", "reason": f"Month token {bad} has more than two digits; no digit was removed."}
    return {"status": "ambiguous", "reason": f"Found {digit_count} digits; cannot insert separators safely. Raw OCR is unchanged."}


def validate_national_id_format(raw_text: str | None) -> dict[str, Any]:
    """Check known Egyptian ID structure without accepting/correcting OCR text."""
    compact = digits_for_validation(raw_text or "")
    checks: list[dict[str, Any]] = []
    exact = len(compact) == 14
    checks.append({"name": "14_digit_format", "passed": exact,
                   "message": "Candidate contains 14 digits; visual review is still required." if exact else f"Candidate contains {len(compact)} digits; expected 14. No digits were added or corrected."})
    if exact:
        century = compact[0]
        checks.append({"name": "century_digit", "passed": century in "23",
                       "message": "Leading digit 2/3 identifies the 1900/2000 century." if century in "23" else f"Leading digit {century} is not a supported current ID century marker."})
        if century in "23":
            year = (1900 if century == "2" else 2000) + int(compact[1:3])
            month, day = int(compact[3:5]), int(compact[5:7])
            try:
                date(year, month, day)
                date_ok, date_message = True, f"Encoded birth date {year:04d}-{month:02d}-{day:02d} is a calendar date."
            except ValueError:
                date_ok, date_message = False, f"Encoded birth date {year:04d}-{month:02d}-{day:02d} is not a calendar date."
            checks.append({"name": "encoded_birth_date", "passed": date_ok, "message": date_message})
        else:
            checks.append({"name": "encoded_birth_date", "passed": False, "message": "Cannot decode date with unsupported century digit."})
        governorate = compact[7:9]
        valid_governorates = {
            "01", "02", "03", "04", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "31", "32", "33", "34", "35", "88",
        }
        gov_ok = governorate in valid_governorates
        checks.append({"name": "governorate_code", "passed": gov_ok,
                       "message": f"Governorate code {governorate} is {'recognized' if gov_ok else 'not in the recognized code set'}; 88 denotes outside Egypt."})
        checks.append({"name": "checksum", "passed": None, "message": "No checksum is defined by this validation; a structurally valid number is not proof of correct recognition."})
    return {
        "field_name": "national_id",
        "raw_text": raw_text,
        "validation_candidate": compact if exact else None,
        "checks": checks,
        "status": "needs_review",
        "accepted_value": None,
    }


def derive_birth_date_from_national_id(raw_text: str | None) -> dict[str, Any]:
    """Derive a review-only DOB candidate from a structurally valid Egyptian ID."""
    compact = digits_for_validation(raw_text or "")
    result: dict[str, Any] = {
        "value": None,
        "source": "national_id_encoded_date",
        "status": "unavailable",
        "reason": "A valid 14-digit national ID with a decodable date is required.",
    }
    if len(compact) != 14 or compact[0] not in "23":
        return result
    structural = validate_national_id_format(compact)
    if any(check["passed"] is False for check in structural["checks"]):
        result["reason"] = "The ID does not pass the available structural checks, so no birth date was derived."
        return result
    year = (1900 if compact[0] == "2" else 2000) + int(compact[1:3])
    month, day = int(compact[3:5]), int(compact[5:7])
    try:
        date(year, month, day)
    except ValueError:
        result["reason"] = "The ID's encoded date is not a valid calendar date."
        return result
    value = f"{year:04d}/{month:02d}/{day:02d}"
    if any("٠" <= char <= "٩" for char in (raw_text or "")):
        value = value.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    result.update({
        "value": value,
        "status": "derived_needs_human_review",
        "reason": "Date decoded from the national ID; compare it with the printed date and verify every ID digit visually.",
    })
    return result


def reconcile_birth_date_candidates(
    printed_date: str,
    id_date: str | None,
    id_confidence: float | None,
    printed_confidence: float | None,
) -> dict[str, Any]:
    """Keep the printed date unless a valid ID date conflicts and scores higher.

    Scores are only a user-requested tie-breaker; neither value is accepted as
    verified ground truth, and the printed OCR candidate is always retained.
    """
    result = {
        "printed_date_candidate": printed_date,
        "id_date_candidate": id_date,
        "value": printed_date,
        "source": "printed_date_ocr",
        "conflict": False,
        "id_preferred": False,
    }
    if not id_date or digits_for_validation(printed_date) == digits_for_validation(id_date):
        return result
    result["conflict"] = True
    if (
        isinstance(id_confidence, (int, float))
        and isinstance(printed_confidence, (int, float))
        and id_confidence > printed_confidence
    ):
        result.update({"value": id_date, "source": "national_id_encoded_date", "id_preferred": True})
    return result


def validate_field(field_name: str, raw_text: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "field_name": field_name,
        "raw_text": raw_text,
        "validation_candidate": None,
        "checks": [],
        "status": "missing" if raw_text is None or not raw_text.strip() else "needs_review",
        "accepted_value": None,
    }
    if raw_text is None or not raw_text.strip():
        result["checks"].append({"name": "present", "passed": False, "message": "No text detected; value remains null."})
        return result

    text = raw_text.strip()
    result["validation_candidate"] = text.translate(DIGIT_TRANSLATION)

    if field_name == "national_id":
        compact = "".join(text.split()).translate(DIGIT_TRANSLATION)
        ok = re.fullmatch(r"\d{14}", compact) is not None
        digits = compact if ok else digits_for_validation(text)
        result["checks"].append({"name": "14_digit_format", "passed": ok, "message": "Generated IDs are 14 digits after removing spaces." if ok else "Expected 14 digits for this synthetic generator."})
        if ok and digits[0] not in "23":
            result["checks"].append({"name": "generator_century_marker", "passed": False, "message": "Generator uses 2 for years before 2000 and 3 for years from 2000 onward."})
        elif ok:
            result["checks"].append({"name": "generator_century_marker", "passed": True, "message": "Century marker is one used by the upstream generator."})
        result["checks"].append({"name": "gender_or_governorate_digit", "passed": None, "message": "Not checked: the upstream generator uses an unconstrained random seven-digit tail."})

    elif field_name == "birth_date":
        candidate = text.translate(DIGIT_TRANSLATION).replace("-", "/").replace(".", "/")
        match = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", candidate)
        try:
            if not match:
                raise ValueError("Expected a printed year/month/day date.")
            y, m, d = map(int, match.groups())
            date(y, m, d)
            result["checks"].append({"name": "calendar_date", "passed": True, "message": "Valid calendar date."})
        except ValueError as exc:
            result["checks"].append({"name": "calendar_date", "passed": False, "message": f"Unchanged transcription needs review: {exc}"})

    elif field_name in {"issue_date", "expiry_date"}:
        candidate = text.translate(DIGIT_TRANSLATION).replace("-", "/").replace(".", "/")
        if field_name == "issue_date":
            valid = format_date_candidate(text, allow_year_month=True) is not None
        else:
            match = re.fullmatch(r"(\d{4})/(\d{1,2})(?:/(\d{1,2}))?", candidate)
            valid = False
            if match:
                year, month, day = match.groups()
                try:
                    valid = 1 <= int(month) <= 12
                    if valid and day is not None:
                        date(int(year), int(month), int(day))
                except ValueError:
                    valid = False
        result["checks"].append({"name": "date_format_and_calendar", "passed": valid, "message": "Expiry date generation is known to produce impossible dates; preserve the exact text and review it." if field_name == "expiry_date" else ("Valid issue-date format." if valid else "Issue date did not match the expected year/month format." )})

    elif field_name == "gender":
        normalized = re.sub(r"\s+", "", text)
        recognized = normalized in {"ذكر", "أنثى"}
        result["checks"].append({"name": "known_generator_value", "passed": recognized, "message": "Expected one of the two values emitted by the generator; visual review is still required."})

    else:
        result["checks"].append({"name": "transcription_preserved", "passed": None, "message": "No automatic semantic rewrite or acceptance rule is defined for this field."})

    result["status"] = "needs_review"
    result["accepted_value"] = None
    return result
