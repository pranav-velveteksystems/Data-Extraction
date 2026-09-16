"""Correction dictionary and value normalization for numeric measurement cells (Section 30)."""

from __future__ import annotations

import re

from .fractions import FractionValue, normalize_fraction_string, parse_fraction

# Common OCR confusion substitutions in numeric cells
NUMERIC_CORRECTIONS = {
    "I": "1",
    "l": "1",
    "|": "1",
    "!": "1",
    "i": "1",
    "O": "0",
    "o": "0",
    "D": "0",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
}


def correct_numeric_text(text: str) -> str:
    """Apply correction dictionary to OCR output when cell is expected to be numeric (Section 30)."""
    if not text:
        return ""

    t = text.strip()
    # Replace individual character confusions
    corrected_chars = []
    for ch in t:
        if ch in NUMERIC_CORRECTIONS:
            corrected_chars.append(NUMERIC_CORRECTIONS[ch])
        else:
            corrected_chars.append(ch)

    result = "".join(corrected_chars)

    # Clean leading/trailing punctuation or noise characters, keeping numbers and fraction signs
    result = re.sub(r"^[^\d½¼¾]+", "", result)
    result = re.sub(r"[^\d½¼¾/]+$", "", result)

    # Normalize fractions like 2 1/2 -> 2½
    result = normalize_fraction_string(result)

    # If result contains no digits or fraction characters, it is noise
    if not re.search(r"[\d½¼¾⅛⅜⅝⅞⅓⅔]", result):
        return ""

    return result


def normalize_measurement_value(
    text: str | None, unit: str = "inch"
) -> FractionValue | None:
    """Normalize a measurement text string into display and parsed value."""
    if not text:
        return None

    cleaned = correct_numeric_text(text)
    if not cleaned:
        return None

    frac = parse_fraction(cleaned, unit=unit)
    if frac is not None:
        return frac

    return None
