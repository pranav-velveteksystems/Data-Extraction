"""Fraction parsing and normalization (Section 14, 30)."""

from __future__ import annotations

import re
from dataclasses import dataclass

UNICODE_FRACTIONS = {
    "½": (1, 2, 0.5),
    "¼": (1, 4, 0.25),
    "¾": (3, 4, 0.75),
    "⅛": (1, 8, 0.125),
    "⅜": (3, 8, 0.375),
    "⅝": (5, 8, 0.625),
    "⅞": (7, 8, 0.875),
    "⅓": (1, 3, 1 / 3),
    "⅔": (2, 3, 2 / 3),
}

VALUE_TO_UNICODE = {
    0.5: "½",
    0.25: "¼",
    0.75: "¾",
    0.125: "⅛",
    0.375: "⅜",
    0.625: "⅝",
    0.875: "⅞",
}


@dataclass
class FractionValue:
    raw: str
    numeric_value: float
    unit: str = "inch"

    def to_dict(self):
        return {"raw": self.raw, "numeric_value": self.numeric_value, "unit": self.unit}


def normalize_fraction_string(text: str) -> str:
    """Normalize ascii fractions like '2 1/2', '21/2', '2-1/2' to unicode '2½' (Section 30)."""
    if not text:
        return ""
    t = text.strip()

    # Case: Integer + space + unicode fraction, e.g. "2 ½", "4 ½" (Section 30)
    m_unicode_space = re.match(r"^(\d+)\s+([½¼¾⅛⅜⅝⅞⅓⅔])$", t)
    if m_unicode_space:
        return f"{m_unicode_space.group(1)}{m_unicode_space.group(2)}"

    # Direct match for ascii fractions
    # Case: Integer + space/dash + fraction, e.g. "2 1/2", "2-1/2"
    m_int_frac = re.match(r"^(\d+)\s*[- ]\s*(\d+)/(\d+)$", t)
    if m_int_frac:
        whole, num, den = m_int_frac.groups()
        val = int(num) / int(den)
        if val in VALUE_TO_UNICODE:
            return f"{whole}{VALUE_TO_UNICODE[val]}"

    # Case: Run-together e.g. "21/2" -> whole 2, 1/2
    m_run_frac = re.match(r"^(\d)(\d)/(\d+)$", t)
    if m_run_frac:
        whole, num, den = m_run_frac.groups()
        val = int(num) / int(den)
        if val in VALUE_TO_UNICODE:
            return f"{whole}{VALUE_TO_UNICODE[val]}"

    # Case: Pure fraction e.g. "1/2" -> "½"
    m_pure_frac = re.match(r"^(\d+)/(\d+)$", t)
    if m_pure_frac:
        num, den = m_pure_frac.groups()
        val = int(num) / int(den)
        if val in VALUE_TO_UNICODE:
            return VALUE_TO_UNICODE[val]

    # Case: Decimal e.g. "2.5" -> "2½", "2.25" -> "2¼", "2.75" -> "2¾"
    m_dec = re.match(r"^(\d+)\.50?$", t)
    if m_dec:
        return f"{m_dec.group(1)}½"
    if t in ("0.5", "0.50"):
        return "½"

    m_dec25 = re.match(r"^(\d+)\.25$", t)
    if m_dec25:
        return f"{m_dec25.group(1)}¼"
    if t == "0.25":
        return "¼"

    m_dec75 = re.match(r"^(\d+)\.75$", t)
    if m_dec75:
        return f"{m_dec75.group(1)}¾"
    if t == "0.75":
        return "¾"

    return t


def parse_fraction(text: str | None, unit: str = "inch") -> FractionValue | None:
    """Parse text representation of measurement into raw display and float value (Section 14)."""
    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    norm_raw = normalize_fraction_string(cleaned)

    # 1. Unicode fraction with integer prefix: e.g. "2½"
    for char, (_, _, frac_val) in UNICODE_FRACTIONS.items():
        if char in norm_raw:
            prefix = norm_raw.replace(char, "").strip()
            whole = float(prefix) if prefix else 0.0
            total = whole + frac_val
            return FractionValue(raw=norm_raw, numeric_value=total, unit=unit)

    # 2. ASCII fraction with whole: e.g. "2 1/2" or "21/2"
    m_frac = re.match(r"^(\d+)?\s*[- ]?\s*(\d+)/(\d+)$", cleaned)
    if m_frac:
        whole_s, num_s, den_s = m_frac.groups()
        whole = float(whole_s) if whole_s else 0.0
        frac = float(num_s) / float(den_s)
        total = whole + frac
        return FractionValue(raw=norm_raw, numeric_value=total, unit=unit)

    # 3. Decimal or plain integer: e.g. "2.5", "3"
    m_num = re.match(r"^(\d+(?:\.\d+)?)$", cleaned)
    if m_num:
        val = float(m_num.group(1))
        return FractionValue(raw=norm_raw, numeric_value=val, unit=unit)

    return None
