"""Size code formatting and normalization (Section 19)."""

import re

# Standard garment size tokens
STANDARD_SIZES = {"NB", "XXS", "XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "FREE"}


def normalize_size_header(header: str) -> str:
    """Normalize a size header string according to Section 19.

    Examples:
        "0-3M"  -> "0-3 M"
        "6-12M" -> "6-12 M"
        "1-2Y"  -> "1-2 Y"
        "4 - 5 Y" -> "4-5 Y"
        "S"     -> "S"
    """
    if not header:
        return ""

    cleaned = header.strip()
    # Remove unwanted leading/trailing symbols
    cleaned = cleaned.strip(".,;:|_- ")

    # Check if standard adult/general size
    upper_cleaned = cleaned.upper()
    if upper_cleaned in STANDARD_SIZES:
        return upper_cleaned

    # Pattern: Range with unit, e.g. "0-3M", "0 - 3 M", "6-12m", "1-2Y"
    m_range = re.match(r"^(\d+)\s*[-–/]\s*(\d+)\s*([A-Za-z]+)$", cleaned)
    if m_range:
        start, end, unit = m_range.groups()
        return f"{start}-{end} {unit.upper()}"

    # Pattern: Single number with unit, e.g. "3M", "2Y", "10Y"
    m_single = re.match(r"^(\d+)\s*([A-Za-z]+)$", cleaned)
    if m_single:
        num, unit = m_single.groups()
        return f"{num} {unit.upper()}"

    # Pattern: Range without unit, e.g. "0-3", "1-2"
    m_range_nounit = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
    if m_range_nounit:
        return f"{m_range_nounit.group(1)}-{m_range_nounit.group(2)}"

    return cleaned


def normalize_size_list(headers: list[str]) -> list[str]:
    """Normalize a list of size column headers."""
    return [normalize_size_header(h) for h in headers]
