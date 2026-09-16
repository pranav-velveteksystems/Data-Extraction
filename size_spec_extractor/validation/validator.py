"""Validation layer for extracted size tables (Section 29)."""

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import ValidationConfig
from ..normalization.fractions import parse_fraction
from ..reconstruction.schema import SizeSpecTable

ALLOWED_NUMERIC_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?|[½¼¾⅛⅜⅝⅞]|\d+[½¼¾⅛⅜⅝⅞]|\d+\s*[-/ ]\s*\d+/\d+)$"
)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_table(table: SizeSpecTable, config: ValidationConfig) -> ValidationResult:
    """Validate table consistency, column counts, and numeric values (Section 29)."""
    errors: list[str] = []
    warnings: list[str] = []

    num_cols = len(table.columns)
    if num_cols == 0:
        errors.append("Table has no columns detected.")

    # 1. Size header validation (Section 29)
    valid_size_pattern = re.compile(
        r"^(\d+[-–]\d+\s*[A-Za-z]+|\d+\s*[A-Za-z]+|\d+[-–]\d+|[A-Za-z0-9_\-]+)$"
    )
    for c_idx, col in enumerate(table.columns):
        if not col or col.startswith("Size_") or not valid_size_pattern.match(col):
            warnings.append(
                f"Column {c_idx} ('{col}') does not match standard size format."
            )

    for r_idx, row in enumerate(table.rows):
        # 2. Column consistency: len(row['values']) == len(columns)
        if len(row.values) != num_cols:
            errors.append(
                f"Row {r_idx} ('{row.specification}') value count ({len(row.values)}) "
                f"does not match column count ({num_cols})."
            )

        # 3. Numeric and fraction validation
        for c_idx, val in enumerate(row.values):
            if val is None:
                continue

            # Stacked values
            vals_to_check = val if isinstance(val, list) else [val]
            for item in vals_to_check:
                item_str = str(item).strip()
                if not item_str:
                    continue

                if not ALLOWED_NUMERIC_PATTERN.match(item_str):
                    warnings.append(
                        f"Row {r_idx} Col {c_idx} ('{table.columns[c_idx]}'): "
                        f"Unexpected non-numeric value '{item_str}'."
                    )

                parsed = parse_fraction(item_str, unit=table.unit)
                if parsed is not None:
                    if parsed.numeric_value < config.min_measurement_value:
                        warnings.append(
                            f"Row {r_idx} Col {c_idx}: Negative or zero measurement value {parsed.numeric_value}."
                        )
                    if parsed.numeric_value > config.max_measurement_value:
                        warnings.append(
                            f"Row {r_idx} Col {c_idx}: Measurement value {parsed.numeric_value} "
                            f"exceeds maximum expected threshold ({config.max_measurement_value})."
                        )

    is_valid = len(errors) == 0
    if config.strict and len(warnings) > 0:
        is_valid = False

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
