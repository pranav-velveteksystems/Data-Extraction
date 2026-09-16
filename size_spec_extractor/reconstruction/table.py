"""Table reconstruction from cell matrix (Section 9, 19, 20, 21, 22)."""

from __future__ import annotations

from typing import Any

from ..cells.cell_generator import Cell
from ..config import ReconstructionConfig
from ..normalization.fractions import parse_fraction
from ..normalization.sizes import normalize_size_header
from ..normalization.values import correct_numeric_text
from .schema import (
    CellCoordinateMetadata,
    GarmentSpecDocument,
    SizeSpecRow,
    SizeSpecTable,
)


def reconstruct_table(
    cells: list[list[Cell]], config: ReconstructionConfig
) -> tuple[GarmentSpecDocument, list[CellCoordinateMetadata]]:
    """Reconstruct 2D matrix into semantic GarmentSpecDocument and coordinate metadata."""
    if not cells or not cells[0]:
        doc = GarmentSpecDocument(
            category=config.category,
            style_code=config.style_code,
            name=config.name,
            size_spec_table=SizeSpecTable(unit=config.unit, columns=[], rows=[]),
        )
        return doc, []

    num_rows = len(cells)
    num_cols = len(cells[0])

    header_row_idx = config.header_row_index
    spec_col_idx = config.spec_col_index

    # 1. Extract Size Headers from Header Row
    raw_columns: list[str] = []
    header_cells = cells[header_row_idx]
    for c in range(spec_col_idx + 1, num_cols):
        cell = header_cells[c]
        text = cell.single_raw_text or ""
        norm_col = normalize_size_header(text) if text else f"Size_{c}"
        raw_columns.append(norm_col)

    # 2. Extract Specification Rows
    rows: list[SizeSpecRow] = []
    metadata: list[CellCoordinateMetadata] = []

    for r in range(num_rows):
        for c in range(num_cols):
            cell = cells[r][c]
            # Record intermediate coordinates metadata (Section 22)
            meta = CellCoordinateMetadata(
                row=r,
                column=c,
                bbox=[cell.x1, cell.y1, cell.x2, cell.y2],
                ocr={
                    "text": cell.single_raw_text if not cell.is_empty else None,
                    "confidence": round(cell.confidence, 3),
                },
                status=cell.status if not cell.is_empty else "empty",
            )
            metadata.append(meta)

    # Process data rows (skip header row)
    for r in range(num_rows):
        if r == header_row_idx:
            continue

        spec_cell = cells[r][spec_col_idx]
        spec_name = spec_cell.single_raw_text
        if not spec_name or spec_cell.is_empty:
            spec_name = f"Spec_Row_{r}"

        row_values: list[Any] = []
        row_parsed_floats: list[float | None] = []

        for c in range(spec_col_idx + 1, num_cols):
            cell = cells[r][c]
            if cell.is_empty or not cell.raw_texts:
                row_values.append(None)
                row_parsed_floats.append(None)
            elif len(cell.raw_texts) == 1:
                raw_t = cell.raw_texts[0]
                norm_t = correct_numeric_text(raw_t)
                if norm_t:
                    row_values.append(norm_t)
                    parsed = parse_fraction(norm_t, unit=config.unit)
                    row_parsed_floats.append(parsed.numeric_value if parsed else None)
                else:
                    row_values.append(None)
                    row_parsed_floats.append(None)
            else:
                # Vertically stacked values (Section 17 & 18)
                stacked_raw = [correct_numeric_text(t) for t in cell.raw_texts]
                valid_stacked = [t for t in stacked_raw if t]
                if not valid_stacked:
                    row_values.append(None)
                    row_parsed_floats.append(None)
                elif len(valid_stacked) == 1:
                    row_values.append(valid_stacked[0])
                    parsed = parse_fraction(valid_stacked[0], unit=config.unit)
                    row_parsed_floats.append(parsed.numeric_value if parsed else None)
                else:
                    row_values.append(valid_stacked)
                    first_parsed = parse_fraction(valid_stacked[0], unit=config.unit)
                    row_parsed_floats.append(
                        first_parsed.numeric_value if first_parsed else None
                    )

        rows.append(
            SizeSpecRow(
                specification=spec_name,
                values=row_values,
                parsed_values=row_parsed_floats,
            )
        )

    spec_table = SizeSpecTable(unit=config.unit, columns=raw_columns, rows=rows)

    doc = GarmentSpecDocument(
        category=config.category,
        style_code=config.style_code,
        name=config.name,
        size_spec_table=spec_table,
        fabric_colour_code_table=[],
        remarks="",
    )

    return doc, metadata
