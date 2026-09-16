"""Cell generation from grid geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..detection.grid_detector import GridGeometry


@dataclass
class TextRegion:
    crop: np.ndarray
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) relative to cell
    y_position: int
    position_label: str = "single"  # "single", "top", "bottom", etc.
    raw_text: str | None = None
    confidence: float = 0.0


@dataclass
class Cell:
    row: int
    col: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) relative to table
    raw_crop: np.ndarray
    interior_crop: np.ndarray | None = None
    is_empty: bool = False
    text_regions: list[TextRegion] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)
    normalized_values: list[Any] = field(default_factory=list)
    confidence: float = 1.0
    status: str = "pending"  # "empty", "accepted", "needs_review", "rejected"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def single_value(self) -> Any:
        """Convenience property for single-value cells."""
        if self.is_empty or not self.normalized_values:
            return None
        if len(self.normalized_values) == 1:
            return self.normalized_values[0]
        return self.normalized_values

    @property
    def single_raw_text(self) -> str | None:
        """Convenience property for single-value raw text."""
        if self.is_empty or not self.raw_texts:
            return None
        if len(self.raw_texts) == 1:
            return self.raw_texts[0]
        return " / ".join(self.raw_texts)


def create_cells(table_image: np.ndarray, grid: GridGeometry) -> list[list[Cell]]:
    """Slice table image into a 2D matrix of Cell objects based on grid geometry."""
    matrix: list[list[Cell]] = []

    for r in range(grid.num_rows):
        row_cells: list[Cell] = []
        for c in range(grid.num_cols):
            x1, y1, x2, y2 = grid.get_cell_bbox(r, c)
            raw_crop = table_image[y1:y2, x1:x2].copy()
            cell = Cell(row=r, col=c, bbox=(x1, y1, x2, y2), raw_crop=raw_crop)
            row_cells.append(cell)
        matrix.append(row_cells)

    return matrix
