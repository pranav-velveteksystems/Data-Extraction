"""Grid reconstruction from detected horizontal and vertical lines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import LineDetectionConfig
from ..preprocessing.enhancement import to_grayscale
from ..preprocessing.threshold import binarize
from .line_detector import (
    detect_horizontal_lines,
    detect_vertical_lines,
    merge_adjacent_lines,
)


@dataclass
class GridGeometry:
    horizontal_lines: list[int]
    vertical_lines: list[int]
    table_width: int
    table_height: int

    @property
    def num_rows(self) -> int:
        return max(0, len(self.horizontal_lines) - 1)

    @property
    def num_cols(self) -> int:
        return max(0, len(self.vertical_lines) - 1)

    def get_cell_bbox(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Get (x1, y1, x2, y2) bounding box for cell at (row, col)."""
        if row < 0 or row >= self.num_rows or col < 0 or col >= self.num_cols:
            raise IndexError(
                f"Cell ({row}, {col}) out of grid bounds ({self.num_rows}, {self.num_cols})"
            )

        x1 = self.vertical_lines[col]
        x2 = self.vertical_lines[col + 1]
        y1 = self.horizontal_lines[row]
        y2 = self.horizontal_lines[row + 1]
        return (x1, y1, x2, y2)


def filter_dense_lines(lines: list[int], min_spacing: int) -> list[int]:
    """Filter out lines that are too closely spaced compared to min_spacing."""
    if len(lines) <= 1:
        return lines

    filtered = [lines[0]]
    for p in lines[1:]:
        if p - filtered[-1] >= min_spacing:
            filtered.append(p)
    return filtered


def ensure_table_boundaries(
    lines: list[int],
    dimension_size: int,
    min_cell_size: int,
    max_boundary_snap: int = 25,
) -> list[int]:
    """Ensure table has valid outer boundaries (0 and dimension_size) without creating phantom rows."""
    lines = sorted(set(lines))
    if not lines:
        return [0, dimension_size]

    if len(lines) >= 2:
        spacings = [lines[i] - lines[i - 1] for i in range(1, len(lines))]
        median_spacing = float(np.median(spacings))
        snap_thresh = max(
            min_cell_size, min(max_boundary_snap, int(0.25 * median_spacing))
        )
    else:
        snap_thresh = min_cell_size

    # Check start boundary: snap close outer line to 0 to avoid phantom margin rows
    if lines[0] < snap_thresh:
        lines[0] = 0
    else:
        lines = [0] + lines

    # Check end boundary: snap close outer line to dimension_size
    if dimension_size - lines[-1] < snap_thresh:
        lines[-1] = dimension_size
    else:
        lines = lines + [dimension_size]

    filtered = filter_dense_lines(lines, min_cell_size)
    if lines[-1] == dimension_size and filtered[-1] != dimension_size:
        if len(filtered) > 1 and dimension_size - filtered[-1] < min_cell_size:
            filtered[-1] = dimension_size
        else:
            filtered.append(dimension_size)
    return filtered


def detect_grid(
    table_image: np.ndarray, config: LineDetectionConfig | None = None
) -> GridGeometry:
    """Detect complete grid geometry from table image."""
    if config is None:
        config = LineDetectionConfig()
    h, w = table_image.shape[:2]
    gray = to_grayscale(table_image)
    binary = binarize(gray, method="adaptive", invert=True)

    h_lines = detect_horizontal_lines(binary, config)
    v_lines = detect_vertical_lines(binary, config)

    # Merge closely detected lines (lines closer than min_cell_size belong to same physical line)
    h_merge_dist = max(config.line_merge_distance, config.min_cell_height - 1)
    v_merge_dist = max(config.line_merge_distance, config.min_cell_width - 1)
    h_lines = merge_adjacent_lines(h_lines, distance_threshold=h_merge_dist)
    v_lines = merge_adjacent_lines(v_lines, distance_threshold=v_merge_dist)

    # Ensure valid boundaries
    h_lines = ensure_table_boundaries(h_lines, h, config.min_cell_height)
    v_lines = ensure_table_boundaries(v_lines, w, config.min_cell_width)

    return GridGeometry(
        horizontal_lines=h_lines, vertical_lines=v_lines, table_width=w, table_height=h
    )
