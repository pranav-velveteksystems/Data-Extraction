"""Detection package for table, lines, and grid geometry."""

from .grid_detector import GridGeometry, detect_grid
from .line_detector import (
    detect_horizontal_lines,
    detect_vertical_lines,
    find_projection_peaks,
    merge_adjacent_lines,
)
from .table_detector import TableROI, detect_table, detect_table_by_grid_density

__all__ = [
    "GridGeometry",
    "TableROI",
    "detect_grid",
    "detect_horizontal_lines",
    "detect_table",
    "detect_table_by_grid_density",
    "detect_vertical_lines",
    "find_projection_peaks",
    "merge_adjacent_lines",
]
