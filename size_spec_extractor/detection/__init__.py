"""Detection package for table, lines, and grid geometry."""

from .grid_detector import GridGeometry, detect_grid
from .header_detector import HeaderROI, attach_header_to_table, detect_header_box
from .line_detector import (
    detect_horizontal_lines,
    detect_vertical_lines,
    find_projection_peaks,
    merge_adjacent_lines,
)
from .table_detector import TableROI, detect_table, detect_table_by_grid_density

__all__ = [
    "GridGeometry",
    "HeaderROI",
    "TableROI",
    "attach_header_to_table",
    "detect_grid",
    "detect_header_box",
    "detect_horizontal_lines",
    "detect_table",
    "detect_table_by_grid_density",
    "detect_vertical_lines",
    "find_projection_peaks",
    "merge_adjacent_lines",
]
