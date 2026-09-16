"""Cells package for cell creation, border removal, and text region extraction."""

from .border_remover import remove_borders
from .cell_generator import Cell, TextRegion, create_cells
from .text_region import (
    detect_text_regions,
    get_ink_mask,
    is_empty_cell,
    upscale_region,
)

__all__ = [
    "Cell",
    "TextRegion",
    "create_cells",
    "detect_text_regions",
    "get_ink_mask",
    "is_empty_cell",
    "remove_borders",
    "upscale_region",
]
