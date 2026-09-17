"""Cells package for cell creation, border removal, and text region extraction."""

from .border_remover import remove_borders
from .cell_generator import Cell, TextRegion, create_cells
from .centering import (
    center_block_value,
    center_cell_value,
    detect_block_value,
    detect_cell_value,
    reconstruct_table_from_blocks,
    reconstruct_table_from_cells,
)
from .text_region import (
    detect_text_regions,
    get_ink_mask,
    is_empty_cell,
    upscale_region,
)

__all__ = [
    "Cell",
    "TextRegion",
    "center_block_value",
    "center_cell_value",
    "create_cells",
    "detect_block_value",
    "detect_cell_value",
    "detect_text_regions",
    "get_ink_mask",
    "is_empty_cell",
    "reconstruct_table_from_blocks",
    "reconstruct_table_from_cells",
    "remove_borders",
    "upscale_region",
]
