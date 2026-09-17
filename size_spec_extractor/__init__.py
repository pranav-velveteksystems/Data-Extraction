"""Garment Size Specification Table Extraction Package."""

from .cells.centering import (
    center_block_value,
    center_cell_value,
    detect_block_value,
    detect_cell_value,
    reconstruct_table_from_blocks,
    reconstruct_table_from_cells,
)
from .config import ExtractorConfig, TableDetectionConfig
from .detection.table_detector import TableROI, detect_table
from .extractor import (
    SizeSpecExtractor,
    TableSegmentationResult,
    extract_reconstructed_table_image,
    extract_table_image,
    extract_table_segments,
    load_image,
    save_table_image,
    save_table_segments,
)
from .output.html import generate_html_table, save_html
from .output.json import save_json, to_json
from .reconstruction.schema import (
    CellCoordinateMetadata,
    ExtractionResult,
    GarmentSpecDocument,
    SizeSpecRow,
    SizeSpecTable,
)

__all__ = [
    "CellCoordinateMetadata",
    "ExtractionResult",
    "ExtractorConfig",
    "GarmentSpecDocument",
    "SizeSpecExtractor",
    "SizeSpecRow",
    "SizeSpecTable",
    "TableDetectionConfig",
    "TableROI",
    "TableSegmentationResult",
    "center_block_value",
    "center_cell_value",
    "detect_block_value",
    "detect_cell_value",
    "detect_table",
    "extract_reconstructed_table_image",
    "extract_table_image",
    "extract_table_segments",
    "generate_html_table",
    "load_image",
    "reconstruct_table_from_blocks",
    "reconstruct_table_from_cells",
    "save_html",
    "save_json",
    "save_table_image",
    "save_table_segments",
    "to_json",
]
