"""Garment Size Specification Table Extraction Package."""

from .cells.centering import (
    center_block_value,
    center_cell_value,
    detect_block_value,
    detect_cell_value,
    reconstruct_table_from_blocks,
    reconstruct_table_from_cells,
)
from .config import ExtractorConfig, HeaderBoxConfig, LLMConfig, TableDetectionConfig
from .detection.header_detector import (
    HeaderROI,
    attach_header_to_table,
    detect_header_box,
)
from .detection.table_detector import TableROI, detect_table
from .extractor import (
    SizeSpecExtractor,
    TableSegmentationResult,
    extract_header_box_image,
    extract_reconstructed_table_image,
    extract_table_image,
    extract_table_segments,
    load_image,
    save_table_image,
    save_table_segments,
)
from .llm import extract_with_llm
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
    "HeaderBoxConfig",
    "HeaderROI",
    "LLMConfig",
    "SizeSpecExtractor",
    "SizeSpecRow",
    "SizeSpecTable",
    "TableDetectionConfig",
    "TableROI",
    "TableSegmentationResult",
    "attach_header_to_table",
    "center_block_value",
    "center_cell_value",
    "detect_block_value",
    "detect_cell_value",
    "detect_header_box",
    "detect_table",
    "extract_header_box_image",
    "extract_reconstructed_table_image",
    "extract_table_image",
    "extract_table_segments",
    "extract_with_llm",
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
