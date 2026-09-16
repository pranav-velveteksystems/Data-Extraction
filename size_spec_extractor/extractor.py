"""Main SizeSpecExtractor pipeline orchestrating CV, OCR, and table reconstruction."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .cells.border_remover import remove_borders
from .cells.cell_generator import Cell, create_cells
from .cells.text_region import detect_text_regions, is_empty_cell
from .config import ExtractorConfig
from .detection.grid_detector import GridGeometry, detect_grid
from .detection.table_detector import TableROI, detect_table
from .ocr.engine import BaseOCREngine, get_ocr_engine
from .ocr.header_ocr import HeaderOCR
from .ocr.numeric_ocr import NumericOCR
from .preprocessing.perspective import correct_perspective
from .reconstruction.schema import ExtractionResult
from .reconstruction.table import reconstruct_table
from .validation.validator import validate_table
from .visualization.debug import VisualDebugger


def load_image(image_input: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    """Load image from path, numpy array, or PIL Image into BGR numpy array."""
    if isinstance(image_input, (str, Path)):
        image_path = str(image_input)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        return img
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def extract_header_metadata(
    image: np.ndarray, ocr_engine: BaseOCREngine, table_top_y: int
) -> tuple[str, str, str]:
    """Extract Category, Style Code, and Name from region above table if available."""
    category = ""
    style_code = ""
    name = ""

    if table_top_y <= 40:
        return category, style_code, name

    top_region = image[0:table_top_y, :].copy()
    try:
        res = ocr_engine.ocr(top_region, psm=6)
        text = res.text

        # Style code pattern: e.g. "IDF 134", "IDF-134", "Style: IDF 134"
        style_match = re.search(
            r"(?:STYLE|CODE|STYLE\s*NO\.?)?\s*[:\s]?\s*([A-Z]{2,4}\s*[-_]?\s*\d{2,5})",
            text,
            re.IGNORECASE,
        )
        if style_match:
            style_code = style_match.group(1).upper()

        # Category pattern: e.g. "Designen Frock", "Frock", "Dress"
        cat_match = re.search(
            r"(?:CATEGORY|ITEM)?\s*[:\s]?\s*([A-Za-z\s]{3,20}(?:Frock|Dress|Shirt|Pant|Suit|Top))",
            text,
            re.IGNORECASE,
        )
        if cat_match:
            category = cat_match.group(1).strip()
    except (IndexError, AttributeError, ValueError, TypeError):
        pass

    return category, style_code, name


def save_table_image(table_image: np.ndarray, output_path: str | Path) -> str:
    """Save an extracted table image to disk.

    Args:
        table_image: Extracted table image as numpy BGR array.
        output_path: Target path to save the image (e.g. 'table.png', 'output.jpg').

    Returns:
        Absolute path to the saved image file.
    """
    if table_image is None or table_image.size == 0:
        raise ValueError("Cannot save empty or None table image.")
    out_str = str(output_path)
    out_dir = os.path.dirname(out_str)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    success = cv2.imwrite(out_str, table_image)
    if not success:
        raise OSError(f"Failed to write table image to: {out_str}")
    return os.path.abspath(out_str)


@dataclass
class TableSegmentationResult:
    """Result of table extraction and cell segmentation."""

    table_image: np.ndarray
    cells: list[list[Cell]]
    table_roi: TableROI
    grid: GridGeometry
    output_dir: str | None = None
    table_image_path: str | None = None
    cell_image_paths: dict[tuple[int, int], str] = field(default_factory=dict)

    def __iter__(self):
        yield self.table_image
        yield self.cells
        yield self.table_roi
        yield self.grid

    def __len__(self) -> int:
        return 4

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            str_map = {
                "table": self.table_image,
                "table_image": self.table_image,
                "cells": self.cells,
                "roi": self.table_roi,
                "table_roi": self.table_roi,
                "grid": self.grid,
                "output_dir": self.output_dir,
                "table_image_path": self.table_image_path,
            }
            if key in str_map:
                return str_map[key]
            m = re.match(r"^\[(\d+)\]\[(\d+)\]$", key)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                return self.get_cell(r, c)
            raise KeyError(f"Key {key!r} not found in TableSegmentationResult")

        if isinstance(key, tuple) and len(key) == 2:
            r, c = key
            if isinstance(r, int) and isinstance(c, int):
                return self.get_cell(r, c)

        seq = (self.table_image, self.cells, self.table_roi, self.grid)
        return seq[key]

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            return item in self.cell_image_paths or (
                0 <= item[0] < self.num_rows and 0 <= item[1] < self.num_cols
            )
        if isinstance(item, str):
            return item in {
                "table",
                "table_image",
                "cells",
                "roi",
                "table_roi",
                "grid",
                "output_dir",
                "table_image_path",
            }
        return item in (self.table_image, self.cells, self.table_roi, self.grid)

    @property
    def num_rows(self) -> int:
        return self.grid.num_rows

    @property
    def num_cols(self) -> int:
        return self.grid.num_cols

    @property
    def total_cells(self) -> int:
        return self.num_rows * self.num_cols

    def get_cell(self, row: int, col: int) -> Cell:
        """Get cell by (row, col) indices."""
        return self.cells[row][col]

    def get_cell_crop(self, row: int, col: int) -> np.ndarray:
        """Get cropped image for cell at (row, col)."""
        cell = self.get_cell(row, col)
        return cell.raw_crop if hasattr(cell, "raw_crop") else cell

    def get_cell_image(self, row: int, col: int) -> np.ndarray:
        """Alias for get_cell_crop."""
        return self.get_cell_crop(row, col)

    def get_cell_image_path(self, row: int, col: int) -> str | None:
        """Get saved image file path for cell at (row, col)."""
        return self.cell_image_paths.get((row, col))

    def to_dict(self) -> dict[str, Any]:
        """Export segmentation metadata as dictionary."""
        return {
            "output_dir": self.output_dir,
            "table_image_path": self.table_image_path,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "total_cells": self.total_cells,
            "cell_image_paths": {
                f"[{r}][{c}]": path for (r, c), path in self.cell_image_paths.items()
            },
            "table_bbox": self.table_roi.bbox,
        }


def save_table_segments(
    table_image: np.ndarray | Image.Image,
    cells: list[list[Cell]] | list[Cell],
    output_dir: str | Path,
    table_filename: str = "table.png",
    cell_format: str = "png",
) -> dict[str, str]:
    """Save the main table image and individual cell crops into an output folder.

    Files are named:
    - Main table: <table_filename> (e.g. 'table.png')
    - Cell crops: '[r][c].png' matching the 2D row/column grid indices.

    Args:
        table_image: Extracted table image as numpy BGR array or PIL Image.
        cells: 2D matrix of Cell objects (or flat list of Cell objects).
        output_dir: Target directory path.
        table_filename: Filename for the main table image (default 'table.png').
        cell_format: Extension for cell crops without leading dot (default 'png').

    Returns:
        Dictionary mapping identifiers ('table', '[r][c]') to absolute file paths.
    """
    if table_image is None:
        raise ValueError("Cannot save empty or None table image.")

    if isinstance(table_image, Image.Image):
        table_img_arr = cv2.cvtColor(
            np.array(table_image.convert("RGB")), cv2.COLOR_RGB2BGR
        )
    else:
        table_img_arr = table_image

    if getattr(table_img_arr, "size", 0) == 0:
        raise ValueError("Cannot save empty or None table image.")

    out_dir_str = str(output_dir) if output_dir else "."
    os.makedirs(out_dir_str, exist_ok=True)
    abs_out_dir = os.path.abspath(out_dir_str)

    saved_paths: dict[str, str] = {}

    # Save main table image
    table_file_path = os.path.join(abs_out_dir, table_filename)
    parent_dir = os.path.dirname(table_file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    success = cv2.imwrite(table_file_path, table_img_arr)
    if not success:
        raise OSError(f"Failed to write table image to: {table_file_path}")
    saved_paths["table"] = os.path.abspath(table_file_path)

    # Normalize cells to list of (row, col, cell)
    cell_items: list[tuple[int, int, Any]] = []
    if cells and not isinstance(cells[0], (list, tuple)):
        for idx, item in enumerate(cells):
            r = getattr(item, "row", idx)
            c = getattr(item, "col", 0)
            cell_items.append((r, c, item))
    else:
        for r, row in enumerate(cells):
            for c, item in enumerate(row):
                cell_r = getattr(item, "row", r)
                cell_c = getattr(item, "col", c)
                cell_items.append((cell_r, cell_c, item))

    ext = cell_format.lstrip(".") if cell_format else "png"

    # Save segmented cell boxes as 2D array names: [0][0].png, [0][1].png, ...
    for r, c, item in cell_items:
        crop = getattr(item, "raw_crop", item)
        if isinstance(crop, Image.Image):
            crop = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)

        if crop is None or getattr(crop, "size", 0) == 0:
            channels = (
                table_img_arr.shape[2] if getattr(table_img_arr, "ndim", 3) > 2 else 0
            )
            shape = (1, 1, channels) if channels > 0 else (1, 1)
            crop = np.zeros(shape, dtype=getattr(table_img_arr, "dtype", np.uint8))

        cell_name = f"[{r}][{c}].{ext}"
        cell_file_path = os.path.join(abs_out_dir, cell_name)
        success = cv2.imwrite(cell_file_path, crop)
        if not success:
            raise OSError(f"Failed to write cell image to: {cell_file_path}")
        saved_paths[f"[{r}][{c}]"] = os.path.abspath(cell_file_path)

    return saved_paths


def extract_table_segments(
    image_input: str | Path | np.ndarray | Image.Image,
    output_dir: str | Path | None = None,
    config: ExtractorConfig | None = None,
    table_filename: str = "table.png",
    cell_format: str = "png",
) -> TableSegmentationResult:
    """Detect table, extract grid geometry, and segment individual cells into a folder.

    Args:
        image_input: Path to image file, numpy BGR/grayscale array, or PIL Image.
        output_dir: Optional directory path where main table and cell images are saved.
        config: Optional ExtractorConfig instance.
        table_filename: Name of the main table image file inside output_dir (default 'table.png').
        cell_format: Extension for cell crops without leading dot (default 'png').

    Returns:
        TableSegmentationResult containing table_image, cells, table_roi, grid, and saved paths.
    """
    if config is None:
        config = ExtractorConfig()

    image = load_image(image_input)

    debugger = VisualDebugger(config.debug_dir if config.debug else None)
    debugger.stage_01_original(image)

    if config.preprocessing.enable_perspective_correction:
        warped, _corrected, _ = correct_perspective(
            image, min_area_ratio=config.preprocessing.min_quad_area_ratio
        )
    else:
        warped = image

    debugger.stage_02_warped(warped)

    table_roi: TableROI = detect_table(warped, config.table_detection)
    debugger.stage_03_table_detection(warped, table_roi.bbox)

    table_img = table_roi.image

    # Grid Detection
    grid: GridGeometry = detect_grid(table_img, config.line_detection)
    debugger.stage_04_horizontal_lines(table_img, grid.horizontal_lines)
    debugger.stage_05_vertical_lines(table_img, grid.vertical_lines)
    debugger.stage_06_grid(table_img, grid)

    # Cell Generation
    cells = create_cells(table_img, grid)
    debugger.stage_07_cells(table_img, cells)

    table_path = None
    cell_paths: dict[tuple[int, int], str] = {}
    abs_out_dir = None

    if output_dir is not None:
        abs_out_dir = os.path.abspath(str(output_dir) if str(output_dir) else ".")
        saved_dict = save_table_segments(
            table_img,
            cells,
            output_dir,
            table_filename=table_filename,
            cell_format=cell_format,
        )
        table_path = saved_dict.get("table")
        for r, row in enumerate(cells):
            for c, item in enumerate(row):
                cell_r = getattr(item, "row", r)
                cell_c = getattr(item, "col", c)
                key = f"[{cell_r}][{cell_c}]"
                if key in saved_dict:
                    cell_paths[(cell_r, cell_c)] = saved_dict[key]

    return TableSegmentationResult(
        table_image=table_img,
        cells=cells,
        table_roi=table_roi,
        grid=grid,
        output_dir=abs_out_dir,
        table_image_path=table_path,
        cell_image_paths=cell_paths,
    )


def extract_table_image(
    image_input: str | Path | np.ndarray | Image.Image,
    output_path: str | Path | None = None,
    config: ExtractorConfig | None = None,
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, TableROI]:
    """Detect and extract the size specification table from an input image and optionally save it.

    Args:
        image_input: Path to image file, numpy BGR/grayscale array, or PIL Image.
        output_path: Optional file path where the extracted table image will be saved.
        config: Optional ExtractorConfig instance.
        output_dir: Optional directory path where main table and cell segments will be saved.

    Returns:
        tuple of (table_image, table_roi):
            - table_image: Cropped table as numpy BGR array.
            - table_roi: TableROI dataclass containing image, bbox (x1, y1, x2, y2), and confidence.
    """
    if config is None:
        config = ExtractorConfig()

    image = load_image(image_input)

    debugger = VisualDebugger(config.debug_dir if config.debug else None)
    debugger.stage_01_original(image)

    if config.preprocessing.enable_perspective_correction:
        warped, _corrected, _ = correct_perspective(
            image, min_area_ratio=config.preprocessing.min_quad_area_ratio
        )
    else:
        warped = image

    debugger.stage_02_warped(warped)

    table_roi: TableROI = detect_table(warped, config.table_detection)
    debugger.stage_03_table_detection(warped, table_roi.bbox)

    if output_path is not None:
        save_table_image(table_roi.image, output_path)

    if output_dir is not None:
        grid: GridGeometry = detect_grid(table_roi.image, config.line_detection)
        cells = create_cells(table_roi.image, grid)
        save_table_segments(table_roi.image, cells, output_dir)

    return table_roi.image, table_roi


class SizeSpecExtractor:
    """Complete CV + OCR pipeline for Garment Size Specification Sheets."""

    def __init__(
        self,
        config: ExtractorConfig | None = None,
        ocr_engine: BaseOCREngine | None = None,
    ):
        self.config = config or ExtractorConfig()
        self._ocr_engine = ocr_engine
        self._numeric_ocr = None
        self._header_ocr = None
        self.debugger = VisualDebugger(
            self.config.debug_dir if self.config.debug else None
        )

    @property
    def ocr_engine(self) -> BaseOCREngine:
        if self._ocr_engine is None:
            self._ocr_engine = get_ocr_engine(self.config.ocr.engine)
        return self._ocr_engine

    @ocr_engine.setter
    def ocr_engine(self, engine: BaseOCREngine):
        self._ocr_engine = engine
        self._numeric_ocr = None
        self._header_ocr = None

    @property
    def numeric_ocr(self) -> NumericOCR:
        if self._numeric_ocr is None:
            self._numeric_ocr = NumericOCR(self.ocr_engine, self.config.ocr)
        return self._numeric_ocr

    @property
    def header_ocr(self) -> HeaderOCR:
        if self._header_ocr is None:
            self._header_ocr = HeaderOCR(self.ocr_engine, self.config.ocr)
        return self._header_ocr

    def extract_segments(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        output_dir: str | Path | None = None,
        table_filename: str = "table.png",
        cell_format: str = "png",
    ) -> TableSegmentationResult:
        """Extract size spec table and segment all grid cells, saving to output_dir if specified."""
        return extract_table_segments(
            image_input,
            output_dir=output_dir,
            table_filename=table_filename,
            cell_format=cell_format,
            config=self.config,
        )

    def extract_table_and_cells(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        output_dir: str | Path | None = None,
        table_filename: str = "table.png",
        cell_format: str = "png",
    ) -> TableSegmentationResult:
        """Alias for extract_segments."""
        return self.extract_segments(
            image_input,
            output_dir=output_dir,
            table_filename=table_filename,
            cell_format=cell_format,
        )

    def extract_table(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> tuple[np.ndarray, TableROI]:
        """Detect and extract size specification table, optionally saving to output_path."""
        return extract_table_image(
            image_input,
            output_path=output_path,
            config=self.config,
            output_dir=output_dir,
        )

    def extract_table_image(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> tuple[np.ndarray, TableROI]:
        """Alias for extract_table."""
        return self.extract_table(
            image_input,
            output_path=output_path,
            output_dir=output_dir,
        )

    def extract(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        output_table_image: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> ExtractionResult:
        """Run the end-to-end extraction pipeline (Section 33)."""
        # 1. LOAD & PREPROCESS (Section 3 & 4)
        image = load_image(image_input)
        self.debugger.stage_01_original(image)

        if self.config.preprocessing.enable_perspective_correction:
            warped, _corrected, _ = correct_perspective(
                image, min_area_ratio=self.config.preprocessing.min_quad_area_ratio
            )
        else:
            warped, _corrected = image, False

        self.debugger.stage_02_warped(warped)

        # 2. TABLE DETECTION (Section 5)
        table_roi: TableROI = detect_table(warped, self.config.table_detection)
        self.debugger.stage_03_table_detection(warped, table_roi.bbox)

        if output_table_image is not None:
            save_table_image(table_roi.image, output_table_image)

        # Try extracting header metadata from above table if not provided
        cat, style, name = extract_header_metadata(
            warped, self.ocr_engine, table_roi.y1
        )
        recon_config = self.config.reconstruction
        if not recon_config.category and cat:
            recon_config.category = cat
        if not recon_config.style_code and style:
            recon_config.style_code = style
        if not recon_config.name and name:
            recon_config.name = name

        table_img = table_roi.image

        # 3. GRID DETECTION (Section 6, 7, 8, 9)
        grid: GridGeometry = detect_grid(table_img, self.config.line_detection)
        self.debugger.stage_04_horizontal_lines(table_img, grid.horizontal_lines)
        self.debugger.stage_05_vertical_lines(table_img, grid.vertical_lines)
        self.debugger.stage_06_grid(table_img, grid)

        # 4. CELL GENERATION (Section 9)
        cells = create_cells(table_img, grid)
        self.debugger.stage_07_cells(table_img, cells)

        if output_dir is not None:
            save_table_segments(table_img, cells, output_dir)

        # 5. CELL OCR (Section 10, 11, 12, 13, 15, 17, 18, 24, 25, 26)
        num_rows = len(cells)
        num_cols = len(cells[0]) if num_rows > 0 else 0

        header_r = recon_config.header_row_index
        spec_c = recon_config.spec_col_index

        for r in range(num_rows):
            for c in range(num_cols):
                cell = cells[r][c]

                # Inset grid borders (Section 10)
                interior = remove_borders(cell.raw_crop, self.config.cell_processing)
                cell.interior_crop = interior

                # Check emptiness BEFORE OCR (Section 15)
                if is_empty_cell(
                    interior, min_ink_ratio=self.config.cell_processing.min_ink_ratio
                ):
                    cell.is_empty = True
                    cell.status = "empty"
                    cell.confidence = 1.0
                    cell.raw_texts = []
                    continue

                cell.is_empty = False

                # Header row cell: Mode 1 - Header OCR (Section 13)
                if r == header_r:
                    res = self.header_ocr.ocr_header(interior)
                    cell.raw_texts = [res.text] if res.text else []
                    cell.confidence = res.confidence
                    cell.status = "accepted" if res.text else "empty"

                # Specification column label cell: Mode 1 - Header/Text OCR (Section 13)
                elif c == spec_c:
                    res = self.header_ocr.ocr_spec_label(interior)
                    cell.raw_texts = [res.text] if res.text else []
                    cell.confidence = res.confidence
                    cell.status = "accepted" if res.text else "empty"

                # Measurement data cell: Mode 2 - Measurement OCR (Section 13, 17, 18)
                else:
                    # Detect text regions, handling stacked values (Section 17, 18)
                    regions = detect_text_regions(interior, self.config.cell_processing)
                    if not regions:
                        cell.is_empty = True
                        cell.status = "empty"
                        continue

                    raw_vals = []
                    confs = []
                    for reg in regions:
                        ocr_res = self.numeric_ocr.ocr_single_region(reg.crop)
                        reg.raw_text = ocr_res.text
                        reg.confidence = ocr_res.confidence
                        if ocr_res.text:
                            raw_vals.append(ocr_res.text)
                            confs.append(ocr_res.confidence)

                    cell.text_regions = regions
                    cell.raw_texts = raw_vals
                    cell.confidence = float(np.mean(confs)) if confs else 0.0
                    cell.status = self.numeric_ocr.classify_status(cell.confidence)

        self.debugger.stage_08_ocr_results(table_img, cells)

        # 6, 7, 8. RECONSTRUCTION & NORMALIZATION (Section 14, 19, 20, 21, 22)
        document, metadata = reconstruct_table(cells, recon_config)

        # 9. VALIDATION (Section 29)
        val_result = validate_table(document.size_spec_table, self.config.validation)

        # 10. RESULT
        return ExtractionResult(
            document=document,
            metadata=metadata if self.config.include_coordinates else [],
            table_bbox=table_roi.bbox,
            validation=val_result.to_dict(),
        )
