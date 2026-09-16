"""Visual debugging and intermediate image generation (Section 22, 23)."""

from __future__ import annotations

import os

import cv2
import numpy as np

from ..cells.cell_generator import Cell
from ..detection.grid_detector import GridGeometry


class VisualDebugger:
    """Orchestrates saving all 8 visual debugging artifacts described in Section 23."""

    def __init__(self, debug_dir: str | None = None):
        self.debug_dir = debug_dir
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    def is_enabled(self) -> bool:
        return self.debug_dir is not None

    def save_image(self, name: str, image: np.ndarray):
        if not self.is_enabled():
            return
        path = os.path.join(self.debug_dir, name)
        cv2.imwrite(path, image)

    def stage_01_original(self, original_img: np.ndarray):
        self.save_image("01_original.jpg", original_img)

    def stage_02_warped(self, warped_img: np.ndarray):
        self.save_image("02_warped.jpg", warped_img)

    def stage_03_table_detection(
        self, image: np.ndarray, table_bbox: tuple[int, int, int, int]
    ):
        """Draw bounding box around detected table ROI."""
        vis = image.copy()
        x1, y1, x2, y2 = table_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            vis,
            "Detected Table",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        self.save_image("03_table_detection.jpg", vis)

    def stage_04_horizontal_lines(self, table_img: np.ndarray, h_lines: list[int]):
        """Draw detected horizontal lines."""
        vis = table_img.copy()
        w = vis.shape[1]
        for y in h_lines:
            cv2.line(vis, (0, y), (w, y), (0, 0, 255), 2)
        self.save_image("04_horizontal_lines.jpg", vis)

    def stage_05_vertical_lines(self, table_img: np.ndarray, v_lines: list[int]):
        """Draw detected vertical lines."""
        vis = table_img.copy()
        h = vis.shape[0]
        for x in v_lines:
            cv2.line(vis, (x, 0), (x, h), (255, 0, 0), 2)
        self.save_image("05_vertical_lines.jpg", vis)

    def stage_06_grid(self, table_img: np.ndarray, grid: GridGeometry):
        """Draw reconstructed grid: horizontal lines in red, vertical lines in blue (Section 23)."""
        vis = table_img.copy()
        w = vis.shape[1]
        h = vis.shape[0]

        for y in grid.horizontal_lines:
            cv2.line(vis, (0, y), (w, y), (0, 0, 255), 2)  # Red horizontal
        for x in grid.vertical_lines:
            cv2.line(vis, (x, 0), (x, h), (255, 0, 0), 2)  # Blue vertical

        self.save_image("06_grid.jpg", vis)

    def stage_07_cells(self, table_img: np.ndarray, cells: list[list[Cell]]):
        """Draw individual cell boxes labeled R#C# (Section 23)."""
        vis = table_img.copy()
        for row in cells:
            for cell in row:
                cv2.rectangle(
                    vis, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 255), 1
                )
                label = f"R{cell.row}C{cell.col}"
                cv2.putText(
                    vis,
                    label,
                    (cell.x1 + 3, cell.y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 128, 255),
                    1,
                )
        self.save_image("07_cells.jpg", vis)

    def stage_08_ocr_results(self, table_img: np.ndarray, cells: list[list[Cell]]):
        """Draw cell bounding boxes with OCR text and confidences (Section 22, 23)."""
        vis = table_img.copy()
        for row in cells:
            for cell in row:
                cv2.rectangle(
                    vis, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 200, 0), 1
                )
                text = cell.single_raw_text or "null"
                # Abbreviate long text for drawing
                if len(text) > 10:
                    text = text[:9] + ".."
                label = f"{text}"
                conf_label = (
                    f"{int(cell.confidence * 100)}%" if not cell.is_empty else ""
                )
                cv2.putText(
                    vis,
                    label,
                    (cell.x1 + 2, cell.y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 180),
                    1,
                )
                if conf_label:
                    cv2.putText(
                        vis,
                        conf_label,
                        (cell.x1 + 2, cell.y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (80, 80, 80),
                        1,
                    )

        self.save_image("08_ocr_results.jpg", vis)
