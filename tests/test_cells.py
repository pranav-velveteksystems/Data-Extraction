"""Unit tests for cell creation, border removal, emptiness, and stacked value detection."""

import unittest

import cv2
import numpy as np

from size_spec_extractor.cells.border_remover import remove_borders
from size_spec_extractor.cells.cell_generator import create_cells
from size_spec_extractor.cells.text_region import detect_text_regions, is_empty_cell
from size_spec_extractor.config import CellProcessingConfig
from size_spec_extractor.detection.grid_detector import GridGeometry


class TestCells(unittest.TestCase):
    def test_create_cells(self):
        grid = GridGeometry(
            horizontal_lines=[0, 40, 80],
            vertical_lines=[0, 50, 100],
            table_width=100,
            table_height=80,
        )
        img = np.zeros((80, 100, 3), dtype=np.uint8)
        matrix = create_cells(img, grid)

        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 2)
        c00 = matrix[0][0]
        self.assertEqual(c00.row, 0)
        self.assertEqual(c00.col, 0)
        self.assertEqual(c00.bbox, (0, 0, 50, 40))

    def test_remove_borders(self):
        cell_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cfg = CellProcessingConfig(margin_ratio_x=0.10, margin_ratio_y=0.08)
        interior = remove_borders(cell_img, cfg)

        # 100 - 2*10 = 80 width; 100 - 2*8 = 84 height
        self.assertEqual(interior.shape[0], 84)
        self.assertEqual(interior.shape[1], 80)

    def test_is_empty_cell(self):
        # Blank white cell
        blank = np.ones((50, 50, 3), dtype=np.uint8) * 255
        self.assertTrue(is_empty_cell(blank))

        # Cell with substantial dark text/ink
        ink_cell = np.ones((50, 50, 3), dtype=np.uint8) * 255
        cv2.putText(
            ink_cell, "2", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )
        self.assertFalse(is_empty_cell(ink_cell))

    def test_stacked_value_detection(self):
        # Create a tall cell with two distinct vertically separated numbers: top "2", bottom "3"
        cell_img = np.ones((120, 60, 3), dtype=np.uint8) * 255
        cv2.putText(
            cell_img, "2", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )
        cv2.putText(
            cell_img, "3", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )

        cfg = CellProcessingConfig(detect_stacked_values=True)
        regions = detect_text_regions(cell_img, cfg)

        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].position_label, "top")
        self.assertEqual(regions[1].position_label, "bottom")
        self.assertLess(regions[0].y_position, regions[1].y_position)


if __name__ == "__main__":
    unittest.main()
