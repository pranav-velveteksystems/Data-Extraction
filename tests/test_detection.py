"""Unit tests for line detection, merging, and grid geometry reconstruction."""

import os
import unittest

import cv2
import numpy as np

from size_spec_extractor.config import LineDetectionConfig, TableDetectionConfig
from size_spec_extractor.detection.grid_detector import (
    GridGeometry,
    detect_grid,
    ensure_table_boundaries,
)
from size_spec_extractor.detection.line_detector import (
    detect_horizontal_lines,
    detect_vertical_lines,
    merge_adjacent_lines,
)
from size_spec_extractor.detection.table_detector import (
    detect_table,
)
from size_spec_extractor.extractor import extract_table_segments


class TestDetection(unittest.TestCase):
    def test_merge_adjacent_lines(self):
        # Section 7 example: [100, 101, 102, 103] -> single line
        lines = [100, 101, 102, 103, 200, 201, 350]
        merged = merge_adjacent_lines(lines, distance_threshold=5)
        self.assertEqual(len(merged), 3)
        self.assertAlmostEqual(merged[0], 102, delta=1)
        self.assertAlmostEqual(merged[1], 200, delta=1)
        self.assertEqual(merged[2], 350)

    def test_grid_geometry(self):
        geom = GridGeometry(
            horizontal_lines=[0, 50, 100, 150],
            vertical_lines=[0, 80, 160],
            table_width=160,
            table_height=150,
        )
        self.assertEqual(geom.num_rows, 3)
        self.assertEqual(geom.num_cols, 2)
        bbox = geom.get_cell_bbox(1, 1)
        self.assertEqual(bbox, (80, 50, 160, 100))

    def test_detect_grid_on_synthetic_table(self):
        # Create a clean white 300x400 image with black grid lines
        h, w = 300, 400
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Horizontal lines at y = 0, 100, 200, 300
        for y in [0, 100, 200, 299]:
            cv2.line(img, (0, y), (w, y), (0, 0, 0), 2)

        # Vertical lines at x = 0, 100, 200, 300, 399
        for x in [0, 100, 200, 300, 399]:
            cv2.line(img, (x, 0), (x, h), (0, 0, 0), 2)

        config = LineDetectionConfig(
            h_kernel_length=30,
            v_kernel_length=30,
            min_cell_height=15,
            min_cell_width=15,
        )
        grid = detect_grid(img, config)

        # Expect 3 rows and 4 columns
        self.assertEqual(grid.num_rows, 3)
        self.assertEqual(grid.num_cols, 4)

        # Check line positions roughly around 0, 100, 200, 300
        for expected_y in [0, 100, 200, 300]:
            closest_h = min(
                grid.horizontal_lines, key=lambda val: abs(val - expected_y)
            )
            self.assertLessEqual(abs(closest_h - expected_y), 5)

        for expected_x in [0, 100, 200, 300, 400]:
            closest_v = min(grid.vertical_lines, key=lambda val: abs(val - expected_x))
            self.assertLessEqual(abs(closest_v - expected_x), 5)

    def test_detect_table_relative_and_manual(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        # Manual
        cfg_manual = TableDetectionConfig(mode="manual", manual_bbox=(50, 60, 200, 300))
        roi = detect_table(img, cfg_manual)
        self.assertEqual(roi.bbox, (50, 60, 200, 300))
        self.assertEqual(roi.width, 150)
        self.assertEqual(roi.height, 240)

        # Relative
        cfg_rel = TableDetectionConfig(
            mode="relative", relative_roi=(0.1, 0.2, 0.8, 0.9)
        )
        roi_rel = detect_table(img, cfg_rel)
        self.assertEqual(roi_rel.bbox, (100, 50, 450, 400))

    def test_detect_lines_noise_rejection(self):
        # Image with short text stroke (45px on 500px width) must NOT be detected as grid line
        img = np.zeros((200, 500), dtype=np.uint8)
        cv2.line(img, (100, 50), (145, 50), 255, 1)
        cfg = LineDetectionConfig()
        h_lines = detect_horizontal_lines(img, cfg)
        self.assertEqual(h_lines, [])

        # Blank image with tiny speckle
        img_speckle = np.zeros((200, 300), dtype=np.uint8)
        cv2.line(img_speckle, (10, 50), (25, 50), 255, 1)
        v_lines = detect_vertical_lines(img_speckle, cfg)
        self.assertEqual(v_lines, [])

    def test_ensure_table_boundaries_dense_rows(self):
        # In a dense table (20px rows), a missing outer boundary should not destroy row 0 or row N
        lines_missing_top = [20, 40, 60, 80, 100]
        res_top = ensure_table_boundaries(
            lines_missing_top, dimension_size=100, min_cell_size=14
        )
        self.assertEqual(res_top, [0, 20, 40, 60, 80, 100])

        lines_missing_bottom = [0, 25, 50, 75]
        res_bot = ensure_table_boundaries(
            lines_missing_bottom, dimension_size=100, min_cell_size=14
        )
        self.assertEqual(res_bot, [0, 25, 50, 75, 100])

    def test_image1_image2_image3_all_correct_grid(self):
        # image1: 4 rows x 13 cols (cell [2][0] and [3][0] split)
        if os.path.exists("tests/image1.png"):
            res1 = extract_table_segments("tests/image1.png")
            self.assertEqual(res1.num_rows, 4)
            self.assertEqual(res1.num_cols, 13)
            self.assertEqual(res1.total_cells, 52)
            self.assertLess(res1.get_cell(2, 0).raw_crop.shape[0], 300)
            self.assertLess(res1.get_cell(3, 0).raw_crop.shape[0], 300)

        # image2: 9 rows x 13 cols
        if os.path.exists("tests/image2.jpeg"):
            res2 = extract_table_segments("tests/image2.jpeg")
            self.assertEqual(res2.num_rows, 9)
            self.assertEqual(res2.num_cols, 13)
            self.assertEqual(res2.total_cells, 117)

        # image3: 8 rows x 13 cols
        if os.path.exists("tests/image3.jpeg"):
            res3 = extract_table_segments("tests/image3.jpeg")
            self.assertEqual(res3.num_rows, 8)
            self.assertEqual(res3.num_cols, 13)
            self.assertEqual(res3.total_cells, 104)


if __name__ == "__main__":
    unittest.main()
