"""Unit tests for edge cases, error handling, and boundary conditions."""

import unittest

import numpy as np

from size_spec_extractor.cells.border_remover import remove_borders
from size_spec_extractor.cells.text_region import is_empty_cell
from size_spec_extractor.config import (
    ExtractorConfig,
    LineDetectionConfig,
    ValidationConfig,
)
from size_spec_extractor.detection.grid_detector import detect_grid
from size_spec_extractor.detection.line_detector import merge_adjacent_lines
from size_spec_extractor.extractor import load_image
from size_spec_extractor.normalization.fractions import (
    normalize_fraction_string,
    parse_fraction,
)
from size_spec_extractor.normalization.sizes import normalize_size_header
from size_spec_extractor.reconstruction.schema import SizeSpecRow, SizeSpecTable
from size_spec_extractor.validation.validator import validate_table


class TestEdgeCases(unittest.TestCase):
    def test_invalid_image_path(self):
        with self.assertRaises(FileNotFoundError):
            load_image("/non/existent/path/image.png")

    def test_unsupported_image_type(self):
        with self.assertRaises(TypeError):
            load_image(12345)

    def test_tiny_and_empty_cells(self):
        # Empty array
        empty_arr = np.array([], dtype=np.uint8)
        self.assertTrue(is_empty_cell(empty_arr))

        # 2x2 tiny cell
        tiny = np.ones((2, 2, 3), dtype=np.uint8) * 255
        self.assertTrue(is_empty_cell(tiny))

        # Border removal on tiny cell should not collapse to negative dimensions
        cfg = ExtractorConfig().cell_processing
        cropped = remove_borders(tiny, cfg)
        self.assertGreaterEqual(cropped.size, 0)

    def test_fraction_boundary_values(self):
        # None and empty string
        self.assertIsNone(parse_fraction(None))
        self.assertIsNone(parse_fraction(""))
        self.assertIsNone(parse_fraction("   "))
        self.assertEqual(normalize_fraction_string(""), "")

        # Gibberish / non-number
        self.assertIsNone(parse_fraction("hello"))
        self.assertIsNone(parse_fraction("abc/def"))

        # Zero and large values
        res_zero = parse_fraction("0")
        self.assertIsNotNone(res_zero)
        self.assertEqual(res_zero.numeric_value, 0.0)

        res_large = parse_fraction("99 1/2")
        self.assertIsNotNone(res_large)
        self.assertEqual(res_large.numeric_value, 99.5)

    def test_size_header_edge_cases(self):
        self.assertEqual(normalize_size_header(""), "")
        self.assertEqual(normalize_size_header(None), "")
        self.assertEqual(normalize_size_header("---"), "")
        self.assertEqual(normalize_size_header("1-2y"), "1-2 Y")
        self.assertEqual(normalize_size_header("Free"), "FREE")

    def test_line_merging_edge_cases(self):
        # Empty list
        self.assertEqual(merge_adjacent_lines([]), [])

        # Single element
        self.assertEqual(merge_adjacent_lines([50]), [50])

        # Identical duplicates
        self.assertEqual(merge_adjacent_lines([50, 50, 50]), [50])

    def test_grid_with_no_lines(self):
        # Solid white image with zero lines detected
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cfg = LineDetectionConfig(h_kernel_length=30, v_kernel_length=30)
        grid = detect_grid(blank, cfg)
        # Should snap boundaries to [0, 200]
        self.assertEqual(grid.horizontal_lines, [0, 200])
        self.assertEqual(grid.vertical_lines, [0, 200])
        self.assertEqual(grid.num_rows, 1)
        self.assertEqual(grid.num_cols, 1)

    def test_validation_extremes(self):
        # Negative measurement
        tbl_neg = SizeSpecTable(
            unit="inch",
            columns=["0-3 M"],
            rows=[SizeSpecRow(specification="Chest", values=["-5"])],
        )
        res_neg = validate_table(tbl_neg, ValidationConfig(strict=True))
        self.assertFalse(res_neg.is_valid)

        # Extreme large measurement
        tbl_huge = SizeSpecTable(
            unit="inch",
            columns=["0-3 M"],
            rows=[SizeSpecRow(specification="Chest", values=["500"])],
        )
        res_huge = validate_table(tbl_huge, ValidationConfig(strict=True))
        self.assertFalse(res_huge.is_valid)


if __name__ == "__main__":
    unittest.main()
