"""Tests for cell value detection, centering, and table image reconstruction."""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from size_spec_extractor.cells.cell_generator import Cell
from size_spec_extractor.cells.centering import (
    center_cell_value,
    detect_block_value,
    detect_cell_value,
    reconstruct_table_from_blocks,
    reconstruct_table_from_cells,
)
from size_spec_extractor.detection.grid_detector import GridGeometry
from size_spec_extractor.extractor import (
    SizeSpecExtractor,
    TableSegmentationResult,
    extract_reconstructed_table_image,
    extract_table_segments,
    save_table_segments,
)
from size_spec_extractor.main import main


class TestCenteringAndReconstruction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_img2 = "tests/image2.jpeg"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_detect_cell_value_empty(self):
        empty = np.ones((60, 40, 3), dtype=np.uint8) * 255
        self.assertIsNone(detect_cell_value(empty))
        self.assertIsNone(detect_block_value(empty))

    def test_detect_cell_value_tiny(self):
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        self.assertIsNone(detect_cell_value(tiny))

    def test_detect_cell_value_position(self):
        cell = np.ones((100, 60, 3), dtype=np.uint8) * 255
        # Place text at top: y=25
        cv2.putText(cell, "2", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        bbox = detect_cell_value(cell)
        self.assertIsNotNone(bbox)
        _x1, y1, _x2, y2 = bbox  # type: ignore
        cy = (y1 + y2) / 2.0
        self.assertLess(cy, 35.0)

    def test_center_cell_value_moves_to_center(self):
        # Create a tall cell with value at the top
        cell = np.ones((100, 60, 3), dtype=np.uint8) * 250
        cv2.putText(cell, "2", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        orig_bbox = detect_cell_value(cell)
        self.assertIsNotNone(orig_bbox)
        orig_cy = (orig_bbox[1] + orig_bbox[3]) / 2.0  # type: ignore
        self.assertLess(orig_cy, 30.0)

        centered = center_cell_value(cell)
        self.assertEqual(centered.shape, cell.shape)

        new_bbox = detect_cell_value(centered)
        self.assertIsNotNone(new_bbox)
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0  # type: ignore
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0  # type: ignore

        # Center should be close to (30.0, 50.0)
        self.assertAlmostEqual(new_cy, 50.0, delta=4.0)
        self.assertAlmostEqual(new_cx, 30.0, delta=4.0)

    def test_center_cell_value_preserves_borders(self):
        # Cell with dark borders on outer perimeter
        h, w = 80, 50
        cell = np.ones((h, w, 3), dtype=np.uint8) * 240
        # Draw 2px dark border
        cell[:2, :] = 20
        cell[-2:, :] = 20
        cell[:, :2] = 20
        cell[:, -2:] = 20

        # Draw off-center value near bottom
        cv2.putText(cell, "5", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

        centered = center_cell_value(cell)
        # Verify outer 2 pixels on all edges are completely unchanged
        np.testing.assert_array_equal(centered[:2, :], cell[:2, :])
        np.testing.assert_array_equal(centered[-2:, :], cell[-2:, :])
        np.testing.assert_array_equal(centered[:, :2], cell[:, :2])
        np.testing.assert_array_equal(centered[:, -2:], cell[:, -2:])

    def test_center_cell_value_empty_returns_unchanged(self):
        empty = np.ones((50, 50, 3), dtype=np.uint8) * 245
        centered = center_cell_value(empty)
        np.testing.assert_array_equal(centered, empty)

    def test_center_cell_value_pil_image(self):
        cell_arr = np.ones((60, 40, 3), dtype=np.uint8) * 240
        cv2.putText(
            cell_arr, "1", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2
        )
        pil_img = Image.fromarray(cell_arr)
        centered = center_cell_value(pil_img)
        self.assertIsInstance(centered, np.ndarray)
        self.assertEqual(centered.shape, (60, 40, 3))

    def test_reconstruct_table_from_cells_2d_matrix(self):
        # Create a 2x2 grid of cells
        c00_img = np.ones((40, 50, 3), dtype=np.uint8) * 200
        c01_img = np.ones((40, 60, 3), dtype=np.uint8) * 210
        c10_img = np.ones((50, 50, 3), dtype=np.uint8) * 220
        c11_img = np.ones((50, 60, 3), dtype=np.uint8) * 230

        cv2.putText(c00_img, "A", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(c01_img, "B", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(c10_img, "C", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(c11_img, "D", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        c00 = Cell(row=0, col=0, bbox=(0, 0, 50, 40), raw_crop=c00_img)
        c01 = Cell(row=0, col=1, bbox=(50, 0, 110, 40), raw_crop=c01_img)
        c10 = Cell(row=1, col=0, bbox=(0, 40, 50, 90), raw_crop=c10_img)
        c11 = Cell(row=1, col=1, bbox=(50, 40, 110, 90), raw_crop=c11_img)

        grid = GridGeometry(
            horizontal_lines=[0, 40, 90],
            vertical_lines=[0, 50, 110],
            table_width=110,
            table_height=90,
        )

        recon = reconstruct_table_from_cells([[c00, c01], [c10, c11]], grid=grid)
        self.assertEqual(recon.shape, (90, 110, 3))
        # Ensure centered_crop was assigned to each Cell
        self.assertIsNotNone(c00.centered_crop)
        self.assertIsNotNone(c11.centered_crop)

    def test_reconstruct_table_from_numpy_matrix(self):
        row0 = [
            np.ones((30, 40, 3), dtype=np.uint8) * 100,
            np.ones((30, 50, 3), dtype=np.uint8) * 150,
        ]
        row1 = [
            np.ones((40, 40, 3), dtype=np.uint8) * 200,
            np.ones((40, 50, 3), dtype=np.uint8) * 250,
        ]
        recon = reconstruct_table_from_blocks([row0, row1])
        self.assertEqual(recon.shape, (70, 90, 3))

    def test_save_table_segments_saves_reconstructed_table(self):
        dummy_table = np.ones((60, 80, 3), dtype=np.uint8) * 220
        c0 = Cell(
            row=0,
            col=0,
            bbox=(0, 0, 80, 30),
            raw_crop=np.ones((30, 80, 3), dtype=np.uint8) * 200,
        )
        c1 = Cell(
            row=1,
            col=0,
            bbox=(0, 30, 80, 60),
            raw_crop=np.ones((30, 80, 3), dtype=np.uint8) * 240,
        )
        out_dir = os.path.join(self.test_dir, "test_recon_save")

        saved = save_table_segments(
            dummy_table,
            [[c0], [c1]],
            out_dir,
            table_filename="table.png",
            reconstructed_table_filename="reconstructed_table.png",
        )

        self.assertIn("reconstructed_table", saved)
        recon_path = os.path.join(out_dir, "reconstructed_table.png")
        self.assertTrue(os.path.exists(recon_path))
        self.assertEqual(saved["reconstructed_table"], os.path.abspath(recon_path))
        recon_img = cv2.imread(recon_path)
        self.assertEqual(recon_img.shape, (60, 80, 3))

    def test_extract_table_segments_with_reconstructed_table(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "segments_recon_test")
        res = extract_table_segments(self.sample_img2, output_dir=out_dir)

        self.assertIsInstance(res, TableSegmentationResult)
        self.assertIsNotNone(res.reconstructed_table_image)
        self.assertEqual(res.reconstructed_table_image.shape, res.table_image.shape)

        # File checks
        recon_file = os.path.join(out_dir, "reconstructed_table.png")
        self.assertTrue(os.path.exists(recon_file))
        self.assertEqual(res.reconstructed_table_path, os.path.abspath(recon_file))
        self.assertEqual(res.centered_table_path, os.path.abspath(recon_file))

        # Check get_centered_cell and get_centered_cell_crop
        c00 = res.get_centered_cell(0, 0)
        self.assertIsNotNone(c00.centered_crop)
        crop00 = res.get_centered_cell_crop(0, 0)
        np.testing.assert_array_equal(crop00, c00.centered_crop)

        # Dictionary export includes reconstructed_table_path
        d = res.to_dict()
        self.assertIn("reconstructed_table_path", d)
        self.assertEqual(d["reconstructed_table_path"], os.path.abspath(recon_file))

    def test_sizespecextractor_reconstructed_table_methods(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        extractor = SizeSpecExtractor()
        out_img_path = os.path.join(self.test_dir, "standalone_recon.png")
        recon_img, _roi = extractor.extract_reconstructed_table(
            self.sample_img2, output_path=out_img_path
        )

        self.assertIsInstance(recon_img, np.ndarray)
        self.assertTrue(os.path.exists(out_img_path))
        saved = cv2.imread(out_img_path)
        np.testing.assert_array_equal(saved, recon_img)

        # Test alias extract_centered_table
        recon2, _roi2 = extractor.extract_centered_table(self.sample_img2)
        self.assertEqual(recon2.shape, recon_img.shape)

        # Test standalone function extract_reconstructed_table_image
        recon3, _roi3 = extract_reconstructed_table_image(self.sample_img2)
        self.assertEqual(recon3.shape, recon_img.shape)

    def test_cli_main_reconstructed_table_generation(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_recon_test")
        ret_code = main([self.sample_img2, "-o", out_folder])
        self.assertEqual(ret_code, 0)

        # Main table and cell crops exist
        self.assertTrue(os.path.exists(os.path.join(out_folder, "table.png")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[0][0].png")))
        # Reconstructed table image exists in the output folder
        self.assertTrue(
            os.path.exists(os.path.join(out_folder, "reconstructed_table.png"))
        )

    def test_cli_main_custom_reconstructed_filename(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_custom_recon")
        ret_code = main(
            [
                self.sample_img2,
                "-o",
                out_folder,
                "--reconstructed-table-filename",
                "my_centered_table.jpg",
            ]
        )
        self.assertEqual(ret_code, 0)
        self.assertTrue(
            os.path.exists(os.path.join(out_folder, "my_centered_table.jpg"))
        )

    def test_cli_main_output_reconstructed_image_flag(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_flag_folder")
        standalone_recon = os.path.join(self.test_dir, "standalone_recon_flag.png")
        ret_code = main(
            [
                self.sample_img2,
                "-o",
                out_folder,
                "--output-reconstructed-image",
                standalone_recon,
            ]
        )
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(standalone_recon))
        self.assertTrue(
            os.path.exists(os.path.join(out_folder, "reconstructed_table.png"))
        )

    def test_cli_main_slash_boost_argument(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_slash_boost")
        ret_code = main(["/boost", self.sample_img2, "-o", out_folder])
        self.assertEqual(ret_code, 0)
        self.assertTrue(
            os.path.exists(os.path.join(out_folder, "reconstructed_table.png"))
        )

    def test_cli_main_dash_boost_flag(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_dash_boost")
        ret_code = main([self.sample_img2, "--boost", "-o", out_folder])
        self.assertEqual(ret_code, 0)
        self.assertTrue(
            os.path.exists(os.path.join(out_folder, "reconstructed_table.png"))
        )

    def test_cli_main_no_boost_flag(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_no_boost")
        ret_code = main([self.sample_img2, "--no-boost", "-o", out_folder])
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(os.path.join(out_folder, "table.png")))
        self.assertFalse(
            os.path.exists(os.path.join(out_folder, "reconstructed_table.png"))
        )

    def test_cell_value_bbox_populated_in_segmentation(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        res = extract_table_segments(self.sample_img2)
        non_empty_bboxes = [
            res.get_cell(r, c).value_bbox
            for r in range(res.num_rows)
            for c in range(res.num_cols)
            if res.get_cell(r, c).value_bbox is not None
        ]
        self.assertGreater(len(non_empty_bboxes), 0)
        for bbox in non_empty_bboxes:
            self.assertEqual(len(bbox), 4)
            self.assertGreater(bbox[2], bbox[0])
            self.assertGreater(bbox[3], bbox[1])

    def test_bgra_and_float_handling(self):
        # 4-channel BGRA image
        bgra = np.ones((60, 50, 4), dtype=np.uint8) * 240
        cv2.putText(
            bgra, "8", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10, 255), 2
        )
        bbox = detect_cell_value(bgra)
        self.assertIsNotNone(bbox)
        centered_bgra = center_cell_value(bgra)
        self.assertEqual(centered_bgra.shape, (60, 50, 4))

        # float32 image
        flt = np.ones((60, 50, 3), dtype=np.float32)
        flt[10:20, 15:25] = 0.0
        flt_bbox = detect_cell_value(flt)
        self.assertIsNotNone(flt_bbox)
        centered_flt = center_cell_value(flt)
        self.assertEqual(centered_flt.shape[:2], (60, 50))

    def test_border_remnant_filtering(self):
        # 60x60 cell with 3px thick border line remnant at the top
        cell = np.ones((60, 60, 3), dtype=np.uint8) * 255
        cell[2:6, 5:55] = 0  # 4px thick horizontal line
        cv2.putText(cell, "7", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        bbox = detect_cell_value(cell)
        self.assertIsNotNone(bbox)
        # Bbox should belong to '7' (y > 10), not the top border remnant
        self.assertGreater(bbox[1], 8)  # type: ignore

    def test_skip_centering_row_zero_and_col_zero(self):
        # 1. Direct center_cell_value invocation with row=0 or col=0
        crop = np.ones((60, 40, 3), dtype=np.uint8) * 240
        cv2.putText(crop, "5", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Row 0 -> skipped
        res_row0 = center_cell_value(crop, row=0, col=3)
        np.testing.assert_array_equal(res_row0, crop)

        # Col 0 -> skipped
        res_col0 = center_cell_value(crop, row=2, col=0)
        np.testing.assert_array_equal(res_col0, crop)

        # Non-zero row and col -> centered (different from raw crop)
        res_data = center_cell_value(crop, row=1, col=1)
        self.assertFalse(np.array_equal(res_data, crop))

        # 2. In extract_table_segments on real sample:
        res = extract_table_segments("tests/image1.png")
        grid = res.grid

        # All row 0 blocks ([0][0] to [0][12]) must not be centered
        for c in range(grid.num_cols):
            cell_0_c = res.get_cell(0, c)
            np.testing.assert_array_equal(cell_0_c.centered_crop, cell_0_c.raw_crop)

        # All col 0 blocks ([n][0] for all n) must not be centered
        for r in range(grid.num_rows):
            cell_r_0 = res.get_cell(r, 0)
            np.testing.assert_array_equal(cell_r_0.centered_crop, cell_r_0.raw_crop)

        # Internal data block (e.g. row 1, col 3) MUST have been centered
        cell_1_3 = res.get_cell(1, 3)
        self.assertFalse(np.array_equal(cell_1_3.centered_crop, cell_1_3.raw_crop))


if __name__ == "__main__":
    unittest.main()
