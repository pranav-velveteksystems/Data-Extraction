"""Unit tests for top rectangle / metadata box extraction and attachment to reconstructed table."""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from size_spec_extractor.cells.cell_generator import Cell
from size_spec_extractor.config import ExtractorConfig, HeaderBoxConfig
from size_spec_extractor.detection.grid_detector import GridGeometry
from size_spec_extractor.detection.header_detector import (
    HeaderROI,
    attach_header_to_table,
    detect_header_box,
)
from size_spec_extractor.detection.table_detector import TableROI
from size_spec_extractor.extractor import (
    SizeSpecExtractor,
    TableSegmentationResult,
    extract_header_box_image,
    extract_reconstructed_table_image,
    save_table_segments,
)
from size_spec_extractor.main import main


class TestHeaderBoxExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_img1 = "tests/image1.png"
        self.sample_img2 = "tests/image2.jpeg"
        self.sample_img3 = "tests/image3.jpeg"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_attach_header_to_table_dimensions_and_aspect(self):
        # Header: width 400, height 100 (aspect ratio 4:1)
        header = np.full((100, 400, 3), 200, dtype=np.uint8)
        # Table: width 200, height 500
        table = np.full((500, 200, 3), 255, dtype=np.uint8)

        stacked = attach_header_to_table(table, header, divider_thickness=2)
        # Stacked width must match table width (200)
        self.assertEqual(stacked.shape[1], 200)
        # Header scaled to width 200 => height should be round(100 * 200 / 400) = 50
        # Total height: 50 (header) + 2 (divider) + 500 (table) = 552
        self.assertEqual(stacked.shape[0], 552)
        self.assertEqual(stacked.shape[2], 3)

    def test_attach_header_to_table_grayscale(self):
        header_gray = np.full((100, 200), 128, dtype=np.uint8)
        table_color = np.full((300, 200, 3), 255, dtype=np.uint8)

        stacked = attach_header_to_table(table_color, header_gray)
        self.assertEqual(len(stacked.shape), 3)
        self.assertEqual(stacked.shape[1], 200)
        self.assertEqual(stacked.shape[0], 100 + 300)

    def test_attach_header_to_table_invalid_inputs(self):
        table = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(attach_header_to_table(None, table))  # type: ignore
        self.assertIsNone(attach_header_to_table(None, None))  # type: ignore
        res = attach_header_to_table(table, None)  # type: ignore
        self.assertEqual(res.shape, table.shape)

    def test_detect_header_box_image2(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        img = cv2.imread(self.sample_img2)
        table_bbox = (81, 610, 550, 1383)

        # Mode: metadata
        cfg_meta = HeaderBoxConfig(mode="metadata")
        roi_meta = detect_header_box(img, table_bbox, cfg_meta)
        self.assertIsNotNone(roi_meta)
        self.assertEqual(roi_meta.box_type, "metadata")
        self.assertIsNotNone(roi_meta.image)
        # Verify box coordinates surround the metadata block
        x1, y1, x2, y2 = roi_meta.bbox
        self.assertGreater(x1, 250)
        self.assertLess(x1, 450)
        self.assertGreater(x2, 900)
        self.assertLess(x2, 1100)
        self.assertGreater(y1, 40)
        self.assertLess(y2, 200)

        # Mode: full
        cfg_full = HeaderBoxConfig(mode="full")
        roi_full = detect_header_box(img, table_bbox, cfg_full)
        self.assertIsNotNone(roi_full)
        self.assertEqual(roi_full.box_type, "full_header")
        fx1, _fy1, fx2, _fy2 = roi_full.bbox
        self.assertLess(fx1, 100)  # Starts from near the left border
        self.assertGreater(fx2, 900)

    def test_detect_header_box_image3(self):
        if not os.path.exists(self.sample_img3):
            self.skipTest(f"{self.sample_img3} not found")

        img = cv2.imread(self.sample_img3)
        table_bbox = (50, 550, 500, 1200)
        roi = detect_header_box(img, table_bbox, HeaderBoxConfig(mode="metadata"))
        self.assertIsNotNone(roi)
        x1, y1, x2, y2 = roi.bbox
        self.assertGreater(x1, 200)
        self.assertLess(x1, 450)
        self.assertGreater(x2, 850)
        self.assertGreater(y1, 40)
        self.assertLess(y2, 220)

    def test_detect_header_box_manual_bbox(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        manual = (50, 20, 350, 100)
        cfg = HeaderBoxConfig(mode="manual", manual_bbox=manual)
        roi = detect_header_box(img, (50, 200, 450, 450), cfg)
        self.assertIsNotNone(roi)
        self.assertEqual(roi.bbox, manual)
        self.assertEqual(roi.image.shape, (80, 300, 3))

    def test_detect_header_box_synthetic_sheet(self):
        # Synthetic sheet with top rectangle containing Category, Style Code, Name
        h, w = 800, 600
        sheet = np.full((h, w, 3), 255, dtype=np.uint8)
        # Draw outer header rectangle: x: 50..550, y: 30..130
        cv2.rectangle(sheet, (50, 30), (550, 130), (0, 0, 0), 2)
        # Vertical divider at x: 220
        cv2.line(sheet, (220, 30), (220, 130), (0, 0, 0), 2)
        # Horizontal divider inside right block at y: 80
        cv2.line(sheet, (220, 80), (550, 80), (0, 0, 0), 1)
        # Texts
        cv2.putText(
            sheet, "SPEC", (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        cv2.putText(
            sheet,
            "Category: Dress",
            (230, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            sheet,
            "Name: Summer",
            (230, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        # Table below at y: 300..700
        cv2.rectangle(sheet, (50, 300), (550, 700), (0, 0, 0), 2)

        roi = detect_header_box(
            sheet, (50, 300, 550, 700), HeaderBoxConfig(mode="metadata")
        )
        self.assertIsNotNone(roi)
        x1, y1, x2, y2 = roi.bbox
        # The detected metadata box should be near (220, 30, 550, 130)
        self.assertAlmostEqual(x1, 220, delta=10)
        self.assertAlmostEqual(x2, 550, delta=10)
        self.assertAlmostEqual(y1, 30, delta=10)
        self.assertAlmostEqual(y2, 130, delta=10)

    def test_save_table_segments_saves_header_and_stacked_table(self):
        # Create mock TableSegmentationResult
        table_img = np.full((300, 200, 3), 255, dtype=np.uint8)
        recon_img = np.full((300, 200, 3), 240, dtype=np.uint8)
        header_img = np.full((60, 200, 3), 200, dtype=np.uint8)
        table_roi = TableROI(bbox=(50, 100, 250, 400), image=table_img, confidence=1.0)
        grid = GridGeometry(
            horizontal_lines=[0, 300],
            vertical_lines=[0, 200],
            table_width=200,
            table_height=300,
        )
        header_roi = HeaderROI(
            bbox=(50, 10, 250, 70),
            image=header_img,
            box_type="metadata",
            confidence=1.0,
        )

        cell = Cell(row=0, col=0, bbox=(0, 0, 200, 300), raw_crop=table_img)

        res = TableSegmentationResult(
            table_image=table_img,
            cells=[[cell]],
            table_roi=table_roi,
            grid=grid,
            reconstructed_table_image=recon_img,
            header_box_image=header_img,
            header_box_bbox=header_roi.bbox,
            header_roi=header_roi,
        )

        out_dir = os.path.join(self.test_dir, "save_test")
        saved_paths = save_table_segments(
            table_img,
            [[cell]],
            out_dir,
            header_box_image=header_img,
        )

        # Check saved files
        self.assertIn("header_box", saved_paths)
        self.assertIn("metadata_box", saved_paths)
        self.assertIn("reconstructed_table", saved_paths)
        self.assertTrue(os.path.exists(saved_paths["header_box"]))
        self.assertTrue(os.path.exists(saved_paths["metadata_box"]))
        self.assertTrue(os.path.exists(saved_paths["reconstructed_table"]))

        # Verify stacked image on disk has header height + table height
        disk_recon = cv2.imread(saved_paths["reconstructed_table"])
        self.assertEqual(disk_recon.shape[1], 200)
        self.assertEqual(disk_recon.shape[0], 60 + 300)

        # Dict access verification
        self.assertIn("header_box", res)
        self.assertIn("metadata_box", res)
        self.assertIn("reconstructed_table_with_header", res)
        self.assertIsNotNone(res["header_box"])
        self.assertEqual(res.metadata_box_bbox, header_roi.bbox)

    def test_extract_header_box_image_standalone(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_file = os.path.join(self.test_dir, "header_standalone.png")
        header_crop = extract_header_box_image(
            self.sample_img2,
            output_path=out_file,
            table_bbox=(81, 610, 550, 1383),
        )
        self.assertIsNotNone(header_crop)
        self.assertTrue(os.path.exists(out_file))

    def test_cli_header_box_boost(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "cli_boost_test")
        ret = main(
            [
                self.sample_img2,
                "-o",
                out_dir,
                "--boost",
            ]
        )
        self.assertEqual(ret, 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "header_box.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "metadata_box.png")))
        self.assertTrue(
            os.path.exists(os.path.join(out_dir, "reconstructed_table.png"))
        )

        # Check reconstructed table height is greater than main table height due to stacking
        table_img = cv2.imread(os.path.join(out_dir, "table.png"))
        recon_img = cv2.imread(os.path.join(out_dir, "reconstructed_table.png"))
        self.assertGreater(recon_img.shape[0], table_img.shape[0])
        self.assertEqual(recon_img.shape[1], table_img.shape[1])

    def test_cli_no_header_box_flag(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "cli_no_header_test")
        ret = main(
            [
                self.sample_img2,
                "-o",
                out_dir,
                "--boost",
                "--no-header-box",
            ]
        )
        self.assertEqual(ret, 0)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "header_box.png")))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "metadata_box.png")))
        self.assertTrue(
            os.path.exists(os.path.join(out_dir, "reconstructed_table.png"))
        )

    def test_detect_header_box_image1(self):
        if not os.path.exists(self.sample_img1):
            self.skipTest(f"{self.sample_img1} not found")

        img = cv2.imread(self.sample_img1)
        roi = detect_header_box(
            img, (39, 1478, 1155, 2324), HeaderBoxConfig(mode="metadata")
        )
        self.assertIsNotNone(roi)
        x1, y1, x2, y2 = roi.bbox
        self.assertGreater(x1, 550)
        self.assertLess(x1, 700)
        self.assertGreater(x2, 2000)
        self.assertLessEqual(y1, 35)  # Encloses top line
        self.assertGreaterEqual(y2, 225)  # Encloses bottom line

    def test_detect_header_box_scan_page(self):
        scan_path = "tests/Scan_20260916_131828_page-0001.jpg"
        if not os.path.exists(scan_path):
            self.skipTest(f"{scan_path} not found")

        img = cv2.imread(scan_path)
        roi = detect_header_box(
            img, (24, 750, 661, 1711), HeaderBoxConfig(mode="metadata")
        )
        self.assertIsNotNone(roi)
        x1, y1, x2, y2 = roi.bbox
        self.assertGreater(x1, 300)
        self.assertLess(x1, 450)
        self.assertGreater(x2, 1150)
        self.assertLessEqual(y1, 50)
        self.assertGreaterEqual(y2, 125)

    def test_detect_header_box_table_at_top_returns_none(self):
        # When table begins near top (y <= 35), no header space exists
        img = (
            cv2.imread(self.sample_img2)
            if os.path.exists(self.sample_img2)
            else np.zeros((500, 500, 3), dtype=np.uint8)
        )
        roi = detect_header_box(img, table_bbox=(50, 20, 450, 450))
        self.assertIsNone(roi)
        roi0 = detect_header_box(img, table_bbox=(50, 0, 450, 450))
        self.assertIsNone(roi0)

    def test_detect_header_box_blank_image_returns_none(self):
        blank = np.full((600, 600, 3), 255, dtype=np.uint8)
        roi = detect_header_box(blank)
        self.assertIsNone(roi)

    def test_header_box_config_serialization(self):
        cfg = HeaderBoxConfig(
            enabled=True,
            mode="manual",
            manual_bbox=(10, 20, 300, 100),
            divider_thickness=3,
            divider_color=(50, 100, 150),
        )
        ext_cfg = ExtractorConfig(header_box=cfg)
        json_str = ext_cfg.to_json()
        loaded = ExtractorConfig.from_json(json_str)

        self.assertEqual(loaded.header_box.manual_bbox, (10, 20, 300, 100))
        self.assertEqual(loaded.header_box.divider_thickness, 3)
        self.assertEqual(loaded.header_box.divider_color, (50, 100, 150))
        self.assertEqual(loaded.header_box.mode, "manual")

    def test_attach_header_divider_customization(self):
        table = np.full((100, 200, 3), 255, dtype=np.uint8)
        header = np.full((50, 200, 3), 128, dtype=np.uint8)
        stacked = attach_header_to_table(
            table, header, divider_thickness=5, divider_color=(10, 20, 30)
        )
        self.assertIsNotNone(stacked)
        # Total height = 50 + 5 + 100 = 155
        self.assertEqual(stacked.shape, (155, 200, 3))
        # Divider rows at index 50..54 should have color (10, 20, 30)
        np.testing.assert_array_equal(stacked[50, 0], [10, 20, 30])
        np.testing.assert_array_equal(stacked[54, 100], [10, 20, 30])

    def test_extract_reconstructed_table_attach_header_flag(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        extractor = SizeSpecExtractor()
        # Default in-memory: unstacked
        unstacked, _ = extractor.extract_reconstructed_table(self.sample_img2)
        # With attach_header=True: stacked with top rectangle
        stacked, _ = extractor.extract_reconstructed_table(
            self.sample_img2, attach_header=True
        )
        self.assertGreater(stacked.shape[0], unstacked.shape[0])
        self.assertEqual(stacked.shape[1], unstacked.shape[1])

        # Standalone function
        standalone_stacked, _ = extract_reconstructed_table_image(
            self.sample_img2, attach_header=True
        )
        self.assertEqual(standalone_stacked.shape, stacked.shape)

    def test_cli_metadata_box_flag(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "cli_meta_flag_test")
        ret = main(
            [
                self.sample_img2,
                "-o",
                out_dir,
                "--header-box",
            ]
        )
        self.assertEqual(ret, 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "header_box.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "metadata_box.png")))


if __name__ == "__main__":
    unittest.main()
