"""End-to-end pipeline integration tests."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from size_spec_extractor.config import ExtractorConfig
from size_spec_extractor.extractor import SizeSpecExtractor


def create_synthetic_garment_sheet() -> np.ndarray:
    """Generate a clean synthetic garment specification sheet with table and headers."""
    h, w = 600, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Document header
    cv2.putText(
        img,
        "Category: Designen Frock",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        img, "Style: IDF 134", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )

    # Table bounds: y from 120 to 420, x from 30 to 470
    # Rows: 3 rows (y = 120, 220, 320, 420)
    # Cols: 4 cols (x = 30, 150, 250, 360, 470)
    y_lines = [120, 220, 320, 420]
    x_lines = [30, 150, 250, 360, 470]

    for y in y_lines:
        cv2.line(img, (x_lines[0], y), (x_lines[-1], y), (0, 0, 0), 2)

    for x in x_lines:
        cv2.line(img, (x, y_lines[0]), (x, y_lines[-1]), (0, 0, 0), 2)

    # Header Row (Row 0)
    cv2.putText(img, "Spec", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "0-3M", (170, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "3-6M", (275, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "6-12M", (380, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Row 1: Satin
    cv2.putText(img, "Satin", (40, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "2", (185, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "2 1/2", (280, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # (col 3 empty)

    # Row 2: Net
    cv2.putText(img, "Net", (40, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "4", (185, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "5", (295, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "6", (405, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img


class TestPipeline(unittest.TestCase):
    def test_pipeline_with_synthetic_sheet(self):
        sheet_img = create_synthetic_garment_sheet()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractorConfig(
                debug=True, debug_dir=tmpdir, include_coordinates=True
            )
            config.table_detection.mode = "manual"
            config.table_detection.manual_bbox = (25, 115, 475, 425)

            extractor = SizeSpecExtractor(config=config)
            result = extractor.extract(sheet_img)

            self.assertIsNotNone(result)
            self.assertIsNotNone(result.document)
            tbl = result.document.size_spec_table

            # Verify columns
            self.assertEqual(len(tbl.columns), 3)
            self.assertEqual(tbl.columns[0], "0-3 M")
            self.assertEqual(tbl.columns[1], "3-6 M")
            self.assertEqual(tbl.columns[2], "6-12 M")

            # Verify rows
            self.assertEqual(len(tbl.rows), 2)
            self.assertIn("Satin", tbl.rows[0].specification)
            self.assertIn("Net", tbl.rows[1].specification)

            # Check debug artifacts generated
            debug_files = os.listdir(tmpdir)
            self.assertIn("01_original.jpg", debug_files)
            self.assertIn("06_grid.jpg", debug_files)
            self.assertIn("08_ocr_results.jpg", debug_files)

    def test_real_sample_if_available(self):
        sample_path = "/home/user/.gemini/antigravity-cli/brain/2382878e-eb64-4d69-9ecb-6953857629a8/cropped_sheet.png"
        if os.path.exists(sample_path):
            config = ExtractorConfig(debug=False)
            config.ocr.multi_pass = False
            extractor = SizeSpecExtractor(config=config)
            result = extractor.extract(sample_path)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.document)
            self.assertGreater(len(result.document.size_spec_table.columns), 0)

    def test_sample_image1(self):
        path = "tests/image1.png"
        if os.path.exists(path):
            config = ExtractorConfig(debug=False)
            config.ocr.multi_pass = False
            extractor = SizeSpecExtractor(config=config)
            result = extractor.extract(path)
            self.assertIsNotNone(result)
            self.assertGreater(len(result.document.size_spec_table.columns), 0)
            self.assertGreater(len(result.document.size_spec_table.rows), 0)

    def test_pipeline_auto_mode_synthetic(self):
        """Test pipeline table extraction in full auto detection mode (no manual bbox)."""
        sheet_img = create_synthetic_garment_sheet()
        config = ExtractorConfig(debug=False)
        extractor = SizeSpecExtractor(config=config)
        result = extractor.extract(sheet_img)

        self.assertIsNotNone(result)
        tbl = result.document.size_spec_table
        self.assertEqual(len(tbl.columns), 3)
        self.assertEqual(tbl.columns[0], "0-3 M")
        self.assertEqual(tbl.columns[1], "3-6 M")
        self.assertEqual(tbl.columns[2], "6-12 M")
        self.assertEqual(len(tbl.rows), 2)
        self.assertEqual(tbl.rows[0].values, ["2", "2½", None])
        self.assertEqual(tbl.rows[1].values, ["4", "5", "6"])

    def test_sample_image2_and_image3_e2e(self):
        """Test full end-to-end extraction on real garment spec sheets image2 and image3."""
        for img_name in ["image2.jpeg", "image3.jpeg"]:
            path = f"tests/{img_name}"
            if not os.path.exists(path):
                continue
            config = ExtractorConfig(debug=False)
            config.ocr.multi_pass = False
            extractor = SizeSpecExtractor(config=config)
            result = extractor.extract(path)

            self.assertIsNotNone(result)
            tbl = result.document.size_spec_table
            # Must detect size columns and specification rows
            self.assertGreaterEqual(
                len(tbl.columns), 10, f"Expected >= 10 columns on {img_name}"
            )
            self.assertGreaterEqual(
                len(tbl.rows), 5, f"Expected >= 5 rows on {img_name}"
            )

            # Verify no empty strings in values (must be None or valid content per Section 12 & 21)
            for r_idx, row in enumerate(tbl.rows):
                self.assertEqual(len(row.values), len(tbl.columns))
                for c_idx, val in enumerate(row.values):
                    self.assertNotEqual(
                        val,
                        "",
                        f"Empty string found at row {r_idx} col {c_idx} in {img_name}",
                    )
                    if isinstance(val, list):
                        for sub_val in val:
                            self.assertNotEqual(
                                sub_val,
                                "",
                                f"Empty sub-val in stacked cell row {r_idx} col {c_idx} in {img_name}",
                            )

    def test_easyocr_engine(self):
        """Test EasyOCREngine instantiation and recognition."""
        from size_spec_extractor.ocr.engine import get_ocr_engine

        try:
            engine = get_ocr_engine("easy")
            test_img = np.ones((40, 80, 3), dtype=np.uint8) * 255
            cv2.putText(
                test_img, "2 1/2", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )
            res = engine.ocr(test_img)
            self.assertIsNotNone(res)
            self.assertIn("2", res.text)
        except ImportError:
            self.skipTest("EasyOCR not installed")


if __name__ == "__main__":
    unittest.main()
