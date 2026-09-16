"""Unit and integration tests for size spec table image extraction and saving."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from size_spec_extractor.extractor import (
    SizeSpecExtractor,
    TableSegmentationResult,
    extract_table_image,
    extract_table_segments,
    save_table_image,
    save_table_segments,
)
from size_spec_extractor.main import main


class TestTableImageExtraction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_img1 = "tests/image1.png"
        self.sample_img2 = "tests/image2.jpeg"
        self.sample_img3 = "tests/image3.jpeg"

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists("image2_output"):
            shutil.rmtree("image2_output")

    def test_extract_table_image_from_filepath(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        table_img, roi = extract_table_image(self.sample_img2)
        self.assertIsInstance(table_img, np.ndarray)
        self.assertGreater(table_img.shape[0], 100)
        self.assertGreater(table_img.shape[1], 100)
        self.assertEqual(roi.bbox, (81, 610, 550, 1383))
        self.assertEqual(roi.width, 469)
        self.assertEqual(roi.height, 773)

    def test_extract_table_image_with_output_path(self):
        if not os.path.exists(self.sample_img3):
            self.skipTest(f"{self.sample_img3} not found")

        out_path = os.path.join(self.test_dir, "extracted_spec_table.png")
        table_img, _roi = extract_table_image(self.sample_img3, output_path=out_path)

        self.assertTrue(os.path.exists(out_path))
        saved_img = cv2.imread(out_path)
        self.assertIsNotNone(saved_img)
        self.assertEqual(saved_img.shape, table_img.shape)
        np.testing.assert_array_equal(saved_img, table_img)

    def test_extract_table_image_from_numpy_array(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        raw_bgr = cv2.imread(self.sample_img2)
        table_img, _roi = extract_table_image(raw_bgr)
        self.assertIsInstance(table_img, np.ndarray)
        self.assertEqual(table_img.shape, (773, 469, 3))

    def test_extract_table_image_from_pil_image(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        pil_img = Image.open(self.sample_img2)
        table_img, _roi = extract_table_image(pil_img)
        self.assertIsInstance(table_img, np.ndarray)
        self.assertEqual(table_img.shape, (773, 469, 3))

    def test_save_table_image_creates_parent_directories(self):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        nested_out = os.path.join(self.test_dir, "sub1", "sub2", "table.jpg")

        saved_path = save_table_image(dummy_img, nested_out)
        self.assertTrue(os.path.exists(nested_out))
        self.assertEqual(saved_path, os.path.abspath(nested_out))

    def test_save_table_image_validation(self):
        with self.assertRaises(ValueError):
            save_table_image(None, "out.png")  # type: ignore

        with self.assertRaises(ValueError):
            save_table_image(np.array([]), "out.png")

    def test_sizespecextractor_methods(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        extractor = SizeSpecExtractor()
        out_path = os.path.join(self.test_dir, "extractor_out.png")
        _table_img, roi = extractor.extract_table_image(
            self.sample_img2, output_path=out_path
        )

        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(roi.bbox, (81, 610, 550, 1383))

        # Test extract_table alias
        _table_img2, roi2 = extractor.extract_table(self.sample_img2)
        self.assertEqual(roi2.bbox, roi.bbox)

    def test_cli_main_positional_and_output(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_path = os.path.join(self.test_dir, "cli_output.png")
        ret_code = main([self.sample_img2, "-o", out_path])
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(out_path))

    def test_cli_main_flag_input(self):
        if not os.path.exists(self.sample_img3):
            self.skipTest(f"{self.sample_img3} not found")

        out_path = os.path.join(self.test_dir, "cli_flag_output.png")
        ret_code = main(["-i", self.sample_img3, "--output", out_path])
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(out_path))

    def test_cli_main_default_output(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        expected_default = "image2_table.png"
        if os.path.exists(expected_default):
            os.remove(expected_default)

        try:
            ret_code = main([self.sample_img2])
            self.assertEqual(ret_code, 0)
            self.assertTrue(os.path.exists(expected_default))
        finally:
            if os.path.exists(expected_default):
                os.remove(expected_default)

    def test_cli_main_non_existent_file(self):
        ret_code = main(["non_existent_file_path.jpg"])
        self.assertEqual(ret_code, 1)

    def test_cli_main_manual_bbox(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_path = os.path.join(self.test_dir, "manual_bbox.png")
        ret_code = main(
            [
                self.sample_img2,
                "-o",
                out_path,
                "--manual-bbox",
                "100",
                "200",
                "300",
                "400",
            ]
        )
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(out_path))
        saved = cv2.imread(out_path)
        # width = 300 - 100 = 200, height = 400 - 200 = 200
        self.assertEqual(saved.shape[1], 200)
        self.assertEqual(saved.shape[0], 200)

    def test_cli_main_no_perspective(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_path = os.path.join(self.test_dir, "no_perspective.png")
        ret_code = main([self.sample_img2, "-o", out_path, "--no-perspective"])
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(out_path))

    def test_extract_table_image_sample1(self):
        if not os.path.exists(self.sample_img1):
            self.skipTest(f"{self.sample_img1} not found")

        table_img, roi = extract_table_image(self.sample_img1)
        self.assertIsInstance(table_img, np.ndarray)
        # Verify specification column (x1 <= 50) and all size columns are preserved
        self.assertLessEqual(roi.x1, 50)
        self.assertGreaterEqual(roi.x2, 1140)
        self.assertGreaterEqual(roi.width, 1050)
        self.assertGreaterEqual(roi.height, 800)
        self.assertEqual(table_img.shape[0], roi.height)
        self.assertEqual(table_img.shape[1], roi.width)

    def test_extract_table_image_path_object(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_path = Path(self.test_dir) / "from_path.png"
        table_img, roi = extract_table_image(
            Path(self.sample_img2), output_path=out_path
        )
        self.assertTrue(out_path.exists())
        self.assertEqual(roi.width, 469)
        self.assertEqual(table_img.shape[1], 469)

    def test_load_image_corrupt_file(self):
        corrupt_path = os.path.join(self.test_dir, "corrupt.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"not an image file content")
        with self.assertRaises(ValueError):
            extract_table_image(corrupt_path)

    def test_save_table_segments(self):
        table_img = np.ones((100, 150, 3), dtype=np.uint8) * 200
        # Create a dummy 2x3 matrix of cells
        from size_spec_extractor.cells.cell_generator import Cell

        cells = [
            [
                Cell(
                    row=0,
                    col=0,
                    bbox=(0, 0, 50, 50),
                    raw_crop=np.zeros((50, 50, 3), dtype=np.uint8),
                ),
                Cell(
                    row=0,
                    col=1,
                    bbox=(50, 0, 100, 50),
                    raw_crop=np.zeros((50, 50, 3), dtype=np.uint8),
                ),
                Cell(
                    row=0,
                    col=2,
                    bbox=(100, 0, 150, 50),
                    raw_crop=np.zeros((50, 50, 3), dtype=np.uint8),
                ),
            ],
            [
                Cell(
                    row=1,
                    col=0,
                    bbox=(0, 50, 50, 100),
                    raw_crop=np.zeros((50, 50, 3), dtype=np.uint8),
                ),
                Cell(
                    row=1,
                    col=1,
                    bbox=(50, 50, 100, 100),
                    raw_crop=np.zeros((50, 50, 3), dtype=np.uint8),
                ),
                Cell(
                    row=1,
                    col=2,
                    bbox=(100, 50, 150, 100),
                    raw_crop=np.zeros((50, 50, 3), dtype=np.uint8),
                ),
            ],
        ]
        out_dir = os.path.join(self.test_dir, "test_segments")
        saved = save_table_segments(table_img, cells, out_dir)

        # Check main table image
        table_file = os.path.join(out_dir, "table.png")
        self.assertTrue(os.path.exists(table_file))
        self.assertEqual(saved["table"], os.path.abspath(table_file))

        # Check all 6 cells
        for r in range(2):
            for c in range(3):
                cell_name = f"[{r}][{c}].png"
                cell_path = os.path.join(out_dir, cell_name)
                self.assertTrue(os.path.exists(cell_path), f"Missing {cell_name}")
                self.assertEqual(saved[f"[{r}][{c}]"], os.path.abspath(cell_path))
                img = cv2.imread(cell_path)
                self.assertEqual(img.shape, (50, 50, 3))

    def test_save_table_segments_invalid_image(self):
        with self.assertRaises(ValueError):
            save_table_segments(None, [], self.test_dir)  # type: ignore
        with self.assertRaises(ValueError):
            save_table_segments(np.array([]), [], self.test_dir)

    def test_extract_table_segments_from_filepath(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "img2_segments")
        res = extract_table_segments(self.sample_img2, output_dir=out_dir)

        self.assertIsInstance(res, TableSegmentationResult)
        self.assertEqual(res.num_rows, 9)
        self.assertEqual(res.num_cols, 13)
        self.assertEqual(res.total_cells, 117)
        self.assertEqual(res.output_dir, os.path.abspath(out_dir))

        # Test tuple unpacking
        table_img, cells, roi, grid = res
        self.assertEqual(table_img.shape, (773, 469, 3))
        self.assertEqual(len(cells), 9)
        self.assertEqual(len(cells[0]), 13)
        self.assertEqual(roi.bbox, res.table_roi.bbox)
        self.assertEqual(grid.num_rows, 9)

        # Check files on disk
        table_path = os.path.join(out_dir, "table.png")
        self.assertTrue(os.path.exists(table_path))
        self.assertEqual(res.table_image_path, os.path.abspath(table_path))

        for r in range(9):
            for c in range(13):
                cell_file = os.path.join(out_dir, f"[{r}][{c}].png")
                self.assertTrue(
                    os.path.exists(cell_file), f"Missing cell [{r}][{c}].png"
                )
                self.assertEqual(
                    res.get_cell_image_path(r, c), os.path.abspath(cell_file)
                )

    def test_extract_table_segments_sample1(self):
        if not os.path.exists(self.sample_img1):
            self.skipTest(f"{self.sample_img1} not found")

        out_dir = os.path.join(self.test_dir, "img1_segments")
        res = extract_table_segments(self.sample_img1, output_dir=out_dir)

        self.assertIsInstance(res, TableSegmentationResult)
        # All 4 physical rows (header + 3 data rows) must be separated into individual cells
        self.assertEqual(res.num_rows, 4)
        self.assertEqual(res.num_cols, 13)
        self.assertEqual(res.total_cells, 52)

        # Verify cell [2][0] and [3][0] are separate individual boxes
        cell_2_0 = res.get_cell(2, 0)
        cell_3_0 = res.get_cell(3, 0)
        self.assertLess(cell_2_0.raw_crop.shape[0], 300)
        self.assertLess(cell_3_0.raw_crop.shape[0], 300)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "[2][0].png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "[3][0].png")))

    def test_extract_table_segments_from_numpy_and_pil(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        # From numpy
        bgr = cv2.imread(self.sample_img2)
        res_np = extract_table_segments(bgr)
        self.assertEqual(res_np.num_rows, 9)
        self.assertEqual(res_np.num_cols, 13)

        # From PIL
        pil_img = Image.open(self.sample_img2)
        res_pil = extract_table_segments(pil_img)
        self.assertEqual(res_pil.num_rows, 9)
        self.assertEqual(res_pil.num_cols, 13)

    def test_sizespecextractor_extract_segments(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        extractor = SizeSpecExtractor()
        out_dir = os.path.join(self.test_dir, "extractor_segments")
        res = extractor.extract_segments(self.sample_img2, output_dir=out_dir)

        self.assertEqual(res.num_rows, 9)
        self.assertEqual(res.num_cols, 13)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "table.png")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "[0][0].png")))

        # Test alias extract_table_and_cells
        res_alias = extractor.extract_table_and_cells(self.sample_img2)
        self.assertEqual(res_alias.num_rows, 9)
        self.assertEqual(res_alias.num_cols, 13)

    def test_cli_main_folder_output(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "my_custom_folder")
        ret_code = main([self.sample_img2, "-o", out_folder])
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.isdir(out_folder))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "table.png")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[0][0].png")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[8][12].png")))

    def test_cli_main_output_dir_flag(self):
        if not os.path.exists(self.sample_img3):
            self.skipTest(f"{self.sample_img3} not found")

        out_folder = os.path.join(self.test_dir, "dir_flag_folder")
        ret_code = main(["-i", self.sample_img3, "--output-dir", out_folder])
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.isdir(out_folder))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "table.png")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[0][0].png")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[7][12].png")))

    def test_cli_main_default_creates_folder_and_table_png(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        expected_file = "image2_table.png"
        expected_folder = "image2_output"
        if os.path.exists(expected_file):
            os.remove(expected_file)
        if os.path.exists(expected_folder):
            shutil.rmtree(expected_folder)

        try:
            ret_code = main([self.sample_img2])
            self.assertEqual(ret_code, 0)
            self.assertTrue(os.path.exists(expected_file))
            self.assertTrue(os.path.isdir(expected_folder))
            self.assertTrue(os.path.exists(os.path.join(expected_folder, "table.png")))
            self.assertTrue(os.path.exists(os.path.join(expected_folder, "[0][0].png")))
            self.assertTrue(
                os.path.exists(os.path.join(expected_folder, "[8][12].png"))
            )
        finally:
            if os.path.exists(expected_file):
                os.remove(expected_file)
            if os.path.exists(expected_folder):
                shutil.rmtree(expected_folder)

    def test_segment_cell_dimensions_match_grid(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "dimension_check")
        res = extract_table_segments(self.sample_img2, output_dir=out_dir)

        for r in range(res.num_rows):
            for c in range(res.num_cols):
                cell_crop = res.get_cell(r, c).raw_crop
                file_path = os.path.join(out_dir, f"[{r}][{c}].png")
                disk_crop = cv2.imread(file_path)
                self.assertIsNotNone(disk_crop)
                self.assertEqual(disk_crop.shape, cell_crop.shape)
                np.testing.assert_array_equal(disk_crop, cell_crop)

    def test_table_segmentation_result_dunder_methods_and_dict(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_dir = os.path.join(self.test_dir, "dunder_test")
        res = extract_table_segments(self.sample_img2, output_dir=out_dir)

        # __len__
        self.assertEqual(len(res), 4)

        # sequence __getitem__
        self.assertIs(res[0], res.table_image)
        self.assertIs(res[1], res.cells)
        self.assertIs(res[2], res.table_roi)
        self.assertIs(res[3], res.grid)
        self.assertIs(res[-1], res.grid)
        self.assertEqual(len(res[0:2]), 2)

        # coordinate __getitem__
        c00 = res[0, 0]
        self.assertEqual(c00.row, 0)
        self.assertEqual(c00.col, 0)

        # string key __getitem__
        self.assertIs(res["table"], res.table_image)
        self.assertIs(res["table_image"], res.table_image)
        self.assertIs(res["cells"], res.cells)
        self.assertIs(res["roi"], res.table_roi)
        self.assertIs(res["grid"], res.grid)
        self.assertEqual(res["output_dir"], os.path.abspath(out_dir))
        self.assertIsNotNone(res["table_image_path"])
        self.assertEqual(res["[0][0]"].row, 0)

        with self.assertRaises(KeyError):
            _ = res["non_existent_key"]

        # __contains__
        self.assertIn("table", res)
        self.assertIn("grid", res)
        self.assertIn((0, 0), res)
        self.assertIn((8, 12), res)
        self.assertNotIn((100, 100), res)

        # get_cell_crop and get_cell_image
        crop1 = res.get_cell_crop(0, 0)
        crop2 = res.get_cell_image(0, 0)
        self.assertIsInstance(crop1, np.ndarray)
        np.testing.assert_array_equal(crop1, crop2)

        # to_dict
        d = res.to_dict()
        self.assertIn("output_dir", d)
        self.assertIn("table_image_path", d)
        self.assertEqual(d["num_rows"], 9)
        self.assertEqual(d["num_cols"], 13)
        self.assertEqual(d["total_cells"], 117)
        self.assertIn("[0][0]", d["cell_image_paths"])

    def test_save_table_segments_empty_string_output_dir(self):
        from size_spec_extractor.cells.cell_generator import Cell

        table_img = np.ones((50, 50, 3), dtype=np.uint8)
        cell = Cell(row=0, col=0, bbox=(0, 0, 50, 50), raw_crop=table_img)

        cur_dir = os.getcwd()
        try:
            os.chdir(self.test_dir)
            # Empty string should resolve to current working directory without error
            saved = save_table_segments(table_img, [[cell]], "")
            self.assertTrue(os.path.exists("table.png"))
            self.assertTrue(os.path.exists("[0][0].png"))
            self.assertIn("table", saved)
            self.assertIn("[0][0]", saved)
        finally:
            os.chdir(cur_dir)

    def test_save_table_segments_pil_image_inputs(self):
        from size_spec_extractor.cells.cell_generator import Cell

        pil_tbl = Image.new("RGB", (60, 40), color=(100, 150, 200))
        pil_cell_crop = Image.new("RGB", (30, 20), color=(50, 60, 70))
        cell = Cell(row=0, col=0, bbox=(0, 0, 30, 20), raw_crop=pil_cell_crop)  # type: ignore

        out_dir = os.path.join(self.test_dir, "pil_inputs")
        saved = save_table_segments(pil_tbl, [[cell]], out_dir)
        self.assertTrue(os.path.exists(saved["table"]))
        self.assertTrue(os.path.exists(saved["[0][0]"]))

    def test_save_table_segments_flat_cell_list(self):
        from size_spec_extractor.cells.cell_generator import Cell

        table_img = np.ones((60, 60, 3), dtype=np.uint8)
        c0 = Cell(
            row=0,
            col=0,
            bbox=(0, 0, 30, 30),
            raw_crop=np.zeros((30, 30, 3), dtype=np.uint8),
        )
        c1 = Cell(
            row=0,
            col=1,
            bbox=(30, 0, 60, 30),
            raw_crop=np.zeros((30, 30, 3), dtype=np.uint8),
        )
        flat_cells = [c0, c1]

        out_dir = os.path.join(self.test_dir, "flat_cells")
        saved = save_table_segments(table_img, flat_cells, out_dir)
        self.assertTrue(os.path.exists(saved["[0][0]"]))
        self.assertTrue(os.path.exists(saved["[0][1]"]))

    def test_save_table_segments_custom_cell_format(self):
        from size_spec_extractor.cells.cell_generator import Cell

        table_img = np.ones((40, 40, 3), dtype=np.uint8)
        cell = Cell(row=0, col=0, bbox=(0, 0, 40, 40), raw_crop=table_img)

        out_dir = os.path.join(self.test_dir, "custom_fmt")
        saved = save_table_segments(
            table_img, [[cell]], out_dir, table_filename="main.jpg", cell_format="jpg"
        )
        self.assertTrue(os.path.exists(os.path.join(out_dir, "main.jpg")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "[0][0].jpg")))
        self.assertEqual(
            saved["[0][0]"], os.path.abspath(os.path.join(out_dir, "[0][0].jpg"))
        )

    def test_cli_custom_table_filename_and_cell_format(self):
        if not os.path.exists(self.sample_img2):
            self.skipTest(f"{self.sample_img2} not found")

        out_folder = os.path.join(self.test_dir, "cli_custom_fmt")
        ret_code = main(
            [
                self.sample_img2,
                "-o",
                out_folder,
                "--table-filename",
                "custom_table.jpg",
                "--cell-format",
                "jpg",
            ]
        )
        self.assertEqual(ret_code, 0)
        self.assertTrue(os.path.exists(os.path.join(out_folder, "custom_table.jpg")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[0][0].jpg")))
        self.assertTrue(os.path.exists(os.path.join(out_folder, "[8][12].jpg")))


if __name__ == "__main__":
    unittest.main()
