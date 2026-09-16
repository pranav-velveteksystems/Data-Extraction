"""Unit tests for JSON, HTML outputs, and visual debugging."""

import json
import os
import tempfile
import unittest

import numpy as np

from size_spec_extractor.cells.cell_generator import Cell
from size_spec_extractor.detection.grid_detector import GridGeometry
from size_spec_extractor.output.html import generate_html_table, save_html
from size_spec_extractor.output.json import save_json, to_json
from size_spec_extractor.reconstruction.schema import (
    CellCoordinateMetadata,
    ExtractionResult,
    GarmentSpecDocument,
    SizeSpecRow,
    SizeSpecTable,
)
from size_spec_extractor.visualization.debug import VisualDebugger


class TestOutputAndDebug(unittest.TestCase):
    def setUp(self):
        self.doc = GarmentSpecDocument(
            category="Designen Frock",
            style_code="IDF 134",
            name="Sample",
            size_spec_table=SizeSpecTable(
                unit="inch",
                columns=["0-3 M", "3-6 M"],
                rows=[
                    SizeSpecRow(
                        specification="Satin Straight Piece", values=["1½", "2"]
                    )
                ],
            ),
        )
        self.metadata = [
            CellCoordinateMetadata(
                row=0,
                column=0,
                bbox=[0, 0, 50, 20],
                ocr={"text": "0-3 M", "confidence": 0.95},
            )
        ]
        self.result = ExtractionResult(document=self.doc, metadata=self.metadata)

    def test_json_output(self):
        json_str = to_json(self.result, include_metadata=True)
        data = json.loads(json_str)
        self.assertEqual(data["style_code"], "IDF 134")
        self.assertEqual(data["size_spec_table"]["columns"], ["0-3 M", "3-6 M"])
        self.assertIn("_debug_coordinates", data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            save_json(self.result, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["category"], "Designen Frock")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_html_output(self):
        html_str = generate_html_table(self.result)
        self.assertIn("Designen Frock", html_str)
        self.assertIn("IDF 134", html_str)
        self.assertIn("Satin Straight Piece", html_str)
        self.assertIn("1½", html_str)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            tmp_path = f.name

        try:
            save_html(self.result, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("<table>", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_visual_debugger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deb = VisualDebugger(tmpdir)
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            deb.stage_01_original(img)
            deb.stage_02_warped(img)
            deb.stage_03_table_detection(img, (10, 10, 80, 80))
            deb.stage_04_horizontal_lines(img, [20, 50, 80])
            deb.stage_05_vertical_lines(img, [25, 75])
            deb.stage_06_grid(img, GridGeometry([0, 50, 100], [0, 50, 100], 100, 100))

            dummy_cells = [
                [
                    Cell(
                        row=0,
                        col=0,
                        bbox=(0, 0, 50, 50),
                        raw_crop=img[:50, :50],
                        raw_texts=["test"],
                        confidence=0.9,
                    )
                ]
            ]
            deb.stage_07_cells(img, dummy_cells)
            deb.stage_08_ocr_results(img, dummy_cells)

            files = os.listdir(tmpdir)
            self.assertIn("01_original.jpg", files)
            self.assertIn("02_warped.jpg", files)
            self.assertIn("03_table_detection.jpg", files)
            self.assertIn("04_horizontal_lines.jpg", files)
            self.assertIn("05_vertical_lines.jpg", files)
            self.assertIn("06_grid.jpg", files)
            self.assertIn("07_cells.jpg", files)
            self.assertIn("08_ocr_results.jpg", files)


if __name__ == "__main__":
    unittest.main()
