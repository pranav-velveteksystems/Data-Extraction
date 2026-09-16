"""Unit tests for table reconstruction and validation layer."""

import unittest

import numpy as np

from size_spec_extractor.cells.cell_generator import Cell
from size_spec_extractor.config import ReconstructionConfig, ValidationConfig
from size_spec_extractor.reconstruction.schema import (
    SizeSpecRow,
    SizeSpecTable,
)
from size_spec_extractor.reconstruction.table import reconstruct_table
from size_spec_extractor.validation.validator import validate_table


class TestReconstructionAndValidation(unittest.TestCase):
    def test_reconstruct_table(self):
        # 3 rows, 4 columns:
        # Col 0: Spec, Col 1: 0-3M, Col 2: 3-6M, Col 3: 6-12M
        # Row 0 (header): ["Size", "0-3M", "3-6M", "6-12M"]
        # Row 1: ["Satin Straight Piece", "1½", "2", None]
        # Row 2: ["Net Straight Piece", "4", ["5", "6"], "7"]  # stacked values in col 2

        dummy_img = np.zeros((30, 30, 3), dtype=np.uint8)

        cells = []
        # Row 0
        r0 = [
            Cell(
                row=0,
                col=0,
                bbox=(0, 0, 100, 30),
                raw_crop=dummy_img,
                raw_texts=["Size"],
                is_empty=False,
            ),
            Cell(
                row=0,
                col=1,
                bbox=(100, 0, 200, 30),
                raw_crop=dummy_img,
                raw_texts=["0-3M"],
                is_empty=False,
            ),
            Cell(
                row=0,
                col=2,
                bbox=(200, 0, 300, 30),
                raw_crop=dummy_img,
                raw_texts=["3-6M"],
                is_empty=False,
            ),
            Cell(
                row=0,
                col=3,
                bbox=(300, 0, 400, 30),
                raw_crop=dummy_img,
                raw_texts=["6-12M"],
                is_empty=False,
            ),
        ]
        cells.append(r0)

        # Row 1
        r1 = [
            Cell(
                row=1,
                col=0,
                bbox=(0, 30, 100, 60),
                raw_crop=dummy_img,
                raw_texts=["Satin Straight Piece"],
                is_empty=False,
            ),
            Cell(
                row=1,
                col=1,
                bbox=(100, 30, 200, 60),
                raw_crop=dummy_img,
                raw_texts=["1½"],
                is_empty=False,
            ),
            Cell(
                row=1,
                col=2,
                bbox=(200, 30, 300, 60),
                raw_crop=dummy_img,
                raw_texts=["2"],
                is_empty=False,
            ),
            Cell(
                row=1, col=3, bbox=(300, 30, 400, 60), raw_crop=dummy_img, is_empty=True
            ),
        ]
        cells.append(r1)

        # Row 2
        r2 = [
            Cell(
                row=2,
                col=0,
                bbox=(0, 60, 100, 90),
                raw_crop=dummy_img,
                raw_texts=["Net Straight Piece"],
                is_empty=False,
            ),
            Cell(
                row=2,
                col=1,
                bbox=(100, 60, 200, 90),
                raw_crop=dummy_img,
                raw_texts=["4"],
                is_empty=False,
            ),
            Cell(
                row=2,
                col=2,
                bbox=(200, 60, 300, 90),
                raw_crop=dummy_img,
                raw_texts=["5", "6"],
                is_empty=False,
            ),
            Cell(
                row=2,
                col=3,
                bbox=(300, 60, 400, 90),
                raw_crop=dummy_img,
                raw_texts=["7"],
                is_empty=False,
            ),
        ]
        cells.append(r2)

        cfg = ReconstructionConfig(
            category="Designen Frock", style_code="IDF 134", unit="inch"
        )
        doc, metadata = reconstruct_table(cells, cfg)

        self.assertEqual(doc.category, "Designen Frock")
        self.assertEqual(doc.style_code, "IDF 134")

        tbl = doc.size_spec_table
        self.assertEqual(tbl.columns, ["0-3 M", "3-6 M", "6-12 M"])
        self.assertEqual(len(tbl.rows), 2)

        row1 = tbl.rows[0]
        self.assertEqual(row1.specification, "Satin Straight Piece")
        self.assertEqual(row1.values, ["1½", "2", None])
        self.assertEqual(row1.parsed_values, [1.5, 2.0, None])

        row2 = tbl.rows[1]
        self.assertEqual(row2.specification, "Net Straight Piece")
        self.assertEqual(row2.values, ["4", ["5", "6"], "7"])

        # Metadata should contain 12 cells
        self.assertEqual(len(metadata), 12)

    def test_validation_layer(self):
        # Consistent valid table
        tbl = SizeSpecTable(
            unit="inch",
            columns=["0-3 M", "3-6 M"],
            rows=[
                SizeSpecRow(specification="Chest", values=["2½", "3"]),
                SizeSpecRow(specification="Length", values=[None, "5"]),
            ],
        )
        val_res = validate_table(tbl, ValidationConfig())
        self.assertTrue(val_res.is_valid)
        self.assertEqual(len(val_res.errors), 0)

        # Inconsistent table (column mismatch)
        bad_tbl = SizeSpecTable(
            unit="inch",
            columns=["0-3 M", "3-6 M"],
            rows=[
                SizeSpecRow(
                    specification="Chest", values=["2½"]
                )  # only 1 val for 2 cols
            ],
        )
        bad_res = validate_table(bad_tbl, ValidationConfig())
        self.assertFalse(bad_res.is_valid)
        self.assertGreaterEqual(len(bad_res.errors), 1)


if __name__ == "__main__":
    unittest.main()
