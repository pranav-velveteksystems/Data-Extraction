"""Unit tests for fraction, size, and numeric corrections."""

import unittest

from size_spec_extractor.normalization.fractions import (
    normalize_fraction_string,
    parse_fraction,
)
from size_spec_extractor.normalization.sizes import (
    normalize_size_header,
    normalize_size_list,
)
from size_spec_extractor.normalization.values import correct_numeric_text


class TestNormalization(unittest.TestCase):
    def test_fractions(self):
        # Section 14: Preserve raw fraction and parse numeric value
        res_half = parse_fraction("2½")
        self.assertIsNotNone(res_half)
        self.assertEqual(res_half.raw, "2½")
        self.assertEqual(res_half.numeric_value, 2.5)

        res_quarter = parse_fraction("4¼")
        self.assertIsNotNone(res_quarter)
        self.assertEqual(res_quarter.raw, "4¼")
        self.assertEqual(res_quarter.numeric_value, 4.25)

        # ASCII and space-separated unicode fractions (Section 30)
        self.assertEqual(normalize_fraction_string("2 1/2"), "2½")
        self.assertEqual(normalize_fraction_string("21/2"), "2½")
        self.assertEqual(normalize_fraction_string("2½"), "2½")
        self.assertEqual(normalize_fraction_string("2 ½"), "2½")
        self.assertEqual(normalize_fraction_string("4 ½"), "4½")
        self.assertEqual(normalize_fraction_string("2.5"), "2½")
        self.assertEqual(normalize_fraction_string("2.25"), "2¼")
        self.assertEqual(normalize_fraction_string("2.75"), "2¾")
        self.assertEqual(normalize_fraction_string("1/2"), "½")

        parsed_ascii = parse_fraction("2 1/2")
        self.assertEqual(parsed_ascii.raw, "2½")
        self.assertEqual(parsed_ascii.numeric_value, 2.5)

        parsed_space = parse_fraction("2 ½")
        self.assertEqual(parsed_space.raw, "2½")
        self.assertEqual(parsed_space.numeric_value, 2.5)

        parsed_int = parse_fraction("3")
        self.assertEqual(parsed_int.raw, "3")
        self.assertEqual(parsed_int.numeric_value, 3.0)

    def test_size_headers(self):
        # Section 19: Size code formatting with space
        self.assertEqual(normalize_size_header("0-3M"), "0-3 M")
        self.assertEqual(normalize_size_header("3-6M"), "3-6 M")
        self.assertEqual(normalize_size_header("6-12M"), "6-12 M")
        self.assertEqual(normalize_size_header("1-2Y"), "1-2 Y")
        self.assertEqual(normalize_size_header("2-3Y"), "2-3 Y")
        self.assertEqual(normalize_size_header("6-7Y"), "6-7 Y")
        self.assertEqual(normalize_size_header("S"), "S")
        self.assertEqual(normalize_size_header("XL"), "XL")

        headers = ["0-3M", "3-6M", "1-2Y"]
        self.assertEqual(normalize_size_list(headers), ["0-3 M", "3-6 M", "1-2 Y"])

    def test_numeric_correction_dict(self):
        # Section 30: Correction dictionary
        # 'I' -> '1', 'l' -> '1', 'O' -> '0', 'S' -> '5'
        self.assertEqual(correct_numeric_text("I"), "1")
        self.assertEqual(correct_numeric_text("l"), "1")
        self.assertEqual(correct_numeric_text("O"), "0")
        self.assertEqual(correct_numeric_text("S"), "5")
        self.assertEqual(correct_numeric_text("2 1/2"), "2½")
        self.assertEqual(correct_numeric_text("21/2"), "2½")
        self.assertEqual(correct_numeric_text("2 ½"), "2½")
        # Punctuation/noise with no numeric chars should yield empty string
        self.assertEqual(correct_numeric_text(":"), "")
        self.assertEqual(correct_numeric_text("---"), "")


if __name__ == "__main__":
    unittest.main()
