"""Unit tests for enhancement and thresholding."""

import unittest

import numpy as np

from size_spec_extractor.preprocessing.enhancement import (
    apply_clahe,
    denoise,
    preprocess_image,
    to_grayscale,
)
from size_spec_extractor.preprocessing.threshold import (
    adaptive_threshold,
    binarize,
    otsu_threshold,
)


class TestEnhancementAndThreshold(unittest.TestCase):
    def setUp(self):
        # 100x100 3-channel test image
        self.color_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    def test_to_grayscale(self):
        res1 = to_grayscale(self.color_img)
        self.assertEqual(len(res1.shape), 2)
        self.assertEqual(res1.shape, (100, 100))

        res2 = to_grayscale(self.gray_img)
        self.assertEqual(res2.shape, (100, 100))

    def test_clahe_and_denoise(self):
        clahe_out = apply_clahe(self.gray_img, clip_limit=2.0)
        self.assertEqual(clahe_out.shape, (100, 100))
        self.assertEqual(clahe_out.dtype, np.uint8)

        denoised_out = denoise(clahe_out, strength=5)
        self.assertEqual(denoised_out.shape, (100, 100))

    def test_preprocess_image(self):
        result = preprocess_image(self.color_img)
        self.assertIn("original", result)
        self.assertIn("gray", result)
        self.assertIn("enhanced", result)
        self.assertIn("denoised", result)

    def test_thresholding(self):
        adapt = adaptive_threshold(self.gray_img, block_size=15, c=5, invert=True)
        self.assertEqual(adapt.shape, (100, 100))
        self.assertTrue(set(np.unique(adapt)).issubset({0, 255}))

        otsu = otsu_threshold(self.gray_img, invert=True)
        self.assertEqual(otsu.shape, (100, 100))
        self.assertTrue(set(np.unique(otsu)).issubset({0, 255}))

        bin_out = binarize(self.gray_img, method="adaptive")
        self.assertEqual(bin_out.shape, (100, 100))


if __name__ == "__main__":
    unittest.main()
