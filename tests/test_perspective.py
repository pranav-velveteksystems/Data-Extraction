"""Unit tests for perspective correction and point ordering."""

import unittest

import cv2
import numpy as np

from size_spec_extractor.preprocessing.perspective import (
    correct_perspective,
    four_point_transform,
    order_points,
)


class TestPerspective(unittest.TestCase):
    def test_order_points(self):
        # Coordinates for rectangle: (50, 50), (200, 50), (200, 300), (50, 300)
        pts = np.array([[200, 300], [50, 50], [200, 50], [50, 300]], dtype="float32")
        rect = order_points(pts)

        np.testing.assert_allclose(rect[0], [50, 50])  # Top-left
        np.testing.assert_allclose(rect[1], [200, 50])  # Top-right
        np.testing.assert_allclose(rect[2], [200, 300])  # Bottom-right
        np.testing.assert_allclose(rect[3], [50, 300])  # Bottom-left

    def test_four_point_transform(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 350), (255, 255, 255), -1)

        pts = np.array([[50, 50], [250, 50], [250, 350], [50, 350]], dtype="float32")
        warped = four_point_transform(img, pts)

        self.assertEqual(warped.shape[0], 300)
        self.assertEqual(warped.shape[1], 200)

    def test_correct_perspective_fallback(self):
        # Blank image with no document contour should safely return original image with corrected=False
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        res, corrected, corners = correct_perspective(blank)
        self.assertFalse(corrected)
        self.assertIsNone(corners)
        self.assertEqual(res.shape, blank.shape)


if __name__ == "__main__":
    unittest.main()
