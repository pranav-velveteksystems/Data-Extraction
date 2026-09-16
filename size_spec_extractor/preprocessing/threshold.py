"""Binarization and thresholding methods."""

from typing import Literal

import cv2
import numpy as np


def adaptive_threshold(
    gray: np.ndarray, block_size: int = 31, c: int = 11, invert: bool = True
) -> np.ndarray:
    """Apply adaptive Gaussian thresholding.

    If invert=True, foreground ink/lines are 255 (white) on 0 (black) background.
    """
    if block_size % 2 == 0:
        block_size += 1

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, c
    )


def otsu_threshold(gray: np.ndarray, invert: bool = True) -> np.ndarray:
    """Apply Otsu's binarization."""
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    return thresh


def binarize(
    gray: np.ndarray,
    method: Literal["adaptive", "otsu"] = "adaptive",
    block_size: int = 31,
    c: int = 11,
    invert: bool = True,
) -> np.ndarray:
    """Binarize grayscale image using specified method."""
    if method == "otsu":
        return otsu_threshold(gray, invert=invert)
    return adaptive_threshold(gray, block_size=block_size, c=c, invert=invert)
