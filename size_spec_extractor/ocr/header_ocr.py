"""OCR for table headers and row specification labels."""

import re

import cv2
import numpy as np

from ..config import OCRConfig
from ..normalization.sizes import normalize_size_header
from ..preprocessing.enhancement import apply_clahe, to_grayscale
from .engine import BaseOCREngine, OCRResult


def score_size_text(text: str) -> float:
    """Score candidate size text: higher score for standard size patterns (e.g. 0-3M, 1-2Y)."""
    if not text:
        return -1.0
    norm = normalize_size_header(text)
    if re.search(r"\d+[-–]\d+\s*[MY]", norm) or re.search(r"\d+\s*[MY]", norm):
        return 10.0 + len(text)
    if re.search(r"\d+[-–]\d+", norm):
        return 5.0 + len(text)
    return float(len(text))


class HeaderOCR:
    """General OCR for size column headers and specification text (Section 13, 19)."""

    def __init__(self, engine: BaseOCREngine, config: OCRConfig):
        self.engine = engine
        self.config = config

    def ocr_header(self, image: np.ndarray) -> OCRResult:
        """OCR for column headers (e.g. 0-3M, 3-6M, 1-2Y), handling rotated text in narrow columns."""
        gray = to_grayscale(image)
        enhanced = apply_clahe(gray, clip_limit=2.0)
        h, w = enhanced.shape[:2]

        size_wl = getattr(self.config, "whitelist_size_header", "0123456789MYXSL- /")

        # Standard unrotated OCR
        res_normal = self.engine.ocr(
            enhanced, whitelist=self.config.whitelist_header, psm=self.config.header_psm
        )

        # If column is tall and narrow (e.g. height > width * 1.15), test 90-degree clockwise rotation
        if h > w * 1.15:
            rot_90 = cv2.rotate(enhanced, cv2.ROTATE_90_CLOCKWISE)
            res_rot = self.engine.ocr(rot_90, whitelist=size_wl, psm=6)
            # Evaluate candidates based on pattern score and confidence
            score_rot = score_size_text(res_rot.text)
            score_norm = score_size_text(res_normal.text)

            if (
                score_rot > score_norm
                or score_rot == score_norm
                and res_rot.confidence > res_normal.confidence
            ):
                return res_rot

        return res_normal

    def ocr_spec_label(self, image: np.ndarray) -> OCRResult:
        """OCR for row specification names (e.g. 'Satin Straight Piece', 'Chest')."""
        gray = to_grayscale(image)
        enhanced = apply_clahe(gray, clip_limit=2.0)
        # For general text labels, use general alphanumeric without strict whitelist restrictions
        return self.engine.ocr(
            enhanced,
            whitelist=None,
            psm=6,  # Assume a single uniform block of text
        )
