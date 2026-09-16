"""Specialized numeric measurement OCR with multi-pass evaluation."""

from collections import Counter

import cv2
import numpy as np

from ..config import OCRConfig
from ..preprocessing.enhancement import apply_clahe, to_grayscale
from ..preprocessing.threshold import adaptive_threshold, otsu_threshold
from .engine import BaseOCREngine, OCRResult


def sharpen_image(gray: np.ndarray) -> np.ndarray:
    """Sharpen grayscale image using unsharp mask."""
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    return cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)


class NumericOCR:
    """Specialized OCR for numeric size measurements (Section 13, 24, 25)."""

    def __init__(self, engine: BaseOCREngine, config: OCRConfig):
        self.engine = engine
        self.config = config

    def generate_variants(self, image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Generate preprocessing image variants for multi-pass OCR (Section 25)."""
        gray = to_grayscale(image)
        enhanced = apply_clahe(gray, clip_limit=2.0)

        variants = [
            ("enhanced_gray", enhanced),
            ("adaptive_thresh", adaptive_threshold(enhanced, invert=False)),
            ("otsu_thresh", otsu_threshold(enhanced, invert=False)),
            ("sharpened", sharpen_image(enhanced)),
        ]
        return variants

    def ocr_single_region(self, region_crop: np.ndarray) -> OCRResult:
        """Run multi-pass or single-pass OCR on a numeric region crop."""
        if not self.config.multi_pass:
            gray = to_grayscale(region_crop)
            return self.engine.ocr(
                gray,
                whitelist=self.config.whitelist_numeric,
                psm=self.config.numeric_psm,
            )

        variants = self.generate_variants(region_crop)
        results: list[OCRResult] = []

        for name, variant_img in variants:
            res = self.engine.ocr(
                variant_img,
                whitelist=self.config.whitelist_numeric,
                psm=self.config.numeric_psm,
            )
            cleaned = res.text.strip()
            if cleaned:
                results.append(
                    OCRResult(
                        text=cleaned,
                        confidence=res.confidence,
                        details={"variant": name},
                    )
                )

        if not results:
            return OCRResult(text="", confidence=0.0)

        # Majority voting (Section 25)
        text_counts = Counter(r.text for r in results)
        majority_text, count = text_counts.most_common(1)[0]

        # Get average confidence for the majority text
        matching_confs = [r.confidence for r in results if r.text == majority_text]
        avg_conf = float(np.mean(matching_confs))

        # Boost confidence slightly if multiple variants agree
        if count >= 2:
            avg_conf = min(1.0, avg_conf * 1.1)

        return OCRResult(
            text=majority_text, confidence=avg_conf, details={"passes": len(results)}
        )

    def classify_status(self, confidence: float) -> str:
        """Classify acceptance status based on confidence (Section 24)."""
        if confidence >= self.config.high_confidence:
            return "accepted"
        elif confidence >= self.config.min_confidence:
            return "review"
        return "needs_review"
