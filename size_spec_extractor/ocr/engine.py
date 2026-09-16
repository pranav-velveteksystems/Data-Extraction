"""Base OCR engine interface and implementations."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import pytesseract

    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

HAS_EASYOCR = importlib.util.find_spec("easyocr") is not None
HAS_PADDLEOCR = importlib.util.find_spec("paddleocr") is not None


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0.0 to 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class BaseOCREngine(ABC):
    @abstractmethod
    def ocr(
        self, image: np.ndarray, whitelist: str | None = None, psm: int = 7
    ) -> OCRResult:
        """Run OCR on image crop."""


class TesseractOCREngine(BaseOCREngine):
    def __init__(self, tesseract_cmd: str | None = None):
        if not HAS_PYTESSERACT:
            raise ImportError("pytesseract is not installed")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def ocr(
        self, image: np.ndarray, whitelist: str | None = None, psm: int = 7
    ) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(text="", confidence=0.0)

        # Prepare config options
        config_parts = [f"--psm {psm}"]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        config_str = " ".join(config_parts)

        # Use image_to_data for word-level confidence
        try:
            data = pytesseract.image_to_data(
                image, config=config_str, output_type=pytesseract.Output.DICT
            )

            texts = []
            confs = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                t = data["text"][i].strip()
                c = float(data["conf"][i])
                if t:
                    texts.append(t)
                    if c >= 0:
                        confs.append(c / 100.0)

            joined_text = " ".join(texts)
            avg_conf = float(np.mean(confs)) if confs else (0.5 if joined_text else 0.0)
            return OCRResult(
                text=joined_text, confidence=avg_conf, details={"raw_data": data}
            )

        except Exception:
            # Fallback to image_to_string
            try:
                raw_text = pytesseract.image_to_string(image, config=config_str).strip()
                return OCRResult(text=raw_text, confidence=0.6 if raw_text else 0.0)
            except Exception as inner_e:
                return OCRResult(
                    text="", confidence=0.0, details={"error": str(inner_e)}
                )


class EasyOCREngine(BaseOCREngine):
    """EasyOCR backend for deep learning OCR."""

    def __init__(self, languages: list | None = None, gpu: bool = False):
        if not HAS_EASYOCR:
            raise ImportError("easyocr is not installed")
        import easyocr

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def ocr(
        self, image: np.ndarray, whitelist: str | None = None, psm: int = 7
    ) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(text="", confidence=0.0)

        kwargs = {}
        if whitelist:
            kwargs["allowlist"] = whitelist

        try:
            results = self.reader.readtext(image, **kwargs)
            if not results:
                return OCRResult(text="", confidence=0.0)
            texts = [r[1].strip() for r in results if r[1].strip()]
            confs = [float(r[2]) for r in results if r[1].strip()]
            joined = " ".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return OCRResult(text=joined, confidence=avg_conf)
        except Exception as e:
            return OCRResult(text="", confidence=0.0, details={"error": str(e)})


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR backend."""

    def __init__(self, lang: str = "en", **kwargs):
        if not HAS_PADDLEOCR:
            raise ImportError("paddleocr is not installed")
        from paddleocr import PaddleOCR

        self.ocr_model = PaddleOCR(lang=lang, **kwargs)

    def ocr(
        self, image: np.ndarray, whitelist: str | None = None, psm: int = 7
    ) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult(text="", confidence=0.0)

        try:
            results = self.ocr_model.ocr(image)
            if not results or not results[0]:
                return OCRResult(text="", confidence=0.0)
            texts = []
            confs = []
            for line in results[0]:
                txt = line[1][0]
                conf = float(line[1][1])
                if txt.strip():
                    texts.append(txt.strip())
                    confs.append(conf)
            joined = " ".join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return OCRResult(text=joined, confidence=avg_conf)
        except Exception as e:
            return OCRResult(text="", confidence=0.0, details={"error": str(e)})


class MockOCREngine(BaseOCREngine):
    """Mock OCR engine for deterministic unit testing."""

    def __init__(
        self, response_map: dict[str, str] | None = None, default_text: str = ""
    ):
        self.response_map = response_map or {}
        self.default_text = default_text

    def ocr(
        self, image: np.ndarray, whitelist: str | None = None, psm: int = 7
    ) -> OCRResult:
        h, w = image.shape[:2]
        key = f"{w}x{h}"
        text = self.response_map.get(key, self.default_text)
        conf = 0.95 if text else 0.0
        return OCRResult(text=text, confidence=conf)


def get_ocr_engine(engine_name: str = "tesseract", **kwargs) -> BaseOCREngine:
    """Factory to retrieve an OCR engine instance."""
    engine_name = engine_name.lower()
    if engine_name == "tesseract":
        return TesseractOCREngine(kwargs.get("tesseract_cmd"))
    elif engine_name == "easy":
        return EasyOCREngine(kwargs.get("languages"), kwargs.get("gpu", False))
    elif engine_name == "paddle":
        return PaddleOCREngine(kwargs.get("lang", "en"))
    elif engine_name == "mock":
        return MockOCREngine(kwargs.get("response_map"), kwargs.get("default_text", ""))
    else:
        raise ValueError(
            f"Unknown OCR engine: {engine_name}. Choose from 'tesseract', 'easy', 'paddle', 'mock'."
        )
