#!/usr/bin/env python3
"""Read a segmented table folder and OCR each cell into a JSON table.

Usage:
    python ocr_table.py <folder>                     # default: tesseract
    python ocr_table.py <folder> -o result.json      # save to file
    python ocr_table.py <folder> --engine easy        # use EasyOCR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

from size_spec_extractor.normalization.fractions import normalize_fraction_string
from size_spec_extractor.ocr.engine import get_ocr_engine


def discover_cells(folder: str) -> list[tuple[int, int, str]]:
    """Find all [r][c].ext files and return sorted list of (row, col, filepath)."""
    pattern = re.compile(r"^\[(\d+)\]\[(\d+)\]\.\w+$")
    cells: list[tuple[int, int, str]] = []
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            cells.append((r, c, os.path.join(folder, fname)))
    cells.sort(key=lambda x: (x[0], x[1]))
    return cells


def preprocess_cell(img: np.ndarray, is_header: bool = False) -> np.ndarray:
    """Preprocess a cell image for better OCR accuracy."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]

    # Rotate tall narrow cells (likely vertical header text)
    if is_header and h > w * 1.3:
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = gray.shape[:2]

    # Upscale small images
    min_dim = min(h, w)
    if min_dim < 80:
        scale = max(3, 80 // min_dim)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Slight denoise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive threshold for clean binarization
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Inset borders to remove grid lines
    margin_y = max(2, int(h * 0.05))
    margin_x = max(2, int(w * 0.05))
    bh, bw = binary.shape[:2]
    if margin_y * 2 < bh and margin_x * 2 < bw:
        binary = binary[margin_y : bh - margin_y, margin_x : bw - margin_x]

    return binary


def ocr_cell(engine, img: np.ndarray, is_header: bool = False) -> tuple[str, float]:
    """OCR a single cell with preprocessing, return (text, confidence)."""
    processed = preprocess_cell(img, is_header=is_header)

    if is_header:
        # Try both rotated and unrotated for header cells
        result1 = engine.ocr(processed, psm=6)
        # Also try with the original (non-rotated) preprocessing
        processed_no_rot = preprocess_cell(img, is_header=False)
        result2 = engine.ocr(processed_no_rot, psm=6)
        # Pick the one with higher confidence or more alphanumeric chars
        t1 = result1.text.strip()
        t2 = result2.text.strip()
        alpha1 = sum(1 for ch in t1 if ch.isalnum())
        alpha2 = sum(1 for ch in t2 if ch.isalnum())
        if alpha1 >= alpha2 and result1.confidence >= result2.confidence * 0.8:
            text, conf = t1, result1.confidence
        else:
            text, conf = t2, result2.confidence
    else:
        # Measurement cell: try with whitelist first, then without
        result_wl = engine.ocr(processed, whitelist="0123456789½¼¾.-/ ", psm=7)
        result_no = engine.ocr(processed, psm=7)
        t_wl = result_wl.text.strip()
        t_no = result_no.text.strip()
        # Prefer whitelisted if it found digits; otherwise fallback
        digits_wl = sum(1 for ch in t_wl if ch.isdigit() or ch in "½¼¾")
        digits_no = sum(1 for ch in t_no if ch.isdigit() or ch in "½¼¾")
        if digits_wl >= digits_no:
            text, conf = t_wl, result_wl.confidence
        else:
            text, conf = t_no, result_no.confidence

    return text, conf


def ocr_folder(
    folder: str,
    engine_name: str = "tesseract",
    normalize: bool = True,
) -> dict:
    """OCR all cell images in a folder and return structured JSON dict."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    engine = get_ocr_engine(engine_name)
    cells = discover_cells(folder)

    if not cells:
        raise ValueError(f"No cell images ([r][c].ext) found in: {folder}")

    max_row = max(r for r, _, _ in cells)
    max_col = max(c for _, c, _ in cells)

    # Build 2D matrix initialized with nulls
    matrix: list[list[str | None]] = [
        [None for _ in range(max_col + 1)] for _ in range(max_row + 1)
    ]
    confidences: list[list[float | None]] = [
        [None for _ in range(max_col + 1)] for _ in range(max_row + 1)
    ]

    for r, c, filepath in cells:
        img = cv2.imread(filepath)
        if img is None:
            continue

        # Row 0 = header row; Col 0 = spec label column
        is_header = r == 0 or c == 0
        text, conf = ocr_cell(engine, img, is_header=is_header)

        # Normalize fractions (e.g. "2 1/2" -> "2½")
        if normalize and text:
            text = normalize_fraction_string(text)

        # Clean noise: if a measurement cell has no digits/fractions, treat as null
        if not is_header and text and not re.search(r"[0-9½¼¾]", text):
            text = ""

        matrix[r][c] = text if text else None
        confidences[r][c] = round(conf, 3)

    # Build structured output
    headers = matrix[0] if matrix else []
    rows = []
    for r in range(1, max_row + 1):
        spec_label = matrix[r][0]
        values = matrix[r][1:]
        rows.append(
            {
                "specification": spec_label,
                "values": values,
            }
        )

    output = {
        "grid_size": {"rows": max_row + 1, "columns": max_col + 1},
        "columns": headers[1:] if headers else [],
        "rows": rows,
        "raw_matrix": matrix,
        "confidences": confidences,
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OCR a segmented table folder into JSON."
    )
    parser.add_argument(
        "folder", help="Path to folder containing [r][c].png cell images"
    )
    parser.add_argument("-o", "--output", help="Path to save JSON output file")
    parser.add_argument(
        "--engine",
        default="tesseract",
        choices=["tesseract", "easy", "paddle"],
        help="OCR engine (default: tesseract)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip fraction normalization",
    )
    args = parser.parse_args(argv)

    try:
        result = ocr_folder(
            args.folder,
            engine_name=args.engine,
            normalize=not args.no_normalize,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        return 1

    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[+] JSON saved to: {os.path.abspath(args.output)}")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
