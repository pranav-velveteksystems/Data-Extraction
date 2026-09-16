"""Table detection and Region of Interest (ROI) extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import TableDetectionConfig
from ..preprocessing.enhancement import to_grayscale
from ..preprocessing.threshold import binarize


@dataclass
class TableROI:
    image: np.ndarray
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 1.0

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def snap_bbox_to_lines(
    image: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Snap bounding box coordinates to the outermost detected table lines."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return bbox

    sub = image[y1:y2, x1:x2]
    sh, sw = sub.shape[:2]
    gray = to_grayscale(sub)
    binary = binarize(gray, method="adaptive", invert=True)

    h_klen = max(15, sw // 30)
    v_klen = max(15, sh // 30)
    h_lines_img = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_klen, 1))
    )
    v_lines_img = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_klen))
    )

    h_proj = h_lines_img.sum(axis=1) / 255.0
    v_proj = v_lines_img.sum(axis=0) / 255.0

    h_thresh = max(
        sw * 0.15,
        np.max(h_proj) * 0.25 if len(h_proj) > 0 and np.max(h_proj) > 0 else 1,
    )
    v_thresh = max(
        sh * 0.15,
        np.max(v_proj) * 0.25 if len(v_proj) > 0 and np.max(v_proj) > 0 else 1,
    )

    h_idx = np.where(h_proj >= h_thresh)[0]
    v_idx = np.where(v_proj >= v_thresh)[0]

    if len(h_idx) >= 2 and len(v_idx) >= 2:
        snapped_x1 = max(0, x1 + int(v_idx[0]))
        snapped_y1 = max(0, y1 + int(h_idx[0]))
        snapped_x2 = min(w, x1 + int(v_idx[-1]))
        snapped_y2 = min(h, y1 + int(h_idx[-1]))
        if (snapped_x2 - snapped_x1) >= sw * 0.5 and (
            snapped_y2 - snapped_y1
        ) >= sh * 0.5:
            return (snapped_x1, snapped_y1, snapped_x2, snapped_y2)
    return (x1, y1, x2, y2)


def detect_table_by_grid_density(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
    relative_roi: tuple[float, float, float, float] = (0.38, 0.01, 0.98, 0.555),
) -> tuple[int, int, int, int] | None:
    """Detect table bounding box by finding dense intersecting grid lines."""
    gray = to_grayscale(image)
    h, w = gray.shape[:2]
    total_area = h * w

    binary = binarize(gray, method="adaptive", invert=True)

    # Detect horizontal line segments
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical line segments
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Table grid mask is the combination of horizontal and vertical lines
    table_mask = cv2.bitwise_or(h_lines, v_lines)

    # Dilate to connect grid intersections
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(table_mask, connect_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter by area and aspect ratio
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * total_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Table must have reasonable dimensions (at least 15% width and height of sheet)
        if cw > w * 0.15 and ch > h * 0.15:
            is_full_page = (cw > w * 0.75 and ch > h * 0.75) or (
                area > 0.65 * total_area
            )
            candidates.append((area, is_full_page, (x, y, x + cw, y + ch)))

    # If standalone table exists that is not a full-page outer frame
    non_full_candidates = [c for c in candidates if not c[1]]
    if non_full_candidates:
        non_full_candidates.sort(key=lambda item: item[0], reverse=True)
        raw_bbox = non_full_candidates[0][2]
        return snap_bbox_to_lines(image, raw_bbox)

    # If all candidates are full-page outer document frames (e.g. multi-section garment sheet),
    # locate the size spec table using the relative template region (Section 5) and snap to lines
    ymin_r, xmin_r, ymax_r, xmax_r = relative_roi
    rel_bbox = (int(xmin_r * w), int(ymin_r * h), int(xmax_r * w), int(ymax_r * h))
    return snap_bbox_to_lines(image, rel_bbox)


def detect_table(
    image: np.ndarray, config: TableDetectionConfig | None = None
) -> TableROI:
    """Detect and crop the size specification table from the document."""
    if config is None:
        config = TableDetectionConfig()

    h, w = image.shape[:2]

    # Mode: manual bbox
    if config.mode == "manual" and config.manual_bbox is not None:
        x1, y1, x2, y2 = config.manual_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2].copy()
        return TableROI(image=crop, bbox=(x1, y1, x2, y2), confidence=1.0)

    # Mode: auto (grid density detection)
    if config.mode == "auto":
        bbox = detect_table_by_grid_density(
            image, config.min_table_area_ratio, config.relative_roi
        )
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2].copy()
            return TableROI(image=crop, bbox=(x1, y1, x2, y2), confidence=0.90)

    # Fallback or Mode: relative ROI
    # relative_roi format: (ymin_ratio, xmin_ratio, ymax_ratio, xmax_ratio)
    ymin_r, xmin_r, ymax_r, xmax_r = config.relative_roi
    x1 = int(xmin_r * w)
    y1 = int(ymin_r * h)
    x2 = int(xmax_r * w)
    y2 = int(ymax_r * h)

    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))

    snapped = snap_bbox_to_lines(image, (x1, y1, x2, y2))
    x1, y1, x2, y2 = snapped

    crop = image[y1:y2, x1:x2].copy()
    return TableROI(image=crop, bbox=(x1, y1, x2, y2), confidence=0.80)
