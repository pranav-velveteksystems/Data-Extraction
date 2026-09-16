"""Ink detection, emptiness checking, and text region segmentation for stacked values."""

import cv2
import numpy as np

from ..config import CellProcessingConfig
from ..preprocessing.enhancement import to_grayscale
from ..preprocessing.threshold import binarize
from .cell_generator import TextRegion


def get_ink_mask(image: np.ndarray) -> np.ndarray:
    """Generate binary ink mask where ink pixels are 255 (white) and background is 0."""
    gray = to_grayscale(image)
    return binarize(gray, method="adaptive", invert=True)


def is_empty_cell(
    interior_crop: np.ndarray, min_ink_ratio: float = 0.005, min_component_area: int = 6
) -> bool:
    """Determine if a cell is empty before invoking OCR (Section 15).

    Calculates ink pixel ratio and verifies existence of significant connected components.
    """
    if (
        interior_crop.size == 0
        or interior_crop.shape[0] < 3
        or interior_crop.shape[1] < 3
    ):
        return True

    ink_mask = get_ink_mask(interior_crop)
    total_pixels = ink_mask.shape[0] * ink_mask.shape[1]
    ink_pixels = cv2.countNonZero(ink_mask)

    ratio = ink_pixels / float(total_pixels)
    if ratio < min_ink_ratio:
        return True

    # Connected component check to exclude isolated salt noise
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8
    )
    if num_labels <= 1:
        return True

    # Check if there is any component of meaningful size (ignoring background index 0)
    has_meaningful_ink = False
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            has_meaningful_ink = True
            break

    return not has_meaningful_ink


def upscale_region(image: np.ndarray, factor: int = 3) -> np.ndarray:
    """Upscale cell crop using bicubic interpolation (Section 26)."""
    if factor <= 1:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def detect_text_regions(
    interior_crop: np.ndarray, config: CellProcessingConfig
) -> list[TextRegion]:
    """Detect text regions in cell crop, handling vertically stacked values (Section 17 & 18).

    Returns:
        List of TextRegion objects sorted by Y position ascending (top to bottom).
    """
    if is_empty_cell(interior_crop, config.min_ink_ratio):
        return []

    h, w = interior_crop.shape[:2]
    ink_mask = get_ink_mask(interior_crop)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=8
    )
    if num_labels <= 1:
        return []

    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 5:  # filter noise speckles
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            valid_components.append((x, y, x + cw, y + ch, centroids[i][1]))

    if not valid_components:
        return []

    # Sort components by Y centroid
    valid_components.sort(key=lambda comp: comp[4])

    # Check if vertically stacked values exist (Section 17 & 18)
    regions: list[TextRegion] = []
    gap_threshold = max(2, config.stacked_gap_threshold)
    if config.detect_stacked_values and len(valid_components) >= 2 and h >= 25:
        # Check horizontal projection for vertical gap in the middle region
        v_profile = ink_mask.sum(axis=1) / 255.0
        mid_start = int(h * 0.25)
        mid_end = int(h * 0.75)

        # Look for a valley where ink density drops between top and bottom components
        max_proj = np.max(v_profile) if len(v_profile) > 0 else 1.0
        min_valley_val = float("inf")
        split_y = -1
        for y in range(mid_start, mid_end):
            val = v_profile[y]
            if val < min_valley_val:
                min_valley_val = val
                split_y = y

        top_comps = [comp for comp in valid_components if comp[4] < split_y]
        bot_comps = [comp for comp in valid_components if comp[4] >= split_y]

        # Calculate actual gap between top and bottom components
        if top_comps and bot_comps:
            top_bottom_edge = max(comp[3] for comp in top_comps)
            bot_top_edge = min(comp[1] for comp in bot_comps)
            comp_gap = bot_top_edge - top_bottom_edge
        else:
            comp_gap = 0

        # Split condition: either clear component gap >= threshold or pronounced profile dip
        is_split = (comp_gap >= gap_threshold) or (
            min_valley_val <= max(2.0, max_proj * 0.20)
            and bool(top_comps)
            and bool(bot_comps)
        )

        if split_y != -1 and is_split:
            # Top region
            top_crop = interior_crop[0:split_y, :].copy()
            if not is_empty_cell(top_crop, config.min_ink_ratio):
                upscaled_top = upscale_region(top_crop, config.upscale_factor)
                regions.append(
                    TextRegion(
                        crop=upscaled_top,
                        bbox=(0, 0, w, split_y),
                        y_position=split_y // 2,
                        position_label="top",
                    )
                )

            # Bottom region
            bottom_crop = interior_crop[split_y:h, :].copy()
            if not is_empty_cell(bottom_crop, config.min_ink_ratio):
                upscaled_bottom = upscale_region(bottom_crop, config.upscale_factor)
                regions.append(
                    TextRegion(
                        crop=upscaled_bottom,
                        bbox=(0, split_y, w, h),
                        y_position=split_y + (h - split_y) // 2,
                        position_label="bottom",
                    )
                )

    # If not split into stacked values, use entire interior as single region
    if not regions:
        upscaled = upscale_region(interior_crop, config.upscale_factor)
        regions.append(
            TextRegion(
                crop=upscaled,
                bbox=(0, 0, w, h),
                y_position=h // 2,
                position_label="single",
            )
        )

    # Sort strictly by Y position
    regions.sort(key=lambda r: r.y_position)
    return regions
