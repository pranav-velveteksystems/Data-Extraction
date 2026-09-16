"""Detection and clustering of horizontal and vertical table grid lines."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import LineDetectionConfig


def merge_adjacent_lines(
    line_positions: list[int],
    distance_threshold: int = 6,
    weights: list[float] | None = None,
) -> list[int]:
    """Cluster line coordinates that belong to the same physical thick line (Section 7).

    Example: [100, 101, 102, 103] -> [101] or [102]
    """
    if not line_positions:
        return []

    sorted_indices = np.argsort(line_positions)
    sorted_lines = [line_positions[i] for i in sorted_indices]
    if weights is not None:
        sorted_weights = [weights[i] for i in sorted_indices]
    else:
        sorted_weights = [1.0] * len(sorted_lines)

    clusters: list[list[tuple[int, float]]] = []
    current_cluster: list[tuple[int, float]] = [(sorted_lines[0], sorted_weights[0])]

    for i in range(1, len(sorted_lines)):
        pos = sorted_lines[i]
        weight = sorted_weights[i]
        if pos - current_cluster[-1][0] <= distance_threshold:
            current_cluster.append((pos, weight))
        else:
            clusters.append(current_cluster)
            current_cluster = [(pos, weight)]

    if current_cluster:
        clusters.append(current_cluster)

    merged: list[int] = []
    for cluster in clusters:
        total_weight = sum(w for _, w in cluster)
        if total_weight > 0:
            weighted_pos = sum(p * w for p, w in cluster) / total_weight
            merged.append(round(weighted_pos))
        else:
            merged.append(round(np.mean([p for p, _ in cluster])))

    return sorted(set(merged))


def find_projection_peaks(
    projection: np.ndarray, peak_threshold: float, min_distance: int = 6
) -> tuple[list[int], list[float]]:
    """Identify line positions from 1D projection profile peaks."""
    raw_positions: list[int] = []
    weights: list[float] = []

    for i, val in enumerate(projection):
        if val >= peak_threshold:
            raw_positions.append(i)
            weights.append(float(val))

    if not raw_positions:
        return [], []

    merged = merge_adjacent_lines(
        raw_positions, distance_threshold=min_distance, weights=weights
    )
    return merged, [1.0] * len(merged)


def detect_horizontal_lines(
    binary: np.ndarray, config: LineDetectionConfig
) -> list[int]:
    """Detect horizontal lines in binary image using morphological opening and projection."""
    _h, w = binary.shape[:2]
    kernel_len = max(config.h_kernel_length, int(w * config.h_kernel_ratio))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    projection = horizontal.sum(axis=1) / 255.0  # number of active white line pixels
    max_val = np.max(projection) if len(projection) > 0 else 0

    if max_val == 0:
        return []

    # Threshold must ensure a minimum physical line length relative to width
    # and adapt to strong lines in the table while rejecting short text/noise
    min_threshold = max(10.0, w * config.peak_threshold_ratio)
    threshold = max(min_threshold, max_val * 0.25)
    lines, _ = find_projection_peaks(
        projection, threshold, min_distance=config.line_merge_distance
    )
    return lines


def detect_vertical_lines(binary: np.ndarray, config: LineDetectionConfig) -> list[int]:
    """Detect vertical lines in binary image using morphological opening and projection."""
    h, _w = binary.shape[:2]
    kernel_len = max(config.v_kernel_length, int(h * config.v_kernel_ratio))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    projection = vertical.sum(axis=0) / 255.0  # number of active white line pixels
    max_val = np.max(projection) if len(projection) > 0 else 0

    if max_val == 0:
        return []

    # Threshold must ensure a minimum physical line length relative to height
    # and adapt to strong lines in the table while rejecting short text/noise
    min_threshold = max(10.0, h * config.peak_threshold_ratio)
    threshold = max(min_threshold, max_val * 0.25)
    lines, _ = find_projection_peaks(
        projection, threshold, min_distance=config.line_merge_distance
    )
    return lines
