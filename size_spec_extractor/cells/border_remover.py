"""Border removal and interior cell cropping."""

import numpy as np

from ..config import CellProcessingConfig


def remove_borders(cell_crop: np.ndarray, config: CellProcessingConfig) -> np.ndarray:
    """Crop cell interior by removing grid borders (Section 10).

    Takes margin ratios (e.g. 10% x, 8% y) and extracts the interior,
    preventing grid border lines from interfering with OCR.
    """
    h, w = cell_crop.shape[:2]
    if h <= 4 or w <= 4:
        return cell_crop.copy()

    margin_x = int(w * config.margin_ratio_x)
    margin_y = int(h * config.margin_ratio_y)

    # Ensure at least 1 pixel margin if cell is reasonably sized, but leave at least 4x4
    margin_x = min(margin_x, (w - 4) // 2)
    margin_y = min(margin_y, (h - 4) // 2)

    margin_x = max(0, margin_x)
    margin_y = max(0, margin_y)

    cropped = cell_crop[margin_y : h - margin_y, margin_x : w - margin_x].copy()
    return cropped
