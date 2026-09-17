"""Detection and centering of internal values within cell blocks, and table reconstruction."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..config import CellProcessingConfig
from ..detection.grid_detector import GridGeometry
from .cell_generator import Cell


def _to_numpy_image(image: np.ndarray | Image.Image) -> np.ndarray:
    """Ensure image is a numpy BGR or grayscale array in uint8."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                max_val = float(np.max(arr)) if arr.size > 0 else 1.0
                scale = 255.0 if max_val <= 1.01 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def detect_cell_value(
    cell_crop: np.ndarray | Image.Image,
    config: CellProcessingConfig | None = None,
    margin_ratio_x: float = 0.08,
    margin_ratio_y: float = 0.06,
    min_component_area: int = 6,
) -> tuple[int, int, int, int] | None:
    """Detect bounding box (x1, y1, x2, y2) of internal value inside a cell block.

    Args:
        cell_crop: Cell crop image as numpy array or PIL Image.
        config: Optional CellProcessingConfig for margins.
        margin_ratio_x: Fraction of cell width to inset from left/right to exclude borders.
        margin_ratio_y: Fraction of cell height to inset from top/bottom to exclude borders.
        min_component_area: Minimum connected component area in pixels to ignore speckles.

    Returns:
        tuple (x1, y1, x2, y2) bounding box relative to cell_crop, or None if cell is empty.
    """
    img = _to_numpy_image(cell_crop)
    h, w = img.shape[:2]
    if h < 6 or w < 6:
        return None

    if config is not None:
        margin_ratio_x = getattr(config, "margin_ratio_x", margin_ratio_x)
        margin_ratio_y = getattr(config, "margin_ratio_y", margin_ratio_y)

    margin_x = max(1, int(w * margin_ratio_x))
    margin_y = max(1, int(h * margin_ratio_y))
    margin_x = min(margin_x, (w - 4) // 2)
    margin_y = min(margin_y, (h - 4) // 2)

    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0].copy()
    else:
        gray = img.copy()

    inner_gray = gray[margin_y : h - margin_y, margin_x : w - margin_x]
    if inner_gray.size == 0 or inner_gray.shape[0] < 3 or inner_gray.shape[1] < 3:
        return None

    bg_median = float(np.median(inner_gray))
    is_inverted = bool(bg_median < 127)

    if is_inverted:
        ink_binary = (inner_gray > (bg_median + 18)).astype(np.uint8) * 255
    else:
        ink_binary = (inner_gray < (bg_median - 18)).astype(np.uint8) * 255

    if cv2.countNonZero(ink_binary) < min_component_area:
        thresh_type = cv2.THRESH_BINARY if is_inverted else cv2.THRESH_BINARY_INV
        _, ink_binary = cv2.threshold(inner_gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    if cv2.countNonZero(ink_binary) == 0:
        return None

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        ink_binary, connectivity=8
    )

    valid_lbls: list[int] = []
    inner_h, inner_w = inner_gray.shape[:2]

    for lbl in range(1, num_labels):
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        if area >= min_component_area:
            # Exclude border line artifacts that span almost the entire inner width or height
            if (cw >= inner_w - 2 and ch <= 3) or (ch >= inner_h - 2 and cw <= 3):
                continue
            if (cw >= inner_w * 0.50 and ch <= 6) or (ch >= inner_h * 0.50 and cw <= 6):
                continue
            if (cw >= inner_w * 0.65 and cw / max(1, ch) >= 4) or (
                ch >= inner_h * 0.65 and ch / max(1, cw) >= 4
            ):
                continue
            valid_lbls.append(lbl)

    if not valid_lbls:
        return None

    # Filter isolated dust/speckles if there is a dominant content component
    max_area = max(stats[lbl, cv2.CC_STAT_AREA] for lbl in valid_lbls)
    main_lbls = [
        lbl
        for lbl in valid_lbls
        if stats[lbl, cv2.CC_STAT_AREA] >= max(min_component_area, max_area * 0.08)
    ]
    if not main_lbls:
        main_lbls = valid_lbls

    min_x = min(stats[lbl, cv2.CC_STAT_LEFT] for lbl in main_lbls) + margin_x
    min_y = min(stats[lbl, cv2.CC_STAT_TOP] for lbl in main_lbls) + margin_y
    max_x = (
        max(
            stats[lbl, cv2.CC_STAT_LEFT] + stats[lbl, cv2.CC_STAT_WIDTH]
            for lbl in main_lbls
        )
        + margin_x
    )
    max_y = (
        max(
            stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT]
            for lbl in main_lbls
        )
        + margin_y
    )

    return (min_x, min_y, max_x, max_y)


def center_cell_value(
    cell_crop: np.ndarray | Image.Image,
    config: CellProcessingConfig | None = None,
    margin_ratio_x: float = 0.08,
    margin_ratio_y: float = 0.06,
    min_component_area: int = 6,
    is_header: bool = False,
    row: int | None = None,
    col: int | None = None,
) -> np.ndarray:
    """Detect internal value inside cell_crop, move it to center, and return centered crop.

    Uses clean paper cloning and darkness-alpha blending to eliminate white ghosting,
    bleaching, and inpainting artifacts.
    Skips centering if is_header is True or if row == 0 or col == 0 (e.g. blocks [0][0]..[0][12] and [n][0]).

    Args:
        cell_crop: Cell crop image as numpy array or PIL Image.
        config: Optional CellProcessingConfig.
        margin_ratio_x: Inset ratio from left/right to exclude outer border lines.
        margin_ratio_y: Inset ratio from top/bottom to exclude outer border lines.
        min_component_area: Minimum connected component area.
        is_header: If True, multi-line header/label cells are preserved as-is.
        row: Optional row index in table grid.
        col: Optional column index in table grid.

    Returns:
        numpy uint8 array with the internal value cleanly centered (or raw crop if skipped).
    """
    raw_img = _to_numpy_image(cell_crop)
    h, w = raw_img.shape[:2]
    if h < 10 or w < 10 or is_header or row == 0 or col == 0:
        return raw_img.copy()

    orig_has_alpha = raw_img.ndim == 3 and raw_img.shape[2] == 4
    if orig_has_alpha:
        img = raw_img[:, :, :3].copy()
        orig_alpha = raw_img[:, :, 3].copy()
    else:
        img = raw_img.copy()
        orig_alpha = None

    bbox = detect_cell_value(
        img,
        config=config,
        margin_ratio_x=margin_ratio_x,
        margin_ratio_y=margin_ratio_y,
        min_component_area=min_component_area,
    )
    if bbox is None:
        if orig_has_alpha and orig_alpha is not None:
            return np.dstack([img, orig_alpha])
        return img.copy()

    min_x, min_y, max_x, max_y = bbox
    val_w = max_x - min_x
    val_h = max_y - min_y

    # If content occupies almost the entire cell, it is already a full-box label
    if val_h > 0.65 * h and val_w > 0.65 * w:
        if orig_has_alpha and orig_alpha is not None:
            return np.dstack([img, orig_alpha])
        return img.copy()

    val_cx = (min_x + max_x) / 2.0
    val_cy = (min_y + max_y) / 2.0
    target_cx = w / 2.0
    target_cy = h / 2.0

    dx = round(target_cx - val_cx)
    dy = round(target_cy - val_cy)

    # If already centered within 2 pixels, return as-is
    if abs(dx) <= 2 and abs(dy) <= 2:
        if orig_has_alpha and orig_alpha is not None:
            return np.dstack([img, orig_alpha])
        return img.copy()

    if config is not None:
        margin_ratio_x = getattr(config, "margin_ratio_x", margin_ratio_x)
        margin_ratio_y = getattr(config, "margin_ratio_y", margin_ratio_y)
    margin_x = max(2, int(w * margin_ratio_x))
    margin_y = max(2, int(h * margin_ratio_y))

    # Erase patch: vertical slice spanning the value with padding
    pad_y = 4
    if val_cy < h / 2.0:
        py1 = 2
        py2 = min(h - 2, max_y + pad_y)
    else:
        py1 = max(2, min_y - pad_y)
        py2 = h - 2

    patch_h = py2 - py1
    if patch_h < 4:
        if orig_has_alpha and orig_alpha is not None:
            return np.dstack([img, orig_alpha])
        return img.copy()

    clean_w_start = margin_x
    clean_w_end = w - margin_x

    # Sample clean paper from opposite vertical half
    if val_cy < h / 2.0:
        src_clean_y = max(2, min(h - 2 - patch_h, int(h * 0.55)))
    else:
        src_clean_y = max(2, min(h - 2 - patch_h, int(h * 0.10)))

    result = img.copy()
    clean_paper_slice = img[
        src_clean_y : src_clean_y + patch_h, clean_w_start:clean_w_end
    ]

    # Vertical feathering (top and bottom 3 pixels)
    blend_v = np.ones((patch_h, clean_w_end - clean_w_start, 1), dtype=np.float32)
    feather = min(4, patch_h // 4)
    if feather > 0:
        for f in range(feather):
            wt = (f + 1) / (feather + 1)
            blend_v[f, :, :] = wt
            blend_v[-(f + 1), :, :] = wt

    orig_slice = result[py1:py2, clean_w_start:clean_w_end].astype(np.float32)
    clean_float = clean_paper_slice.astype(np.float32)
    erased_slice = clean_float * blend_v + orig_slice * (1.0 - blend_v)
    result[py1:py2, clean_w_start:clean_w_end] = erased_slice.astype(np.uint8)

    # Now paste the original value patch at the destination
    pad_x = 2
    vx1 = max(margin_x, min_x - pad_x)
    vx2 = min(w - margin_x, max_x + pad_x)
    vy1 = max(2, min_y - pad_y)
    vy2 = min(h - 2, max_y + pad_y)

    dest_vy1 = vy1 + dy
    dest_vy2 = vy2 + dy
    dest_vx1 = vx1 + dx
    dest_vx2 = vx2 + dx

    # Clamp destination strictly inside cell borders
    if dest_vy1 < 2:
        shift = 2 - dest_vy1
        dest_vy1 += shift
        dest_vy2 += shift
        dy += shift
    if dest_vy2 > h - 2:
        shift = dest_vy2 - (h - 2)
        dest_vy1 -= shift
        dest_vy2 -= shift
        dy -= shift
    if dest_vx1 < margin_x:
        shift = margin_x - dest_vx1
        dest_vx1 += shift
        dest_vx2 += shift
        dx += shift
    if dest_vx2 > w - margin_x:
        shift = dest_vx2 - (w - margin_x)
        dest_vx1 -= shift
        dest_vx2 -= shift
        dx -= shift

    val_patch = img[vy1:vy2, vx1:vx2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    val_gray = gray[vy1:vy2, vx1:vx2]

    inner_gray = gray[margin_y : h - margin_y, margin_x : w - margin_x]
    bg_median = (
        float(np.median(inner_gray)) if inner_gray.size > 0 else float(np.median(gray))
    )

    darkness = (bg_median - val_gray.astype(np.float32)).clip(0, 255)
    alpha = np.clip(darkness / 16.0, 0.0, 1.0)
    if alpha.shape[0] >= 3 and alpha.shape[1] >= 3:
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = alpha[:, :, None]

    dest_area = result[dest_vy1:dest_vy2, dest_vx1:dest_vx2].astype(np.float32)
    val_float = val_patch.astype(np.float32)
    blended = val_float * alpha + dest_area * (1.0 - alpha)
    result[dest_vy1:dest_vy2, dest_vx1:dest_vx2] = blended.astype(np.uint8)

    # Preserve outer 2px borders strictly
    result[:2, :] = img[:2, :]
    result[-2:, :] = img[-2:, :]
    result[:, :2] = img[:, :2]
    result[:, -2:] = img[:, -2:]

    if orig_has_alpha and orig_alpha is not None:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_alpha = cv2.warpAffine(orig_alpha, M, (w, h), flags=cv2.INTER_NEAREST)
        return np.dstack([result, shifted_alpha])

    return result


# Aliases for block-level terminology
detect_block_value = detect_cell_value
center_block_value = center_cell_value


def reconstruct_table_from_cells(
    cells: list[list[Cell]] | list[Cell] | list[list[np.ndarray]],
    grid: GridGeometry | None = None,
    table_shape: tuple[int, int] | None = None,
    center_cells: bool = True,
    config: CellProcessingConfig | None = None,
    table_image: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a complete table image assembled from centered cell blocks.

    Args:
        cells: 2D matrix of Cell objects, flat list of Cell objects, or 2D list of numpy crops.
        grid: Optional GridGeometry defining cell coordinates.
        table_shape: Optional (height, width) of the target reconstructed table.
        center_cells: If True, centers the internal value of each cell if not already centered.
        config: Optional CellProcessingConfig.
        table_image: Optional original table image to serve as the base canvas.

    Returns:
        numpy uint8 BGR image representing the newly reconstructed table.
    """
    if not cells:
        raise ValueError("cells list cannot be empty")

    # Determine input type
    first_elem = cells[0]
    if isinstance(first_elem, list) and len(first_elem) > 0:
        first_item = first_elem[0]
        if isinstance(first_item, (np.ndarray, Image.Image)):
            # 2D matrix of crops
            centered_matrix: list[list[np.ndarray]] = []
            for r_idx, row in enumerate(cells):
                centered_row: list[np.ndarray] = []
                for c_idx, crop in enumerate(row):
                    is_skip = r_idx == 0 or c_idx == 0
                    c_crop = (
                        center_cell_value(
                            crop,
                            config=config,
                            is_header=is_skip,
                            row=r_idx,
                            col=c_idx,
                        )
                        if (center_cells and not is_skip)
                        else crop
                    )
                    centered_row.append(_to_numpy_image(c_crop))
                centered_matrix.append(centered_row)

            row_images = [np.hstack(row) for row in centered_matrix]
            return np.vstack(row_images)
        else:
            # 2D matrix of Cell objects
            cell_matrix = cells  # type: ignore
    else:
        # Flat list of Cell objects
        flat_list = cells  # type: ignore
        max_r = max(getattr(c, "row", 0) for c in flat_list)
        max_c = max(getattr(c, "col", 0) for c in flat_list)
        cell_matrix = [[None for _ in range(max_c + 1)] for _ in range(max_r + 1)]  # type: ignore
        for c in flat_list:
            r = getattr(c, "row", 0)
            col = getattr(c, "col", 0)
            cell_matrix[r][col] = c

    num_rows = len(cell_matrix)
    num_cols = len(cell_matrix[0]) if num_rows > 0 else 0

    # Ensure centered crops and value_bboxes exist
    for r in range(num_rows):
        for c in range(num_cols):
            cell = cell_matrix[r][c]
            if cell is None:
                continue
            raw = getattr(cell, "raw_crop", None)
            is_skip = r == 0 or c == 0
            if is_skip:
                cell.value_bbox = None
                if getattr(cell, "centered_crop", None) is None:
                    cell.centered_crop = (
                        _to_numpy_image(raw) if raw is not None else None
                    )
                continue
            if raw is not None:
                raw_np = _to_numpy_image(raw)
                if raw_np.size > 0:
                    if getattr(cell, "value_bbox", None) is None:
                        cell.value_bbox = detect_cell_value(raw_np, config=config)
                    if getattr(cell, "centered_crop", None) is None:
                        c_crop = (
                            center_cell_value(
                                raw_np,
                                config=config,
                                is_header=False,
                                row=r,
                                col=c,
                            )
                            if center_cells
                            else raw_np
                        )
                        cell.centered_crop = c_crop
                else:
                    if getattr(cell, "centered_crop", None) is None:
                        cell.centered_crop = raw_np
            else:
                if getattr(cell, "centered_crop", None) is None:
                    cell.centered_crop = raw

    # Strategy 1: GridGeometry or Cell bboxes to place onto canvas
    if grid is not None or table_shape is not None or table_image is not None:
        if table_image is not None:
            th, tw = table_image.shape[:2]
        elif grid is not None:
            th, tw = grid.table_height, grid.table_width
        else:
            th, tw = table_shape[0], table_shape[1]  # type: ignore

        if table_image is not None:
            canvas = table_image.copy()
        else:
            sample_cell = cell_matrix[0][0]
            sample_crop = getattr(
                sample_cell,
                "centered_crop",
                getattr(sample_cell, "raw_crop", None),
            )
            sample_np = (
                _to_numpy_image(sample_crop) if sample_crop is not None else None
            )
            channels = (
                sample_np.shape[2]
                if sample_np is not None and sample_np.ndim > 2
                else 3
            )
            dtype = sample_np.dtype if sample_np is not None else np.uint8
            canvas = (
                np.full((th, tw, channels), 255, dtype=dtype)
                if channels > 1
                else np.full((th, tw), 255, dtype=dtype)
            )

        for r in range(num_rows):
            for c in range(num_cols):
                cell = cell_matrix[r][c]
                if cell is None:
                    continue
                if hasattr(cell, "bbox") and cell.bbox and len(cell.bbox) == 4:
                    x1, y1, x2, y2 = cell.bbox
                elif grid is not None:
                    x1, y1, x2, y2 = grid.get_cell_bbox(r, c)
                else:
                    x1 = y1 = x2 = y2 = 0

                crop = getattr(cell, "centered_crop", getattr(cell, "raw_crop", None))
                if crop is not None and x2 > x1 and y2 > y1:
                    crop_np = _to_numpy_image(crop)
                    ch, cw = crop_np.shape[:2]
                    target_h = y2 - y1
                    target_w = x2 - x1
                    if ch == target_h and cw == target_w:
                        canvas[y1:y2, x1:x2] = crop_np
                    else:
                        resized = cv2.resize(crop_np, (target_w, target_h))
                        canvas[y1:y2, x1:x2] = resized

        return canvas

    # Strategy 2: Horizontal stack per row, then vertical stack across rows
    centered_blocks: list[list[np.ndarray]] = []
    for r in range(num_rows):
        row_crops: list[np.ndarray] = []
        for c in range(num_cols):
            cell = cell_matrix[r][c]
            crop = getattr(cell, "centered_crop", getattr(cell, "raw_crop", None))
            row_crops.append(_to_numpy_image(crop))
        centered_blocks.append(row_crops)

    row_images = [np.hstack(row) for row in centered_blocks]
    return np.vstack(row_images)


# Alias
reconstruct_table_from_blocks = reconstruct_table_from_cells
