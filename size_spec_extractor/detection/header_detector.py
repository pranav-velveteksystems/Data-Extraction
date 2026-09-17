"""Top rectangle (Category, Style Code, Name) detection and table header attachment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import HeaderBoxConfig


@dataclass
class HeaderROI:
    """Region of Interest for the extracted top rectangle containing metadata."""

    image: np.ndarray
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 1.0
    box_type: str = "metadata"

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
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def attach_header_to_table(
    table_image: np.ndarray,
    header_image: np.ndarray,
    divider_thickness: int = 0,
    divider_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray | None:
    """Attach the extracted top metadata rectangle to the top of a table image.

    The header image is scaled horizontally to match the table width while
    preserving its original aspect ratio, then stacked vertically above the table.

    Args:
        table_image: BGR or grayscale numpy image of the table.
        header_image: BGR or grayscale numpy image of the top metadata box.
        divider_thickness: Optional thickness in pixels for a divider line between header and table.
        divider_color: Optional BGR color tuple for the divider line.

    Returns:
        New numpy BGR image with the header stacked on top of the table, or None if table_image is invalid.
    """
    if table_image is None or getattr(table_image, "size", 0) == 0:
        return None
    if header_image is None or getattr(header_image, "size", 0) == 0:
        return table_image.copy() if hasattr(table_image, "copy") else table_image

    tbl = table_image
    hdr = header_image

    # Harmonize channels
    if tbl.ndim == 3 and hdr.ndim == 2:
        hdr = cv2.cvtColor(hdr, cv2.COLOR_GRAY2BGR)
    elif tbl.ndim == 2 and hdr.ndim == 3:
        tbl = cv2.cvtColor(tbl, cv2.COLOR_GRAY2BGR)

    tw = tbl.shape[1]
    hh, hw = hdr.shape[:2]

    if hw <= 0 or hh <= 0 or tw <= 0:
        return tbl

    target_w = tw
    target_h = max(1, round(hh * float(target_w) / float(hw)))

    interp = cv2.INTER_AREA if target_w < hw else cv2.INTER_CUBIC
    resized_hdr = cv2.resize(hdr, (target_w, target_h), interpolation=interp)

    parts = [resized_hdr]
    if divider_thickness > 0:
        if tbl.ndim == 3:
            div = np.full(
                (divider_thickness, target_w, 3), divider_color, dtype=np.uint8
            )
        else:
            div_val = int(np.mean(divider_color))
            div = np.full((divider_thickness, target_w), div_val, dtype=np.uint8)
        parts.append(div)
    parts.append(tbl)

    return np.vstack(parts)


def detect_header_box(
    image: np.ndarray,
    table_bbox: tuple[int, int, int, int] | None = None,
    config: HeaderBoxConfig | None = None,
) -> HeaderROI | None:
    """Detect the top rectangle containing Category, Style Code, and Name.

    Args:
        image: Full document image (perspective-corrected) as numpy array.
        table_bbox: Optional bounding box (x1, y1, x2, y2) of the detected size spec table.
        config: Optional HeaderBoxConfig.

    Returns:
        HeaderROI containing the cropped box image and its coordinates (x1, y1, x2, y2),
        or None if disabled or not found.
    """
    if config is not None and not config.enabled:
        return None

    h, w = image.shape[:2]
    if h < 20 or w < 20:
        return None

    # Handle manual bbox
    if config is not None and config.manual_bbox is not None:
        mx1, my1, mx2, my2 = config.manual_bbox
        mx1, my1 = max(0, mx1), max(0, my1)
        mx2, my2 = min(w, mx2), min(h, my2)
        if mx2 > mx1 and my2 > my1:
            crop = image[my1:my2, mx1:mx2].copy()
            return HeaderROI(
                image=crop, bbox=(mx1, my1, mx2, my2), confidence=1.0, box_type="manual"
            )

    # Table at the top edge means no header space above table
    if table_bbox is not None and table_bbox[1] <= 35:
        return None

    mode = config.mode if config is not None else "auto"

    table_y1 = table_bbox[1] if table_bbox else int(h * 0.45)
    max_search_y = max(40, min(table_y1, int(h * 0.40)))
    max_header_h = max(35, int(h * 0.12))

    top = image[:max_search_y, :]
    th, tw = top.shape[:2]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top.copy()
    bw = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -2
    )

    # 1. Line kernels
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(tw * 0.05)), 1))
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(th * 0.04))))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)

    # Connect small vertical gaps (e.g. 5px gap between rows)
    vl_closed = cv2.morphologyEx(
        vl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    )

    # Method 1: Find vertical divider in range x between 0.15*w and 0.55*w
    cnts_v, _ = cv2.findContours(vl_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    div_cand = []
    min_div_h = max(15, int(th * 0.02))
    for c in cnts_v:
        x, y, cw, ch = cv2.boundingRect(c)
        if 0.15 * tw <= (x + cw / 2.0) <= 0.55 * tw and ch >= min_div_h:
            div_cand.append((x, y, cw, ch))

    if div_cand:
        # Pick the divider closest to the top
        div_cand.sort(key=lambda d: (d[1], -d[3]))
        best_div = div_cand[0]
        x_div = best_div[0]
        y_top_est = best_div[1]
        y_bot_est = best_div[1] + best_div[3]
        div_h = best_div[3]

        # Snap y_top and y_bot to horizontal lines in hl focused in the metadata block
        h_proj = hl[:, max(0, x_div - 10) :].sum(axis=1) / 255.0
        win = max(18, int(div_h * 0.20))
        thresh = max(
            tw * 0.08, float(np.max(h_proj)) * 0.20 if np.max(h_proj) > 0 else 1.0
        )

        # Snap top line: look for line cluster around y_top_est
        top_slice = h_proj[max(0, y_top_est - win) : min(th, y_top_est + win)]
        top_cand = np.where(top_slice >= thresh)[0]
        if len(top_cand) > 0:
            top_ys = top_cand + max(0, y_top_est - win)
            peak_y = int(top_ys[np.argmax(h_proj[top_ys])])
            y1_snapped = peak_y
            while y1_snapped > 0 and h_proj[y1_snapped - 1] >= thresh * 0.4:
                y1_snapped -= 1
            y_top = y1_snapped
        else:
            y_top = y_top_est

        # Snap bot line: look for line cluster around y_bot_est
        bot_slice = h_proj[max(0, y_bot_est - win) : min(th, y_bot_est + win)]
        bot_cand = np.where(bot_slice >= thresh)[0]
        if len(bot_cand) > 0:
            bot_ys = bot_cand + max(0, y_bot_est - win)
            peak_y = int(bot_ys[np.argmax(h_proj[bot_ys])])
            y2_snapped = peak_y
            while y2_snapped < th - 1 and h_proj[y2_snapped + 1] >= thresh * 0.4:
                y2_snapped += 1
            y_bot = y2_snapped + 1
        else:
            y_bot = y_bot_est + 1

        # Find line extent around y_top and y_bot to determine x_left and x_right
        h_top_strip = hl[max(0, y_top - 2) : min(th, y_top + 3), :]
        h_bot_strip = hl[max(0, y_bot - 3) : min(th, y_bot + 2), :]
        comb_strip = cv2.bitwise_or(h_top_strip, h_bot_strip)
        comb_xs = np.where(comb_strip.sum(axis=0) > 0)[0]
        if len(comb_xs) > 0:
            x_left = int(comb_xs[0])
            x_right = int(comb_xs[-1]) + 1
        else:
            x_left = 0
            x_right = tw

        if mode == "full":
            x1 = max(0, x_left)
            x2 = min(tw, x_right)
            box_type = "full_header"
        else:  # "metadata" or "auto"
            x1 = max(0, x_div)
            x2 = min(tw, max(x_right, x_div + 50))
            box_type = "metadata"

        y1 = max(0, y_top)
        y2 = min(th, y_bot)

        if x2 > x1 + 20 and y2 > y1 + 15:
            crop = image[y1:y2, x1:x2].copy()
            return HeaderROI(
                image=crop, bbox=(x1, y1, x2, y2), confidence=0.95, box_type=box_type
            )

    # Method 2: Contours on combined lines
    grid = cv2.add(hl, vl)
    grid_dilated = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts, _ = cv2.findContours(grid_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    header_boxes = []
    for c in cnts:
        x, y, bw_box, bh_box = cv2.boundingRect(c)
        if (
            y <= max_header_h * 1.5
            and bw_box >= tw * 0.15
            and 18 <= bh_box <= max_header_h
        ):
            header_boxes.append((x, y, x + bw_box, y + bh_box))

    if header_boxes:
        wide_box = max(header_boxes, key=lambda b: b[2] - b[0])
        min_x, min_y, max_x, max_y = wide_box
        box_type = "metadata"
        if mode == "full":
            box_type = "full_header"
            left_boxes = [
                b for b in header_boxes if b[2] <= min_x + 20 and abs(b[1] - min_y) < 25
            ]
            if left_boxes:
                min_x = min(b[0] for b in left_boxes)

        x1, y1, x2, y2 = max(0, min_x), max(0, min_y), min(tw, max_x), min(th, max_y)
        if x2 > x1 + 20 and y2 > y1 + 15:
            crop = image[y1:y2, x1:x2].copy()
            return HeaderROI(
                image=crop, bbox=(x1, y1, x2, y2), confidence=0.90, box_type=box_type
            )

    # Method 3: Text / ink fallback (e.g. synthetic sheet with no borders)
    non_zero = cv2.findNonZero(bw)
    if non_zero is not None:
        pts = non_zero.reshape(-1, 2)
        pts_top = pts[pts[:, 1] < max_header_h * 1.5]
        if len(pts_top) >= 50:
            x1 = max(0, int(np.percentile(pts_top[:, 0], 1)) - 8)
            x2 = min(w, int(np.percentile(pts_top[:, 0], 99)) + 8)
            y1 = max(0, int(np.percentile(pts_top[:, 1], 1)) - 8)
            y2 = min(h, int(np.percentile(pts_top[:, 1], 99)) + 8)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crop = image[y1:y2, x1:x2].copy()
                return HeaderROI(
                    image=crop,
                    bbox=(x1, y1, x2, y2),
                    confidence=0.75,
                    box_type="fallback",
                )

    return None
