"""Perspective correction for document images."""

from __future__ import annotations

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff (x - y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff (x - y)

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to warp image based on 4 points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


def is_already_rectilinear_document(
    image: np.ndarray, border_thickness: int = 10, min_white_mean: float = 140.0
) -> bool:
    """Check if image borders are predominantly white/light, indicating a scanned or cropped page."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    if h < border_thickness * 2 or w < border_thickness * 2:
        return True

    top = np.mean(gray[0:border_thickness, :])
    bottom = np.mean(gray[h - border_thickness : h, :])
    left = np.mean(gray[:, 0:border_thickness])
    right = np.mean(gray[:, w - border_thickness : w])

    avg_border = (top + bottom + left + right) / 4.0
    return avg_border >= min_white_mean


def is_quadrilateral_perspective_distorted(
    pts: np.ndarray, angle_threshold_deg: float = 4.0
) -> bool:
    """Check whether a 4-point contour exhibits significant perspective tilt or distortion."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Top edge angle relative to horizontal
    dx_top, dy_top = tr[0] - tl[0], tr[1] - tl[1]
    angle_top = abs(np.degrees(np.arctan2(dy_top, dx_top)))

    # Bottom edge angle relative to horizontal
    dx_bot, dy_bot = br[0] - bl[0], br[1] - bl[1]
    angle_bot = abs(np.degrees(np.arctan2(dy_bot, dx_bot)))

    # Left edge angle relative to vertical
    dx_left, dy_left = bl[0] - tl[0], bl[1] - tl[1]
    angle_left = abs(np.degrees(np.arctan2(dx_left, dy_left)))

    # Right edge angle relative to vertical
    dx_right, dy_right = br[0] - tr[0], br[1] - tr[1]
    angle_right = abs(np.degrees(np.arctan2(dx_right, dy_right)))

    max_tilt = max(angle_top, angle_bot, angle_left, angle_right)
    return max_tilt >= angle_threshold_deg


def find_document_contour(
    image: np.ndarray, min_area_ratio: float = 0.35
) -> np.ndarray | None:
    """Find the 4-corner document contour in the image, if present.

    Avoids false positives when the image is already a white document sheet.
    """
    # If the image border is already white paper, no perspective warp is needed
    if is_already_rectilinear_document(image):
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    total_area = image.shape[0] * image.shape[1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Multi-strategy contour detection: Canny vs Otsu
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * total_area:
            break

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # If the contour spans > 95% of the image, it is essentially the image itself
            if area > 0.95 * total_area:
                return None
            return approx.reshape(4, 2)

    # Fallback to Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * total_area:
            break

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            if area > 0.95 * total_area:
                return None
            return approx.reshape(4, 2)

    return None


def correct_perspective(
    image: np.ndarray, min_area_ratio: float = 0.20
) -> tuple[np.ndarray, bool, np.ndarray | None]:
    """Detect document contour and correct perspective.

    Returns:
        Tuple of (result_image, was_corrected, corners)
    """
    corners = find_document_contour(image, min_area_ratio=min_area_ratio)
    if corners is not None and is_quadrilateral_perspective_distorted(corners):
        try:
            warped = four_point_transform(image, corners)
            return warped, True, corners
        except Exception:
            return image, False, None

    return image, False, corners
