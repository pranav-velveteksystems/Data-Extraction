"""Image enhancement and contrast adjustment."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if not already."""
    if len(image.shape) == 2:
        return image.copy()
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")


def apply_clahe(
    gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def denoise(gray: np.ndarray, strength: int = 5) -> np.ndarray:
    """Denoise grayscale image while preserving edges."""
    if strength <= 0:
        return gray
    # Bilateral filter preserves sharp grid lines while smoothing noise
    diameter = max(3, strength)
    return cv2.bilateralFilter(gray, diameter, 75, 75)


def preprocess_image(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    denoise_strength: int = 5,
) -> dict[str, np.ndarray]:
    """Perform full enhancement stage returning multiple image representations.

    Returns:
        Dict with 'original', 'gray', 'enhanced', and 'denoised'
    """
    gray = to_grayscale(image)
    enhanced = apply_clahe(gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    denoised = denoise(enhanced, strength=denoise_strength)

    return {"original": image, "gray": gray, "enhanced": enhanced, "denoised": denoised}
