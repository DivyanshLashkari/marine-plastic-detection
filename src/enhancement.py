"""
Marine Plastic Detection - Image Enhancement Pipeline

Implements CLAHE, gamma correction, denoising, and contrast
enhancement for improving detection under degraded conditions.
Each method accepts a BGR numpy array and returns the enhanced BGR array.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    GAMMA_VALUE,
    DENOISE_H,
    DENOISE_TEMPLATE_WINDOW,
    DENOISE_SEARCH_WINDOW,
    CONTRAST_ALPHA,
    CONTRAST_BETA,
)


# ──────────────────────────────────────────────
# Individual Enhancement Methods
# ──────────────────────────────────────────────

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Converts to LAB color space, applies CLAHE on the L channel,
    then converts back to BGR.  Effective for low-light and
    uneven-illumination scenarios.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_ch)

    merged = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(
    img: np.ndarray,
    gamma: float = GAMMA_VALUE,
) -> np.ndarray:
    """
    Apply gamma correction.

    gamma > 1  →  brighten dark regions (useful for low-light images).
    gamma < 1  →  darken bright regions.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def apply_denoise(
    img: np.ndarray,
    h: int = DENOISE_H,
    template_window: int = DENOISE_TEMPLATE_WINDOW,
    search_window: int = DENOISE_SEARCH_WINDOW,
) -> np.ndarray:
    """
    Apply Non-Local Means Denoising (color variant).

    Reduces high-frequency noise while preserving edges.
    """
    return cv2.fastNlMeansDenoisingColored(
        img, None, h, h, template_window, search_window
    )


def apply_contrast(
    img: np.ndarray,
    alpha: float = CONTRAST_ALPHA,
    beta: int = CONTRAST_BETA,
) -> np.ndarray:
    """
    Apply linear contrast / brightness adjustment.

    new_pixel = alpha * pixel + beta
    alpha > 1 increases contrast; beta shifts brightness.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ──────────────────────────────────────────────
# Enhancement Dispatcher
# ──────────────────────────────────────────────

_METHODS = {
    "clahe": apply_clahe,
    "gamma": apply_gamma_correction,
    "denoise": apply_denoise,
    "contrast": apply_contrast,
}


def enhance_image(
    img: np.ndarray,
    method: str,
    **kwargs,
) -> np.ndarray:
    """
    Apply a named enhancement method.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format.
    method : str
        One of 'clahe', 'gamma', 'denoise', 'contrast'.
    **kwargs
        Optional overrides forwarded to the underlying function.

    Returns
    -------
    np.ndarray
        Enhanced image in BGR format.

    Raises
    ------
    ValueError
        If *method* is not recognized.
    """
    method = method.lower().strip()
    if method not in _METHODS:
        raise ValueError(
            f"Unknown enhancement method '{method}'. "
            f"Choose from: {', '.join(sorted(_METHODS))}"
        )
    return _METHODS[method](img, **kwargs)


def get_available_methods():
    """Return a list of registered enhancement method names."""
    return list(_METHODS.keys())


def get_method_description(method: str) -> str:
    """Return a human-readable description for an enhancement method."""
    descriptions = {
        "clahe": "CLAHE — Adaptive histogram equalization for uneven illumination",
        "gamma": "Gamma Correction — Brighten or darken via power-law transform",
        "denoise": "Denoising — Non-local means noise reduction preserving edges",
        "contrast": "Contrast — Linear contrast and brightness adjustment",
    }
    return descriptions.get(method.lower(), method)
