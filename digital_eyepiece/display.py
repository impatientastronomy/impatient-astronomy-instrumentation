"""
Display utilities for the digital eyepiece.

Converts raw uint16 numpy frames to pygame Surfaces for rendering.
"""

import numpy as np
import pygame


def stretch_to_uint8(
    image: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
    brightness: float = 1.0,
) -> np.ndarray:
    """
    Percentile stretch a uint16 image to uint8 for display.

    Works on 2-D (mono) or 3-D (H×W×C) arrays.  The percentiles are computed
    over all pixels so colour channels are scaled consistently.
    brightness is applied at float32 precision before quantising to uint8.
    """
    lo = float(np.percentile(image, lo_pct))
    hi = float(np.percentile(image, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(image.astype(np.float32), lo, hi)
    scaled = np.clip((clipped - lo) / (hi - lo) * 255.0 * brightness, 0.0, 255.0)
    return scaled.astype(np.uint8)


def to_surface(image: np.ndarray) -> pygame.Surface:
    """
    Convert a uint8 numpy array to a pygame Surface.

    Accepts:
      - H×W      → greyscale, displayed as RGB
      - H×W×3    → BGR (from cv2 debayer), converted to RGB
    """
    if image.ndim == 2:
        # Mono: broadcast to RGB
        rgb = np.stack([image, image, image], axis=2)
    else:
        # cv2 returns BGR; pygame wants RGB
        rgb = image[:, :, ::-1].copy()

    # pygame.surfarray expects (W, H, 3) with C-contiguous memory
    transposed = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    return pygame.surfarray.make_surface(transposed)
