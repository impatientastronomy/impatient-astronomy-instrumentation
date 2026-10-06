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
    max_gain: float = 30.0,
) -> np.ndarray:
    """
    Percentile stretch a uint16 image to uint8 for display.

    Works on 2-D (mono) or 3-D (H×W×C) arrays.  The percentiles are computed
    over all pixels so colour channels are scaled consistently.
    brightness is a separate, user-controlled multiplier applied on top.

    max_gain caps the histogram-derived multiplier (255 / (hi - lo)). Without
    it, a near-uniform frame -- lens cap on, or a blank patch of sky -- has a
    vanishingly small (hi - lo) and gets stretched into full-contrast noise.
    Tune this value directly if streamed frames still look too bright/noisy
    (lower = less amplification) or too flat (higher = more).
    """
    lo = float(np.percentile(image, lo_pct))
    hi = float(np.percentile(image, hi_pct))
    gain = min(255.0 / max(hi - lo, 1e-6), max_gain)
    clipped = np.clip(image.astype(np.float32), lo, None)
    scaled = np.clip((clipped - lo) * gain * brightness, 0.0, 255.0)
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
