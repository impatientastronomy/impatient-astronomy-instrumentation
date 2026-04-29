"""
Digital zoom — crop and upscale a numpy image array.

apply_digital_zoom() is a pure function: it takes an image and zoom
parameters and returns a new image of the same shape. The main loop calls
it on each frame using the current zoom_level from ViewState.

Requires opencv-python: pip install opencv-python
"""

from __future__ import annotations

import numpy as np


def apply_digital_zoom(
    image: np.ndarray,
    zoom: float,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> np.ndarray:
    """
    Crop and resize *image* to simulate digital zoom.

    Parameters
    ----------
    image :
        Input array, shape (H, W) or (H, W, C). Any dtype.
    zoom :
        Zoom factor. 1.0 returns the image unchanged.
        2.0 shows the central 50 % of the image scaled to full size.
        Values < 1.0 are treated as 1.0 (no zoom-out on a single camera).
    center_x, center_y :
        Zoom center as normalized coordinates in [0, 1].
        (0.5, 0.5) is the image center (default).

    Returns
    -------
    np.ndarray
        Same shape and dtype as *image*.
    """
    if zoom <= 1.0:
        return image

    import cv2

    h, w = image.shape[:2]

    # Crop window dimensions
    crop_h = max(1, round(h / zoom))
    crop_w = max(1, round(w / zoom))

    # Desired center in pixel space, clamped so the window fits inside the image
    cx = round(center_x * w)
    cy = round(center_y * h)
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))

    crop = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
