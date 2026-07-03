"""
Standalone image-processing utilities for stretching and tonal adjustment.

All functions operate on float32 images in the [0, 65535] range (matching
raw camera ADU scale) unless otherwise noted. This preserves the calibrated
constants from the original MATLAB implementation without rescaling.
"""

import numpy as np
import cv2


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma to a float32 [0, 65535] image. Clips negatives before power."""
    return np.clip(image, 0.0, None) ** gamma


def sky_coefficient(t_accum: float) -> float:
    """Sky-subtraction strength: ramps 0 → 1 over ~30 s, starting after t = 3 s."""
    return float(max(0.0, 1 - np.exp(-(t_accum - 3.0) / 10.0)))


def saturation_coefficient(t_accum: float) -> float:
    """Saturation multiplier: ramps 1 → 2 starting around t = 7 s."""
    return float(max(1.0, min(2, 2 - np.exp(-(t_accum - 7.0) / 4.0))))


def auto_brightness(imdat: np.ndarray, gamma: float, skyco: float) -> float:
    """
    Compute a brightness gain for a gamma-corrected [0, 65535] image.

    Applies the tighter of two constraints:
      - post-gamma noise  ≤ 10% of full scale (65535)
      - post-gamma, sky-subtracted mean ≤ 28% of full scale

    imdat  : float32 [0, 65535], pre-gamma stack (before sky subtraction)
    skyco  : sky coefficient — used to estimate the sky-subtracted mean
    Returns: gain to multiply against gamma_correct(imdat, gamma)
    """
    MAX_NOISE_FRAC = 0.07
    MAX_MEAN_FRAC = 0.2

    median_val = float(np.median(imdat))
    mean_val = float(np.mean(imdat))

    # Poisson noise estimate: sqrt(median/4), gamma-compressed
    nval = max(1e-6, ((median_val / 4.0) ** 0.5) ** gamma)
    # Sky-subtracted mean, gamma-compressed
    mval = max(1e-6, ((1.0 - skyco) * mean_val) ** gamma)

    return min(65535.0 * MAX_NOISE_FRAC / nval, 65535.0 * MAX_MEAN_FRAC / mval)


def saturation_fraction(image: np.ndarray, threshold: int = 65000) -> float:
    """Fraction of pixels at or above threshold. Used for auto-exposure decisions."""
    return float(np.sum(image >= threshold)) / image.size


def adjust_saturation(image: np.ndarray, satco: float) -> np.ndarray:
    """
    Scale the saturation channel of a BGR float32 [0, 65535] image.

    satco = 1.0 → no change; satco = 2.0 → double saturation.
    Normalizes to [0, 1] for the HLS conversion, then restores the original scale.
    """
    norm = (image / 65535.0).astype(np.float32)
    hls = cv2.cvtColor(norm, cv2.COLOR_BGR2HLS)
    hls[:, :, 2] = np.clip(hls[:, :, 2] * satco, 0.0, 1.0)
    return (cv2.cvtColor(hls, cv2.COLOR_HLS2BGR) * 65535.0).astype(np.float32)
