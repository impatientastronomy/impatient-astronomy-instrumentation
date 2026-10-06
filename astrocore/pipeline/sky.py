"""
Sky background estimation via sigma-clipping and separable polynomial fitting.

The approach mirrors the original MATLAB method:
  1. Clip bright pixels (stars) down toward the local mean.
  2. Fit a polynomial to each row to capture horizontal background variation.
  3. Fit a polynomial to each column of that result to capture vertical variation.

The two-pass separable fit is fast (pure matrix math via lstsq) and produces
a smooth 2-D background model equivalent to a low-order 2-D polynomial without
the expense of a full 2-D fit.
"""

import cv2
import numpy as np


def fit_sky_model(
    image: np.ndarray,
    sigma: float = 2.0,
    degree: int = 2,
    dwn_size: int = 400,
) -> np.ndarray:
    """
    Estimate the sky background for a float32 [0, 65535] image.

    image    : shape (H, W) or (H, W, C) — processed per channel
    sigma    : pixels brighter than mean + sigma*std are treated as stars and clipped
    degree   : polynomial degree for row and column fits (1 = linear, 2 = quadratic)
    dwn_size : if the image's longer side exceeds this, fit on a downsampled
        copy and upscale the result instead. The sky background is a smooth,
        low-order surface by construction -- fitting it at full sensor
        resolution (this runs every accepted stacked frame, on the main
        thread) wastes cost for no change to the fitted model. Matches the
        same downsample-then-upscale approach _setup_reference() already
        uses for its own sky estimate.
    Returns: sky model of the same shape and dtype as image
    """
    h, w = image.shape[:2]
    scale = dwn_size / max(h, w)
    if scale < 1.0:
        small = cv2.resize(
            image, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        small_sky = _fit_all_channels(small, sigma, degree)
        return cv2.resize(small_sky, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return _fit_all_channels(image, sigma, degree)


def _fit_all_channels(image: np.ndarray, sigma: float, degree: int) -> np.ndarray:
    if image.ndim == 3:
        return np.stack(
            [_fit_channel(image[:, :, c], sigma, degree) for c in range(image.shape[2])],
            axis=2,
        )
    return _fit_channel(image, sigma, degree)


def _fit_channel(img: np.ndarray, sigma: float, degree: int) -> np.ndarray:
    h, w = img.shape
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)

    # Sigma clip: replace star pixels with the image mean so they don't bias the fit
    mean = img.mean()
    std = img.std()
    clipped = np.where(img > mean + sigma * std, mean, img).astype(np.float64)

    # Fit all rows at once: solve Vx @ C = clipped.T
    # Vx: (W, degree+1), clipped.T: (W, H), C: (degree+1, H)
    Vx = np.vander(x, degree + 1)
    C_rows, _, _, _ = np.linalg.lstsq(Vx, clipped.T, rcond=None)
    row_fit = (Vx @ C_rows).T  # (H, W)

    # Fit all columns of the row-smoothed result: solve Vy @ C = row_fit
    # Vy: (H, degree+1), row_fit: (H, W), C: (degree+1, W)
    Vy = np.vander(y, degree + 1)
    C_cols, _, _, _ = np.linalg.lstsq(Vy, row_fit, rcond=None)
    sky = Vy @ C_cols  # (H, W)

    return sky.astype(np.float32)
