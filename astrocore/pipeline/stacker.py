"""
Live image stacker for electronically assisted astronomy.

Accumulates calibrated frames into a running mean, then on each display tick
applies sky subtraction, gamma, auto-brightness, saturation, and temporal blending
to produce a smooth, progressively improving display image.

Classes
-------
ExposureSequence  -- steps through a predefined ramp of exposure times
Stacker           -- accumulates frames and produces display-ready images
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

from .sky import fit_sky_model
from .stretch import (
    auto_brightness,
    adjust_saturation,
    gamma_correct,
    saturation_coefficient,
    sky_coefficient,
)

_BLEND_ALPHA    = 0.06    # blending weight for each display tick
_MAX_SHIFT_PX   = 50.0    # reject frames with alignment shift larger than this
_ALIGN_RF       = 8       # block-sum downsample factor for alignment preprocessing


class ExposureSequence:
    """
    Steps through a predefined list of exposure times for a stacking session.

    Advances by one step each time a frame is successfully added to the stack.
    Stays at the final value once the end of the sequence is reached.
    Call reset() when the user switches back to streaming so the next stacking
    session starts fresh from the shortest exposure.

    Example::

        seq = ExposureSequence([0.1, 1, 2, 5, 5, 10, 20])
        while stacking:
            cam.exposure_time = seq.current
            if stacker.add_frame(frame, seq.current):
                seq.advance()
    """

    def __init__(self, sequence: list[float]) -> None:
        if not sequence:
            raise ValueError("sequence must not be empty")
        self._sequence = list(sequence)
        self._index = 0

    @property
    def current(self) -> float:
        """Current exposure time in seconds."""
        return self._sequence[self._index]

    def advance(self) -> None:
        """Move to the next exposure; stay at the last if already at the end."""
        self._index = min(self._index + 1, len(self._sequence) - 1)

    def reset(self) -> None:
        """Return to the first exposure in the sequence."""
        self._index = 0

    @property
    def at_end(self) -> bool:
        """True once the final exposure has been reached."""
        return self._index == len(self._sequence) - 1


class Stacker:
    """
    Accumulates calibrated frames and produces display-ready uint8 images.

    Typical usage in a stacking loop::

        stacker = Stacker()
        while True:
            result = grabber.grab_frame(exposure_us=exposure_us)
            if result.status == GrabStatus.SUCCESS:
                stacker.add_frame(result.frame.data, exposure_s=exposure_s)
            display = stacker.get_display_frame()
            if display is not None:
                show(display)
    """

    def __init__(self, gamma: float = 0.7, median_kernel: int = 3) -> None:
        """
        gamma         : gamma exponent applied before display (0.7 is a good starting point)
        median_kernel : spatial median filter kernel size applied to each incoming frame
        """
        self.gamma = gamma
        self.median_kernel = median_kernel

        self._stack_sum: np.ndarray | None = None
        self._ref_frame: np.ndarray | None = None   # previous accepted raw frame
        self._cumulative_shift: tuple[float, float] = (0.0, 0.0)
        self._frame_count: int = 0
        self._skipped_count: int = 0
        self._t_accum: float = 0.0
        self._blend: np.ndarray | None = None
        self._sky_model: np.ndarray | None = None
        self._sky_dirty: bool = True

    # ── read-only state ───────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def skipped_count(self) -> int:
        return self._skipped_count

    @property
    def t_accum(self) -> float:
        """Total accumulated exposure time in seconds."""
        return self._t_accum

    @property
    def stack(self) -> np.ndarray | None:
        """Mean-stacked image, float32 [0, 65535]. None until the first frame."""
        if self._stack_sum is None or self._frame_count == 0:
            return None
        return self._stack_sum / self._frame_count

    # ── frame ingestion ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Discard the current stack and start fresh."""
        self._stack_sum = None
        self._ref_frame = None
        self._cumulative_shift = (0.0, 0.0)
        self._frame_count = 0
        self._skipped_count = 0
        self._t_accum = 0.0
        self._blend = None
        self._sky_model = None
        self._sky_dirty = True

    def add_frame(self, frame: np.ndarray, exposure_s: float) -> bool:
        """
        Add a calibrated frame to the stack.

        Applies a median filter, aligns the frame to the running mean via phase
        correlation, and rejects frames whose shift exceeds _MAX_SHIFT_PX.

        frame      : uint16 or float32 from grab_frame() — (H, W) or (H, W, C)
        exposure_s : exposure duration in seconds
        Returns True if the frame passed quality checks and was added.
        """
        img = _median_filter(frame.astype(np.float32), self.median_kernel)

        if self._stack_sum is None:
            self._stack_sum = img.copy()
            self._ref_frame = img.copy()
            self._cumulative_shift = (0.0, 0.0)
            self._frame_count = 1
            self._t_accum = exposure_s
            self._sky_dirty = True
            return True

        # Incremental shift: compare to the previous accepted raw frame so only
        # per-frame drift matters, not total drift from the first frame.
        incremental, _ = _estimate_shift(img, self._ref_frame)
        if not _is_good_shift(incremental):
            self._skipped_count += 1
            dx, dy = incremental
            logging.debug("frame skipped: shift=(%.1f, %.1f) mag=%.1f", dx, dy, (dx**2+dy**2)**0.5)
            return False

        cx, cy = self._cumulative_shift
        ix, iy = incremental
        self._cumulative_shift = (cx + ix, cy + iy)

        self._stack_sum = self._stack_sum + _apply_shift(img, self._cumulative_shift)
        self._frame_count += 1
        self._t_accum += exposure_s
        self._sky_dirty = True
        self._ref_frame = img.copy()
        return True

    # ── display output ────────────────────────────────────────────────────

    def get_display_frame(self) -> np.ndarray | None:
        """
        Process the current stack into a displayable BGR uint8 image.

        Pipeline per call:
          sky subtraction → gamma + auto-brightness → saturation → temporal blend

        The sky model is recomputed only when a new frame has been added.
        The temporal blend is updated on every call, smoothing the transition
        between stack updates.

        Returns None until the first frame has been added.
        """
        imStack = self.stack
        if imStack is None:
            return None

        skyco = sky_coefficient(self._t_accum)
        satco = saturation_coefficient(self._t_accum)

        # Sky model is cached and only refit after a new frame arrives
        if self._sky_dirty or self._sky_model is None:
            self._sky_model = fit_sky_model(imStack)
            self._sky_dirty = False

        # Sky subtraction: remove a fraction of the background model
        imPrc = np.clip(imStack - skyco * self._sky_model, 0.0, None)

        # Brightness gain computed on the pre-gamma stack; (1-skyco) factor
        # inside auto_brightness accounts for the sky subtraction effect on the mean
        gn = auto_brightness(imStack, self.gamma, skyco)
        imPrc = np.clip(gamma_correct(imPrc, self.gamma) * gn, 0.0, 65535.0)

        if imPrc.ndim == 3:
            imPrc = adjust_saturation(imPrc, satco)

        # Temporal blend: smooths the display across loop iterations so the image
        # doesn't jump when a new frame is added to the stack
        if self._blend is None:
            self._blend = imPrc.copy()
        else:
            self._blend = (1.0 - _BLEND_ALPHA) * self._blend + _BLEND_ALPHA * imPrc

        return (np.clip(self._blend, 0.0, 65535.0) / 256.0).astype(np.uint8)


# ── internal helpers ──────────────────────────────────────────────────────────

def _median_filter(image: np.ndarray, kernel: int) -> np.ndarray:
    # cv2.medianBlur requires an integer type for kernels > 3; cast through uint16.
    u16 = np.clip(image, 0, 65535).astype(np.uint16)
    if u16.ndim == 3:
        filtered = np.stack(
            [cv2.medianBlur(u16[:, :, c], kernel) for c in range(u16.shape[2])],
            axis=2,
        )
    else:
        filtered = cv2.medianBlur(u16, kernel)
    return filtered.astype(np.float32)


def _make_star_image(img: np.ndarray) -> np.ndarray:
    """
    Downsample by block sum and subtract a blurred background to isolate star signals.

    Block-summing integrates star flux into fewer pixels while averaging down
    noise; background subtraction removes sky gradient and light pollution so
    the phase correlator locks onto stars instead of broad background structure.
    """
    mono = img[:, :, 0].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    h, w = mono.shape
    h2 = (h // _ALIGN_RF) * _ALIGN_RF
    w2 = (w // _ALIGN_RF) * _ALIGN_RF
    reduced = (
        mono[:h2, :w2]
        .reshape(h2 // _ALIGN_RF, _ALIGN_RF, w2 // _ALIGN_RF, _ALIGN_RF)
        .sum(axis=(1, 3))
    )
    # Large box blur estimates the sky background at reduced resolution.
    # 31x31 at rf=8 covers ~248 full-res px — large enough to span inter-star gaps.
    bg = cv2.blur(reduced, (31, 31))
    return np.maximum(0.0, reduced - bg)


def _estimate_shift(
    frame: np.ndarray, reference: np.ndarray
) -> tuple[tuple[float, float], float]:
    """
    Phase-correlation shift estimate between frame and reference.

    Downsamples and background-subtracts both images first so the correlator
    locks onto star signals rather than sky background or noise.
    The reduced-resolution shift is scaled back to full-pixel coordinates.
    Returns ((dx, dy), response).
    """
    ref_s = _make_star_image(reference)
    frm_s = _make_star_image(frame)
    win = cv2.createHanningWindow((ref_s.shape[1], ref_s.shape[0]), cv2.CV_32F)
    shift_r, response = cv2.phaseCorrelate(ref_s, frm_s, win)
    shift = (float(shift_r[0]) * _ALIGN_RF, float(shift_r[1]) * _ALIGN_RF)
    return shift, float(response)


def _is_good_shift(shift: tuple[float, float]) -> bool:
    dx, dy = shift
    return (dx ** 2 + dy ** 2) ** 0.5 < _MAX_SHIFT_PX


def _apply_shift(image: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    dx, dy = shift
    h, w = image.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(image, M, (w, h))
