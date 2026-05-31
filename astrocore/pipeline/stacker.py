"""
Live image stacker for electronically assisted astronomy.

Accumulates calibrated frames into a running mean, then on each display tick
applies sky subtraction, gamma, auto-brightness, saturation, and temporal blending
to produce a smooth, progressively improving display image.

Classes
-------
ExposureSequence      -- steps through a predefined ramp of exposure times
Stacker               -- accumulates frames via incremental phase-correlation alignment
QuadrantAlignStacker  -- experimental: quadrant phase-correlation with rotation recovery
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
_REGISTER_MIN_S = 5.0     # frames below this exposure are stacked without alignment


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

        # Short exposures: accumulate without alignment — low SNR makes phase
        # correlation unreliable, and hot pixels can cause spurious rejections.
        if exposure_s < _REGISTER_MIN_S:
            self._stack_sum = self._stack_sum + img
            self._frame_count += 1
            self._t_accum += exposure_s
            self._sky_dirty = True
            self._ref_frame = img.copy()
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

    def get_display_frame(self, sky_sub_scale: float = 1.0) -> np.ndarray | None:
        """
        Process the current stack into a displayable BGR uint8 image.

        Pipeline per call:
          sky subtraction → gamma + auto-brightness → saturation → temporal blend

        sky_sub_scale multiplies the auto-computed sky coefficient, allowing
        the user to dial the subtraction up or down without affecting the ramp.

        The sky model is recomputed only when a new frame has been added.
        The temporal blend is updated on every call, smoothing the transition
        between stack updates.

        Returns None until the first frame has been added.
        """
        imStack = self.stack
        if imStack is None:
            return None

        skyco = sky_coefficient(self._t_accum) * sky_sub_scale
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


# ── QuadrantAlignStacker — experimental quadrant phase-correlation stacker ────

_QUAD_SIZE        = 1000    # resize target height for star-mask computation
_QUAD_MEDFILT     = 15      # median filter kernel (uint8, must be odd)
_QUAD_MAX_SHIFT   = 50      # max allowed shift in 1000-px space before rejection
_QUAD_MAX_ROT_DEG = 5.0     # max allowed rotation (degrees) before rejection
_QUAD_MIN_STARS   = 5       # minimum star count in each quadrant for reliable phase correlation


def _make_star_mask(frame: np.ndarray) -> np.ndarray:
    """
    Build a binary star mask from a full-resolution BGR (or mono) frame.

    Pipeline (equivalent to the MATLAB reference):
      1. Extract green channel (channel index 1 for BGR; full image if mono)
      2. Resize so height = _QUAD_SIZE, preserving aspect ratio
      3. Compute background: medianBlur(_QUAD_MEDFILT × _QUAD_MEDFILT) on uint8 copy
      4. Subtract 2 × background from the original; clamp negatives to 0
      5. Binarize: any pixel > 0 becomes 1
      6. Zero the 15-pixel corner squares to suppress edge artefacts
    """
    ch = frame[:, :, 1] if frame.ndim == 3 else frame
    H, W = ch.shape
    new_w = max(1, int(round(W * _QUAD_SIZE / H)))
    imR = cv2.resize(ch.astype(np.uint16), (new_w, _QUAD_SIZE), interpolation=cv2.INTER_AREA)

    # medianBlur with kernel > 5 requires uint8
    imR_u8 = (imR >> 8).astype(np.uint8)
    bg_u8  = cv2.medianBlur(imR_u8, _QUAD_MEDFILT)

    diff = imR_u8.astype(np.int16) - 2 * bg_u8.astype(np.int16)
    mask = (diff > 0).astype(np.uint8)

    c = 15
    mask[:c, :c] = 0;  mask[:c, -c:] = 0
    mask[-c:, :c] = 0; mask[-c:, -c:] = 0
    return mask


def _com(mask: np.ndarray) -> tuple[float, float] | None:
    """Return (col, row) centre-of-mass of a binary mask, or None if empty."""
    total = int(mask.sum())
    if total == 0:
        return None
    rows, cols = np.where(mask > 0)
    return float(cols.mean()), float(rows.mean())


def _quadrant_registration(
    imMov: np.ndarray,
    imRef: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    Estimate (tx, ty, theta) that maps imRef → imMov using 4-quadrant phase correlation.

    Algorithm
    ---------
    1. Split both masks into four quadrants (top-left, top-right, bottom-left,
       bottom-right).
    2. Phase-correlate each quadrant of imMov against the same quadrant of imRef.
    3. Identify the opposite-quadrant pair (TL+BR or TR+BL) that does NOT contain
       the quadrant with the lowest phase-correlation response.
    4. Compute the centre-of-mass of each chosen quadrant in imRef — these are the
       spatial anchors for the rigid-body fit.
    5. Fit (tx, ty, theta) from the two (anchor, shift) observations using the
       small-angle rigid-body displacement model:
           shift_x(x,y) = tx − (y − cy)·θ
           shift_y(x,y) = ty + (x − cx)·θ
    6. Reject if tx, ty, or theta exceeds the configured limits.

    Returns (tx, ty, theta) in 1000-px coordinates, or None on failure.
    """
    H, W  = imRef.shape
    mh, mw = H // 2, W // 2
    cx, cy = W / 2.0, H / 2.0

    # Quadrant slices: 0=TL, 1=TR, 2=BL, 3=BR
    slices = [
        (slice(0,  mh), slice(0,  mw)),
        (slice(0,  mh), slice(mw, W )),
        (slice(mh, H ), slice(0,  mw)),
        (slice(mh, H ), slice(mw, W )),
    ]

    shifts: list[tuple[float, float]] = []
    responses: list[float] = []
    star_counts: list[int] = []
    for rs, cs in slices:
        ref_q = imRef[rs, cs].astype(np.float32)
        mov_q = imMov[rs, cs].astype(np.float32)
        star_counts.append(int(imRef[rs, cs].sum()))
        win   = cv2.createHanningWindow((ref_q.shape[1], ref_q.shape[0]), cv2.CV_32F)
        s, r  = cv2.phaseCorrelate(ref_q, mov_q, win)
        shifts.append((float(s[0]), float(s[1])))
        responses.append(float(r))

    import sys
    print(
        f"quad stars TL={star_counts[0]} TR={star_counts[1]} "
        f"BL={star_counts[2]} BR={star_counts[3]}  "
        f"responses TL={responses[0]:.3f} TR={responses[1]:.3f} "
        f"BL={responses[2]:.3f} BR={responses[3]:.3f}  "
        f"shifts TL={shifts[0]} TR={shifts[1]} BL={shifts[2]} BR={shifts[3]}",
        file=sys.stderr, flush=True,
    )

    # Select the best opposite pair: both quadrants must have >= _QUAD_MIN_STARS stars.
    # Among valid pairs, prefer the one whose weaker quadrant has the highest response.
    def _qual(qa, qb):
        if star_counts[qa] < _QUAD_MIN_STARS or star_counts[qb] < _QUAD_MIN_STARS:
            return -1.0
        return min(responses[qa], responses[qb])

    score_a = _qual(0, 3)   # TL + BR
    score_b = _qual(1, 2)   # TR + BL

    if score_a < 0 and score_b < 0:
        print(
            f"quad: SKIP — no pair has >={_QUAD_MIN_STARS} stars in both quadrants "
            f"(A: {star_counts[0]},{star_counts[3]}  B: {star_counts[1]},{star_counts[2]})",
            file=sys.stderr, flush=True,
        )
        return None

    qa, qb = (0, 3) if score_a >= score_b else (1, 2)
    print(f"quad selected pair=Q{qa}+Q{qb}  scores A={score_a:.3f} B={score_b:.3f}",
          file=sys.stderr, flush=True)

    # Centre-of-mass anchors in imRef global coordinates
    rs_a, cs_a = slices[qa]
    rs_b, cs_b = slices[qb]
    com_a = _com(imRef[rs_a, cs_a])
    com_b = _com(imRef[rs_b, cs_b])
    if com_a is None or com_b is None:
        print(f"quad: SKIP — CoM is None for Q{qa} or Q{qb} (no stars in quadrant)",
              file=sys.stderr, flush=True)
        return None

    x1 = cs_a.start + com_a[0];  y1 = rs_a.start + com_a[1]
    x2 = cs_b.start + com_b[0];  y2 = rs_b.start + com_b[1]

    dx1, dy1 = shifts[qa]
    dx2, dy2 = shifts[qb]

    # Solve for theta using whichever axis gives the larger spatial separation
    denom_x = y2 - y1   # from x-component equations
    denom_y = x1 - x2   # from y-component equations
    if abs(denom_x) >= abs(denom_y) and abs(denom_x) > 1e-3:
        theta = (dx1 - dx2) / denom_x
    elif abs(denom_y) > 1e-3:
        theta = (dy1 - dy2) / denom_y
    else:
        print(f"quad: SKIP — anchors too close (denom_x={denom_x:.1f} denom_y={denom_y:.1f})",
              file=sys.stderr, flush=True)
        return None

    # Recover translation from the first anchor
    tx = dx1 + (y1 - cy) * theta
    ty = dy1 - (x1 - cx) * theta

    print(f"quad: tx={tx:.2f} ty={ty:.2f} theta={np.degrees(theta):.3f}°  "
          f"anchors P1=({x1:.0f},{y1:.0f}) P2=({x2:.0f},{y2:.0f})",
          file=sys.stderr, flush=True)

    # Reject if out of bounds
    if abs(tx) > _QUAD_MAX_SHIFT or abs(ty) > _QUAD_MAX_SHIFT:
        print(f"quad: SKIP — shift ({tx:.1f}, {ty:.1f}) exceeds ±{_QUAD_MAX_SHIFT}",
              file=sys.stderr, flush=True)
        return None
    if abs(np.degrees(theta)) > _QUAD_MAX_ROT_DEG:
        print(f"quad: SKIP — rotation {np.degrees(theta):.2f}° exceeds ±{_QUAD_MAX_ROT_DEG}°",
              file=sys.stderr, flush=True)
        return None

    return tx, ty, theta


def _rigid_warp_matrix(
    tx: float, ty: float, theta: float, W: int, H: int
) -> np.ndarray:
    """
    2×3 affine matrix for: translate by (tx, ty), then rotate by theta about
    the image centre (W/2, H/2).

    Derivation: apply translation first, then rotate the translated point about
    (cx, cy).  The combined forward map is:
        x' = cos_t*(x+tx-cx) - sin_t*(y+ty-cy) + cx
        y' = sin_t*(x+tx-cx) + cos_t*(y+ty-cy) + cy
    """
    cx, cy   = W / 2.0, H / 2.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.float32([
        [cos_t, -sin_t, cx*(1-cos_t) + cy*sin_t  + tx*cos_t - ty*sin_t],
        [sin_t,  cos_t, cy*(1-cos_t) - cx*sin_t  + tx*sin_t + ty*cos_t],
    ])


class QuadrantAlignStacker:
    """
    Experimental stacker that estimates per-frame translation + rotation via
    4-quadrant phase correlation on binary star masks.

    Short exposures (< _REGISTER_MIN_S) accumulate without alignment, matching
    the behaviour of Stacker.  On the first long exposure the reference mask is
    set; each subsequent long frame re-registers the accumulated stack into the
    new frame's coordinate system before adding it.

    The display pipeline (sky subtraction, gamma, saturation, temporal blend) is
    identical to Stacker.
    """

    def __init__(self, gamma: float = 0.7) -> None:
        self.gamma = gamma
        self._stack_sum: np.ndarray | None = None
        self._imRef: np.ndarray | None = None
        self._ref_frame_full: np.ndarray | None = None  # full-res float32 for fallback
        self._frame_count:   int   = 0
        self._skipped_count: int   = 0
        self._t_accum:       float = 0.0
        self._blend:     np.ndarray | None = None
        self._sky_model: np.ndarray | None = None
        self._sky_dirty: bool = True

    # ── read-only state ───────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:   return self._frame_count
    @property
    def skipped_count(self) -> int: return self._skipped_count
    @property
    def t_accum(self) -> float:     return self._t_accum

    @property
    def stack(self) -> np.ndarray | None:
        if self._stack_sum is None or self._frame_count == 0:
            return None
        return self._stack_sum / self._frame_count

    # ── frame ingestion ───────────────────────────────────────────────────────

    def reset(self) -> None:
        self._stack_sum      = None
        self._imRef          = None
        self._ref_frame_full = None
        self._frame_count    = 0
        self._skipped_count  = 0
        self._t_accum        = 0.0
        self._blend          = None
        self._sky_model      = None
        self._sky_dirty      = True

    def add_frame(self, frame: np.ndarray, exposure_s: float) -> bool:
        img = frame.astype(np.float32)

        # ── first frame ever ─────────────────────────────────────────────────
        if self._stack_sum is None:
            self._stack_sum   = img.copy()
            self._frame_count = 1
            self._t_accum     = exposure_s
            self._sky_dirty   = True
            if exposure_s >= _REGISTER_MIN_S:
                self._imRef          = _make_star_mask(frame)
                self._ref_frame_full = img.copy()
            return True

        # ── short exposure: accumulate without alignment ──────────────────────
        if self._imRef is None:
            self._stack_sum  += img
            self._frame_count += 1
            self._t_accum    += exposure_s
            self._sky_dirty   = True
            if exposure_s >= _REGISTER_MIN_S:
                self._imRef          = _make_star_mask(frame)
                self._ref_frame_full = img.copy()
            return True

        # ── long exposure: attempt rigid correction, fall back to translation ─
        imMov  = _make_star_mask(frame)
        result = _quadrant_registration(imMov, self._imRef)
        H, W   = img.shape[:2]

        if result is not None:
            tx_1k, ty_1k, theta = result
            scale = frame.shape[0] / _QUAD_SIZE
            tx, ty = tx_1k * scale, ty_1k * scale
            M = _rigid_warp_matrix(tx, ty, theta, W, H)
        elif self._ref_frame_full is not None:
            # Quadrant rotation fit failed — fall back to translation-only.
            incremental, _ = _estimate_shift(img, self._ref_frame_full)
            if not _is_good_shift(incremental):
                self._skipped_count += 1
                dx, dy = incremental
                print(f"quad fallback: SKIP shift=({dx:.1f},{dy:.1f})",
                      file=sys.stderr, flush=True)
                return False
            dx, dy = incremental
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            print(f"quad fallback: translation-only shift=({dx:.1f},{dy:.1f})",
                  file=sys.stderr, flush=True)
        else:
            self._skipped_count += 1
            return False

        self._stack_sum = cv2.warpAffine(
            self._stack_sum, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self._stack_sum      += img
        self._frame_count    += 1
        self._t_accum        += exposure_s
        self._sky_dirty       = True
        self._imRef           = imMov
        self._ref_frame_full  = img.copy()
        return True

    # ── display output ────────────────────────────────────────────────────────

    def get_display_frame(self, sky_sub_scale: float = 1.0) -> np.ndarray | None:
        imStack = self.stack
        if imStack is None:
            return None

        skyco = sky_coefficient(self._t_accum) * sky_sub_scale
        satco = saturation_coefficient(self._t_accum)

        if self._sky_dirty or self._sky_model is None:
            self._sky_model = fit_sky_model(imStack)
            self._sky_dirty = False

        imPrc = np.clip(imStack - skyco * self._sky_model, 0.0, None)
        gn    = auto_brightness(imStack, self.gamma, skyco)
        imPrc = np.clip(gamma_correct(imPrc, self.gamma) * gn, 0.0, 65535.0)

        if imPrc.ndim == 3:
            imPrc = adjust_saturation(imPrc, satco)

        if self._blend is None:
            self._blend = imPrc.copy()
        else:
            self._blend = (1.0 - _BLEND_ALPHA) * self._blend + _BLEND_ALPHA * imPrc

        return (np.clip(self._blend, 0.0, 65535.0) / 256.0).astype(np.uint8)
