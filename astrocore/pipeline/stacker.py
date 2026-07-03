"""
Live image stacker for electronically assisted astronomy.

Accumulates calibrated frames into a running mean, then on each display tick
applies sky subtraction, gamma, auto-brightness, saturation, and temporal blending
to produce a smooth, progressively improving display image.

Classes
-------
ExposureSequence      -- steps through a predefined ramp of exposure times
ConstellationStacker  -- CoM star tracking with polynomial-fit rigid registration
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

_BLEND_ALPHA    = 0.06   # blending weight for each display tick
_REGISTER_MIN_S = 5.0    # frames below this exposure are stacked without alignment


def ishow(img, lo=None, hi=None, cmap='gray'):
    from PIL import Image
    import numpy as np
    a = img.astype(float)
    lo = lo if lo is not None else a.min()
    hi = hi if hi is not None else a.max()
    norm = np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(norm).show()


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


# ── ConstellationStacker — CoM star tracking with polynomial-fit rigid registration ──

_CSTL_DWN_HGT    = 1000    # target height for star-detection downsample
_CSTL_SKY_KRNL   = 15      # effective sky kernel (pixels) in the downsampled image
_CSTL_NUM_STARS  = 20      # stars to track per frame
_CSTL_MAX_SHIFT  = 50      # max per-frame shift in full-res pixels
_CSTL_MAX_ROT_DEG = 5.0    # max per-frame rotation in degrees


def _cstl_sky(imdwn: np.ndarray, skykrnl: int) -> np.ndarray:
    """Sky background estimate matching MATLAB lines 38-41 / 99-102.

    Further downsample by rf2 = max(1, skykrnl/5), apply a 5×5 median
    (supported for uint16 by cv2.medianBlur), then upscale back.
    With skykrnl=15 → rf2=3: equivalent to a ~15-pixel sky kernel.
    """
    rf2  = max(1, round(skykrnl / 5))
    H, W = imdwn.shape[:2]
    h2   = max(1, round(H / rf2))
    w2   = max(1, round(W / rf2))
    imR2 = cv2.resize(imdwn.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
    u16  = np.clip(imR2, 0, 65535).astype(np.uint16)
    sky2 = cv2.medianBlur(u16, 5).astype(np.float32)
    return cv2.resize(sky2, (W, H), interpolation=cv2.INTER_LINEAR)


class ConstellationStacker:
    """
    Stacker using star center-of-mass tracking and polynomial-fit rigid registration.

    Reference frame: detect stars on a sky-subtracted, downsampled green channel via
    two-pass dilation-based local maxima; refine to sub-pixel centroids using
    intensity-weighted CoM in full-resolution ROI windows; score by peak/noise/offset
    and keep the best numstars.

    Subsequent frames: extract an ROI around each tracked star, normalize and cube the
    intensities to emphasise the stellar peak, compute CoM shift; decompose all
    per-star shifts into a global translation (polynomial fit evaluated at image centre)
    plus a field rotation (median tangential residual vs column position); warp each
    incoming frame into the reference coordinate system and accumulate.

    Short exposures (< min_exp) accumulate without alignment.
    """

    def __init__(
        self,
        gamma: float   = 0.7,
        min_exp: float = _REGISTER_MIN_S,
        dwn_hgt: int   = _CSTL_DWN_HGT,
        skykrnl: int   = _CSTL_SKY_KRNL,
        numstars: int  = _CSTL_NUM_STARS,
        maxshift: int  = _CSTL_MAX_SHIFT,
    ) -> None:
        self.gamma      = gamma
        self._min_exp   = float(min_exp)
        self._dwn_hgt   = int(dwn_hgt)
        self._skykrnl   = int(skykrnl)
        self._numstars  = int(numstars)
        self._maxshift  = int(maxshift)

        # Offset grids for CoM — shape (2·maxshift+1, 2·maxshift+1), centred at zero
        h     = self._maxshift
        r_off = np.arange(2 * h + 1, dtype=np.float64) - h
        c_off = np.arange(2 * h + 1, dtype=np.float64) - h
        self._rs_mesh, self._cs_mesh = np.meshgrid(r_off, c_off, indexing='ij')

        self._stack_sum:     np.ndarray | None = None
        self._frame_count:   int   = 0
        self._skipped_count: int   = 0
        self._t_accum:       float = 0.0
        self._blend:         np.ndarray | None = None
        self._imPrc:         np.ndarray | None = None
        self._sky_model:     np.ndarray | None = None
        self._sky_dirty:     bool = True

        self._star_rows: np.ndarray | None = None
        self._star_cols: np.ndarray | None = None
        self._rf:        float | None      = None
        self._H_cum:     np.ndarray        = np.eye(3, dtype=np.float64)  # cumulative warpAffine transform (new frame → reference)

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
        self._stack_sum     = None
        self._frame_count   = 0
        self._skipped_count = 0
        self._t_accum       = 0.0
        self._blend         = None
        self._imPrc         = None
        self._sky_model     = None
        self._sky_dirty     = True
        self._star_rows     = None
        self._star_cols     = None
        self._rf            = None
        self._H_cum         = np.eye(3, dtype=np.float64)

    def add_frame(self, frame: np.ndarray, exposure_s: float) -> bool:
        """
        Add a calibrated frame to the stack.

        frame      : uint16 or float32, shape (H, W) or (H, W, C)
        exposure_s : exposure duration in seconds
        Returns True if the frame was added.
        """
        img = frame.astype(np.float32)
        
        if self._stack_sum is None:
            self._stack_sum   = img.copy()
            self._frame_count = 1
            self._t_accum     = exposure_s
            self._sky_dirty   = True
            if exposure_s >= self._min_exp:
                self._setup_reference(frame)
            return True

        if self._star_rows is None:
            self._stack_sum  += img
            self._frame_count += 1
            self._t_accum    += exposure_s
            self._sky_dirty   = True
            if exposure_s >= self._min_exp:
                self._setup_reference(frame)
            return True

        registration = self._register_frame(frame)
        if registration is None:
            self._skipped_count += 1
            return False

        xshift, yshift, rot_deg = registration
        H, W = img.shape[:2]
        cx, cy = W / 2.0, H / 2.0

        # Build the incremental warpAffine matrix (maps new-frame coords → prev-frame coords).
        # This is the inverse of the forward rigid transform so warpAffine can look up
        # where each output pixel came from in the source.
        # MATLAB CCW convention (y-down): forward map is x'=cx+(x-cx)cosθ+(y-cy)sinθ,
        # y'=cy-(x-cx)sinθ+(y-cy)cosθ. Inverse map (dst→src) for warpAffine:
        rot_rad = np.radians(rot_deg)
        c_r, s_r = np.cos(rot_rad), np.sin(rot_rad)
        M_inc = np.float64([
            [c_r, -s_r, cx * (1 - c_r) + cy * s_r - xshift],
            [s_r,  c_r, cy * (1 - c_r) - cx * s_r - yshift],
            [0.0,  0.0, 1.0],
        ])

        # Accumulate: H_cum maps new-frame coords → reference-frame coords.
        # Each incoming frame is warped ONCE into the reference coordinate system
        # so interpolation error does not compound across frames.
        self._H_cum = self._H_cum @ M_inc
        warped = cv2.warpAffine(
            img, self._H_cum[:2].astype(np.float32), (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self._stack_sum  += warped
        self._frame_count += 1
        self._t_accum    += exposure_s
        self._sky_dirty   = True

        # Update tracked star positions to new frame's coordinate system.
        # Mirrors MATLAB lines 124-128: translate both, then apply rotation
        # sequentially (T.R updated first, its new value feeds T.C update).
        tan_rot  = np.tan(np.radians(rot_deg))
        new_rows = self._star_rows + yshift
        new_cols = self._star_cols + xshift
        new_rows = np.round(new_rows - (new_cols - cx) * tan_rot)
        new_cols = np.round(new_cols - (new_rows - cy) * tan_rot)
        margin   = float(self._maxshift)
        valid    = (
            (new_rows >= margin) & (new_rows < H - margin) &
            (new_cols >= margin) & (new_cols < W - margin)
        )
        self._star_rows = new_rows[valid]
        self._star_cols = new_cols[valid]

        import sys
        print(
            f"cstl: xshift={xshift:.1f} yshift={yshift:.1f} rot={rot_deg:.3f}°"
            f"  stars_kept={int(valid.sum())}",
            file=sys.stderr, flush=True,
        )
        return True

    # ── display output ────────────────────────────────────────────────────────

    def process_stack(self, sky_sub_scale: float = 1.0) -> None:
        """Expensive processing pass — call once per new frame, not every render tick."""
        imStack = self.stack
        if imStack is None:
            return

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

        self._imPrc = imPrc

    def get_display_frame(self, brightness: float = 1.0) -> np.ndarray | None:
        """Cheap temporal blend — call every render tick for smooth display."""
        if self._imPrc is None:
            return None

        if self._blend is None:
            self._blend = self._imPrc.copy()
        else:
            self._blend = (1.0 - _BLEND_ALPHA) * self._blend + _BLEND_ALPHA * self._imPrc

        return (np.clip(brightness * self._blend, 0.0, 65535.0) / 256.0).astype(np.uint8)

    # ── private helpers ───────────────────────────────────────────────────────

    def _setup_reference(self, frame: np.ndarray) -> None:
        """Build star table from the first long-exposure frame."""
        green = (frame[:, :, 1] if frame.ndim == 3 else frame).astype(np.float32)
        H, W  = green.shape

        # 3×3 median filter on green channel (removes hot pixels)
        im1g  = cv2.medianBlur(
            np.clip(green, 0, 65535).astype(np.uint16), 3
        ).astype(np.float32)

        # Downsample
        rf    = H / self._dwn_hgt
        new_w = max(1, round(W / rf))
        im1R  = cv2.resize(im1g, (new_w, self._dwn_hgt), interpolation=cv2.INTER_AREA)

        # Sky estimate at downsample resolution; upscale for full-res subtraction
        imsky_dwn  = _cstl_sky(im1R, self._skykrnl)
        imsky_full = cv2.resize(imsky_dwn, (W, H), interpolation=cv2.INTER_LINEAR)

        # Sky-subtracted downsampled star image
        im1star = np.maximum(0.0, im1R.astype(np.float32) - imsky_dwn)

        # First local-maxima pass: dilation window = search radius in downsampled space
        ssz     = int(np.ceil((2 * self._maxshift + 1) / rf))
        dilated = cv2.dilate(im1star, np.ones((ssz, ssz), dtype=np.uint8))
        immsk   = im1star == dilated
        immsk[:ssz, :]  = False;  immsk[:, :ssz]  = False
        immsk[-ssz:, :] = False;  immsk[:, -ssz:] = False

        # Second pass: Gaussian filter then re-detect to de-cluster blended stars
        # (apply second mask to first-pass result, matching MATLAB line 50)
        im1starmsk = np.where(immsk, im1star, 0.0).astype(np.float32)
        imf        = cv2.GaussianBlur(im1starmsk, (0, 0), 2.0)
        immsk2     = imf == cv2.dilate(imf, np.ones((7, 7), dtype=np.uint8))
        im1starmsk = np.where(immsk2, im1starmsk, 0.0).astype(np.float32)

        # Full-res sky-subtracted image for centroid refinement
        im1g_sky = np.maximum(0.0, im1g - imsky_full)

        # Top 2×numstars candidates by peak brightness in downsampled star image
        n_cand = 2 * self._numstars
        flat   = im1starmsk.ravel()
        n_pos  = int((flat > 0).sum())
        if n_pos < 3:
            logging.warning("cstl: fewer than 3 star candidates in reference frame")
            return
        n_cand = min(n_cand, n_pos)
        idx    = np.argpartition(flat, -n_cand)[-n_cand:]
        idx    = idx[np.argsort(flat[idx])[::-1]]
        rows_d, cols_d = np.unravel_index(idx, im1starmsk.shape)

        # Convert downsampled → full-res integer positions
        rows_full = (rows_d.astype(np.float64) * rf).round().astype(np.int32)
        cols_full = (cols_d.astype(np.float64) * rf).round().astype(np.int32)

        half    = self._maxshift
        rows_f  = rows_full.astype(np.float64)
        cols_f  = cols_full.astype(np.float64)
        weights = np.zeros(n_cand)

        for i in range(n_cand):
            R, C = int(rows_full[i]), int(cols_full[i])
            r0, r1 = R - half, R + half + 1
            c0, c1 = C - half, C + half + 1
            if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
                continue
            roi        = im1g_sky[r0:r1, c0:c1].astype(np.float64)
            roi_emp    = roi ** 3.0        # scalar multiply (invariant to CoM, matches MATLAB)
            total_mass = roi_emp.sum()
            if total_mass < 1.0:
                continue
            ysh = float((self._rs_mesh * roi_emp).sum()) / total_mass  # row offset
            xsh = float((self._cs_mesh * roi_emp).sum()) / total_mass  # col offset
            rows_f[i]  = R + round(ysh)
            cols_f[i]  = C + round(xsh)
            peak       = float(roi_emp.max())
            border     = roi_emp.copy(); border[1:-1, 1:-1] = 0.0
            noise      = max(float(border.max()), 0.1 * peak)
            weights[i] = (peak / noise) / (1.0 + (xsh**2 + ysh**2)**0.5)

        # Select best numstars by weight
        order    = np.argsort(weights)[::-1]
        selected = [i for i in order if weights[i] > 0][:self._numstars]
        if len(selected) < 3:
            logging.warning("cstl: fewer than 3 usable stars; alignment may be unreliable")

        self._star_rows = rows_f[np.array(selected)].copy()
        self._star_cols = cols_f[np.array(selected)].copy()
        self._rf        = rf
        logging.debug("cstl: reference set — %d stars tracked (rf=%.2f)", len(selected), rf)

    def _register_frame(
        self, frame: np.ndarray
    ) -> tuple[float, float, float] | None:
        """
        Estimate (xshift_col, yshift_row, rot_deg) mapping the stack → new frame.

        Per-star CoM shifts are decomposed into translation (polynomial fit evaluated
        at image centre) plus rotation (median tangential residual vs column position).
        Returns None if the frame should be rejected.
        """
        green = (frame[:, :, 1] if frame.ndim == 3 else frame).astype(np.float32)
        H, W  = green.shape

        # Median filter + sky subtract
        im2g  = cv2.medianBlur(
            np.clip(green, 0, 65535).astype(np.uint16), 3
        ).astype(np.float32)
        new_w      = max(1, round(W / self._rf))
        im2R       = cv2.resize(im2g, (new_w, self._dwn_hgt), interpolation=cv2.INTER_AREA)
        imsky_dwn  = _cstl_sky(im2R, self._skykrnl)
        imsky_full = cv2.resize(imsky_dwn, (W, H), interpolation=cv2.INTER_LINEAR)
        im2g_sky   = np.maximum(0.0, im2g - imsky_full)

        half     = self._maxshift
        xsh_list : list[float] = []
        ysh_list : list[float] = []
        rows_used: list[float] = []
        cols_used: list[float] = []

        for r, c in zip(self._star_rows, self._star_cols):
            R, C = int(round(r)), int(round(c))
            r0, r1 = R - half, R + half + 1
            c0, c1 = C - half, C + half + 1
            if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
                continue
            roi = im2g_sky[r0:r1, c0:c1].astype(np.float64)
            mx  = float(roi.max())
            if mx < 1.0:
                continue
            roi_norm   = (roi / mx) ** 3   # normalize then cube to emphasise stellar peak
            total_mass = roi_norm.sum()
            if total_mass < 1e-6:
                continue
            ysh = float((self._rs_mesh * roi_norm).sum()) / total_mass
            xsh = float((self._cs_mesh * roi_norm).sum()) / total_mass
            xsh_list.append(xsh);    ysh_list.append(ysh)
            rows_used.append(r);     cols_used.append(c)

        if len(xsh_list) < 3:
            logging.debug("cstl: SKIP — only %d usable stars", len(xsh_list))
            return None

        xsh_arr  = np.array(xsh_list)
        ysh_arr  = np.array(ysh_list)
        rows_arr = np.array(rows_used)
        cols_arr = np.array(cols_used)

        # Translation: polynomial fit of per-star shifts vs star position, evaluated
        # at image centre — disentangles translation from rotation about a non-centre axis.
        p_x    = np.polyfit(rows_arr, xsh_arr, 1)
        xshift = float(np.polyval(p_x, H / 2.0))
        p_y    = np.polyfit(cols_arr, ysh_arr, 1)
        yshift = float(np.polyval(p_y, W / 2.0))

        # Rotation: median of (tangential residual / distance from centre column).
        # For field rotation θ about centre: ysh_resid ≈ (C − cx)·θ_rad for each star.
        ysh_resid = ysh_arr - yshift
        cx        = W / 2.0
        dist      = cols_arr - cx
        valid_rot = np.abs(dist) > 0.5   # exclude stars exactly on the centre column
        if valid_rot.sum() >= 1:
            rot_deg = -float(np.median(
                np.degrees(np.arctan(ysh_resid[valid_rot] / dist[valid_rot]))
            ))
        else:
            rot_deg = 0.0

        import sys
        #print(
        #    f"cstl reg: n={len(xsh_arr)}"
        #    f"  xsh=[{float(xsh_arr.min()):.1f},{float(xsh_arr.max()):.1f}]"
        #    f"  ysh=[{float(ysh_arr.min()):.1f},{float(ysh_arr.max()):.1f}]"
        #    f"  → xshift={xshift:.2f} yshift={yshift:.2f} rot={rot_deg:.3f}°",
        #    file=sys.stderr, flush=True,
        #)

        if abs(xshift) > self._maxshift or abs(yshift) > self._maxshift:
            print(f"cstl: SKIP — shift ({xshift:.1f},{yshift:.1f}) exceeds ±{self._maxshift}",
                  file=sys.stderr, flush=True)
            return None
        if abs(rot_deg) > _CSTL_MAX_ROT_DEG:
            print(f"cstl: SKIP — rotation {rot_deg:.2f}° exceeds ±{_CSTL_MAX_ROT_DEG}°",
                  file=sys.stderr, flush=True)
            return None

        return xshift, yshift, rot_deg
    

