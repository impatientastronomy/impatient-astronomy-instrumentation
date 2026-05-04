"""
FrameGrabber — non-blocking frame acquisition with integrated calibration pipeline.

Sits between a camera driver and your application loop. The camera knows how to
talk to hardware. FrameGrabber knows when to start exposures, when to collect
them, and how to calibrate the raw data before handing it to you.

Typical use::

    grabber = FrameGrabber(camera)
    grabber.cal_path = "/data/calibration"
    grabber.pattern = "GRBG"      # effective Bayer pattern for current flip/bin

    while running:
        result = grabber.grab_frame(exposure_us=2_000_000)

        if result.status == GrabStatus.SUCCESS:
            display(result.frame)
        elif result.status == GrabStatus.TIMEOUT:
            log.warning(result.message)
        # WORKING and STARTED: do nothing, loop again

Calibration is lazy — files are read from cal_path on first use and cached in
imDark, imFlat, and imDPC. The dark is automatically refreshed when the sensor
temperature drifts to a better match in the dark library.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Camera, ExposureStatus, Frame
from .catalog import scan_folder


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class GrabStatus(Enum):
    STARTED = auto()   # exposure just kicked off — loop again
    WORKING = auto()   # exposure in progress — loop again
    SUCCESS = auto()   # frame ready in result.frame
    TIMEOUT = auto()   # took too long — exposure was aborted
    FAILED  = auto()   # camera reported a hardware failure


@dataclass
class GrabResult:
    status: GrabStatus
    frame: Frame | None = None
    raw_frame: Frame | None = None   # pre-calibration; populated on SUCCESS
    message: str = ""


# ---------------------------------------------------------------------------
# FrameGrabber
# ---------------------------------------------------------------------------

class FrameGrabber:
    """
    Non-blocking frame acquisition state machine with calibration pipeline.

    Call grab_frame() on every tick of your main loop. It returns immediately
    and tells you what state the camera is in.

    Calibration properties
    ----------------------
    cal_path :
        Root folder for calibration data, with subfolders darks/, flats/, DPC/.
    pattern :
        Effective Bayer demosaic pattern after flip/bin transforms (e.g. 'GRBG').
        Distinct from the sensor's native bayer_pattern: if the image is flipped,
        this must be updated accordingly. None disables demosaicing.
    imDark :
        Loaded dark frame (uint16 ndarray). Scalar 0 means not yet loaded.
    imFlat :
        Loaded flat field (float32 ndarray, normalized so 1000 = unity gain).
        Scalar 0 means not yet loaded.
    imDPC :
        Loaded dead-pixel mask (bool ndarray). True = dead pixel.
        Scalar 0 means not yet loaded.
    current_dark :
        Filename of the dark frame currently in imDark. The DPC file for this
        dark lives at cal_path/DPC/{current_dark}.
    dark_table :
        DataFrame of available dark frames. Populated lazily from cal_path/darks/
        on first request; can also be assigned directly.
    """

    def __init__(self, camera: Camera, timeout_margin: float = 5.0) -> None:
        self._camera = camera
        self._timeout_margin = timeout_margin
        self._deadline: datetime | None = None
        self._exposure_us: int | None = None

        self.cal_path: Path | str | None = None
        self.pattern: str | None = None
        self.imDark: np.ndarray | int = 0
        self.imFlat: np.ndarray | int = 0
        self.imDPC: np.ndarray | int = 0
        self.current_dark: str = ""
        self.current_dark_flip: int = 0  # flip int of the dark file currently in imDark
        self.dark_table: pd.DataFrame = pd.DataFrame()

    @property
    def cam(self) -> Camera:
        return self._camera

    @cam.setter
    def cam(self, camera: Camera) -> None:
        self._camera = camera

    # -- main interface -----------------------------------------------------

    def grab_frame(
        self,
        exposure_us: int | None = None,
        dark: bool = True,
        flat: bool = True,
        dpc: bool = False,
        demosaic: bool = True,
    ) -> GrabResult:
        """
        Call every loop tick. Returns immediately with current state.

        Parameters
        ----------
        exposure_us :
            Desired exposure in microseconds. If provided and different from the
            camera's current setting, aborts/discards the current exposure,
            reconfigures the camera, invalidates imDark, and starts fresh.
            If omitted, uses whatever exposure the camera is already set to.
        dark :
            Subtract the best-matching dark from cal_path/darks/.
        flat :
            Apply flat-field correction from cal_path/flats/.
        dpc :
            Replace dead pixels (per imDPC mask) with 2nd-neighbor mean.
            Requires dark to have been loaded first (DPC file shares dark's name).
        demosaic :
            Debayer the image using self.pattern. No-op if pattern is None.
        """
        # -- Handle exposure change --
        if exposure_us is not None:
            current_us = self._camera_exposure_us()
            if current_us is None or exposure_us != current_us:
                status = self._camera.exposure_status
                if status == ExposureStatus.SUCCESS:
                    self._camera.read_frame()   # drain completed frame
                self._camera.abort()
                self._set_camera_exposure(exposure_us)
                self._exposure_us = exposure_us
                self.imDark = 0                 # dark depends on exposure time
                self._start_exposure()
                return GrabResult(status=GrabStatus.STARTED)

        # -- Normal state machine --
        status = self._camera.exposure_status

        if status in (ExposureStatus.IDLE, ExposureStatus.FAILED):
            self._start_exposure()
            return GrabResult(status=GrabStatus.STARTED)

        if status == ExposureStatus.WORKING:
            if self._deadline is not None and _now() > self._deadline:
                self._camera.abort()
                exp_s = (self._exposure_us or 0) / 1_000_000
                msg = f"Exposure timed out after {exp_s + self._timeout_margin:.1f} s"
                return GrabResult(status=GrabStatus.TIMEOUT, message=msg)
            return GrabResult(status=GrabStatus.WORKING)

        if status == ExposureStatus.SUCCESS:
            raw_frame = self._camera.read_frame()
            self._start_exposure()              # immediately queue next exposure

            imR = raw_frame.data.copy().astype(np.uint16)
            meta = raw_frame.meta

            if dark:
                imR = self._pipeline_dark(imR, meta)
            if dpc:
                imR = self._pipeline_dpc(imR, meta)
            if flat:
                imR = self._pipeline_flat(imR, meta)

            if demosaic and self.pattern is not None:
                imP = _debayer(imR, self.pattern)
            else:
                imP = imR

            return GrabResult(
                status=GrabStatus.SUCCESS,
                frame=Frame(data=imP, meta=meta),
                raw_frame=raw_frame,
            )

        return GrabResult(status=GrabStatus.FAILED, message="Unexpected exposure status.")

    def reset(self) -> None:
        """
        Abort any in-progress exposure and discard the result.

        Does not clear calibration data (imDark, imFlat, imDPC).
        """
        already_done = self._camera.exposure_status == ExposureStatus.SUCCESS
        self._camera.abort()
        if already_done:
            self._camera.read_frame()
        self._deadline = None

    # -- calibration pipeline -----------------------------------------------

    def _pipeline_dark(self, imR: np.ndarray, meta) -> np.ndarray:
        """Lazy-load best-matching dark, refresh if temperature drifted, subtract."""
        if _is_array(self.imDark):
            # Already loaded — check if temperature has drifted to a better match.
            self._ensure_dark_table()
            best = self._find_best_dark_row(meta)
            if best is not None and best["filename"] != self.current_dark:
                raw = _read_calibration(best["path"], meta, np.uint16)
                self.imDark = _apply_flip_correction(raw, best["flip"], _flip_int(meta))
                self.current_dark = best["filename"]
                self.current_dark_flip = best["flip"]
                self.imDPC = 0
            return _subtract_dark(imR, self.imDark)

        # Not yet loaded — find best match in the dark library.
        self._ensure_dark_table()
        best = self._find_best_dark_row(meta)
        if best is None:
            _exit_missing_cal("dark", self.cal_path, meta)
        raw = _read_calibration(best["path"], meta, np.uint16)
        self.imDark = _apply_flip_correction(raw, best["flip"], _flip_int(meta))
        self.current_dark = best["filename"]
        self.current_dark_flip = best["flip"]
        self.imDPC = 0
        return _subtract_dark(imR, self.imDark)

    def _pipeline_dpc(self, imR: np.ndarray, meta) -> np.ndarray:
        """Lazy-load dead-pixel mask and replace bad pixels with 2nd-neighbor mean."""
        if not _is_array(self.imDPC):
            if not self.current_dark or self.cal_path is None:
                return imR
            dpc_path = Path(self.cal_path) / "DPC" / self.current_dark
            raw = _read_calibration(dpc_path, meta, np.uint8)
            mask = _apply_flip_correction(raw, self.current_dark_flip, _flip_int(meta)).astype(bool)
            # Zero the 2-pixel border so 2nd-neighbor lookups are always in bounds
            mask[:2, :]  = False
            mask[-2:, :] = False
            mask[:, :2]  = False
            mask[:, -2:] = False
            self.imDPC = mask

        return _fix_dead_pixels_mask(imR, self.imDPC)

    def _pipeline_flat(self, imR: np.ndarray, meta) -> np.ndarray:
        """Lazy-load matching flat (camera + filter + bin) and apply per-pixel scaling."""
        if not _is_array(self.imFlat):
            if self.cal_path is None:
                _exit_missing_cal("flat", self.cal_path, meta)
            flat_table = scan_folder(Path(self.cal_path) / "flats")
            matching = flat_table[
                (flat_table.camera_id == meta.camera_id) &
                (flat_table.filter_id == meta.filter_id) &
                (flat_table["bin"] == meta.binning)
            ]
            if len(matching) == 0:
                _exit_missing_cal("flat", self.cal_path, meta)
            best = matching.iloc[0]
            raw = _read_calibration(best["path"], meta, np.float32)
            self.imFlat = _apply_flip_correction(raw, best["flip"], _flip_int(meta))

        return _apply_flat_field(imR, self.imFlat)

    # -- dark library helpers -----------------------------------------------

    def _ensure_dark_table(self) -> None:
        if len(self.dark_table) == 0 and self.cal_path is not None:
            self.dark_table = scan_folder(Path(self.cal_path) / "darks")

    def _find_best_dark_row(self, meta) -> dict | None:
        if len(self.dark_table) == 0:
            return None
        science_us = round(meta.exposure_seconds * 1_000_000)
        base = (
            (self.dark_table.camera_id == meta.camera_id) &
            (self.dark_table["bin"] == meta.binning)
        )
        candidates = self.dark_table[base & (self.dark_table.exposure_us == science_us)]
        if len(candidates) == 0:
            # Fallback: use 100µs bias-style dark with matching camera/bin
            candidates = self.dark_table[base & (self.dark_table.exposure_us == 100)]
        if len(candidates) == 0:
            return None
        if meta.temperature_c is not None and not candidates.temperature_c.isna().all():
            idx = (candidates.temperature_c - meta.temperature_c).abs().idxmin()
            return candidates.loc[idx].to_dict()
        return candidates.iloc[0].to_dict()

    # -- camera helpers -----------------------------------------------------

    def _start_exposure(self) -> None:
        self._camera.start_exposure()
        exposure_s = (self._exposure_us or 0) / 1_000_000
        self._deadline = _now() + timedelta(seconds=exposure_s + self._timeout_margin)

    def _camera_exposure_us(self) -> int | None:
        cam = self._camera
        if hasattr(cam, "exposure_time"):
            return round(cam.exposure_time * 1_000_000)
        if hasattr(cam, "exposure_seconds"):
            return round(cam.exposure_seconds * 1_000_000)
        return None

    def _set_camera_exposure(self, exposure_us: int) -> None:
        cam = self._camera
        exposure_s = exposure_us / 1_000_000
        if hasattr(cam, "exposure_time"):
            cam.exposure_time = exposure_s
        elif hasattr(cam, "exposure_seconds"):
            cam.exposure_seconds = exposure_s


# ---------------------------------------------------------------------------
# Module-level pure functions (independently testable)
# ---------------------------------------------------------------------------

def _subtract_dark(imR: np.ndarray, imDark: np.ndarray) -> np.ndarray:
    """Subtract dark from image. Result is clamped to [0, 65535] uint16."""
    result = imR.astype(np.int32) - imDark.astype(np.int32)
    return np.clip(result, 0, 65535).astype(np.uint16)


def _fix_dead_pixels_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace each True pixel in mask with the mean of its 4 cardinal 2nd-neighbors.

    Using distance-2 neighbors preserves Bayer channel consistency: for any
    pixel at [r,c], the same-color neighbors in the mosaic are at [r±2, c±2].
    The 2-pixel border of mask must be False (ensured by _pipeline_dpc).
    """
    result = data.copy()
    rows, cols = data.shape[:2]
    dead_rs, dead_cs = np.where(mask)
    for r, c in zip(dead_rs, dead_cs):
        neighbors = [
            data[r + dr, c + dc]
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
            if 0 <= r + dr < rows and 0 <= c + dc < cols
        ]
        if neighbors:
            result[r, c] = np.uint16(np.mean(neighbors))
    return result


def _apply_flat_field(imR: np.ndarray, imFlat: np.ndarray) -> np.ndarray:
    """
    Apply flat-field correction. imFlat values are per-pixel multipliers
    normalized to 1000 (1000 = unity gain). Result is clamped to uint16.
    """
    result = imR.astype(np.float32) * imFlat.astype(np.float32) / 1000.0
    return np.clip(result, 0, 65535).astype(np.uint16)


def _debayer(data: np.ndarray, pattern: str) -> np.ndarray:
    """
    Debayer a raw mosaic image using the given Bayer pattern. Returns BGR uint16.

    Requires opencv-python: pip install opencv-python
    """
    import cv2
    pattern_map = {
        "RGGB": cv2.COLOR_BAYER_RG2BGR,
        "GRBG": cv2.COLOR_BAYER_GR2BGR,
        "GBRG": cv2.COLOR_BAYER_GB2BGR,
        "BGGR": cv2.COLOR_BAYER_BG2BGR,
    }
    code = pattern_map.get(pattern.upper())
    if code is None:
        raise ValueError(
            f"Unknown Bayer pattern: {pattern!r}. Expected one of: {list(pattern_map)}"
        )
    return cv2.cvtColor(data, code)


def _read_calibration(path: Path, meta, dtype: type) -> np.ndarray:
    """
    Read a TIFF calibration file, crop to the ROI in meta, and cast to dtype.

    Requires tifffile: pip install tifffile
    """
    import tifffile
    data = tifffile.imread(str(path))
    return _crop_to_roi(data, meta).astype(dtype)


def _crop_to_roi(data: np.ndarray, meta) -> np.ndarray:
    """Crop data to the ROI described in FrameMeta. No-op if ROI fields are None."""
    if meta.roi_x is None:
        return data
    return data[
        meta.roi_y : meta.roi_y + meta.roi_height,
        meta.roi_x : meta.roi_x + meta.roi_width,
    ]


def _is_array(arr) -> bool:
    """True if arr is a numpy array (loaded), False if it is still the scalar 0."""
    return isinstance(arr, np.ndarray)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


_FLIP_STR_TO_INT: dict[str, int] = {"NONE": 0, "HORIZ": 1, "VERT": 2, "BOTH": 3}


def _flip_int(meta) -> int:
    """Return the flip setting from FrameMeta as an integer (0–3)."""
    return _FLIP_STR_TO_INT.get(meta.flip or "NONE", 0)


def _apply_flip_correction(data: np.ndarray, cal_flip: int, cam_flip: int) -> np.ndarray:
    """
    Flip cal data so its orientation matches the camera's current flip setting.

    Uses XOR of the two flip integers: each bit represents an axis
    (bit 0 = horizontal, bit 1 = vertical).  The correction is applied
    in-place on a copy.
    """
    correction = int(cal_flip) ^ int(cam_flip)
    if correction == 0:
        return data
    result = data.copy()
    if correction & 1:
        result = np.fliplr(result)
    if correction & 2:
        result = np.flipud(result)
    return result


def _exit_missing_cal(kind: str, cal_path, meta) -> None:
    """Print a descriptive error and terminate the process."""
    folder = Path(cal_path) / f"{kind}s" if cal_path else "<cal_path not set>"
    if kind == "dark":
        science_us = round(meta.exposure_seconds * 1_000_000)
        pattern = (
            f"Cam{meta.camera_id}Bin{meta.binning}Flip*Filt*_{science_us}us_*C_*.tif"
            f"  (or 100us fallback)"
        )
    else:
        pattern = (
            f"Cam{meta.camera_id}Bin{meta.binning}Flip*Filt{meta.filter_id}_*us_*C_*.tif"
        )
    print(
        f"\nCalibration error: no {kind} file found.\n"
        f"  Folder  : {folder}\n"
        f"  Pattern : {pattern}",
        file=sys.stderr,
    )
    sys.exit(1)
