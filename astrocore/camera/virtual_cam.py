"""
VirtualCamera — a Camera implementation that replays saved TIFF frames.

Use this for offline testing and demos when no physical camera is available.
Point it at a folder of canonically named frames (see naming.py) and it
behaves exactly like a real camera:

  • start_exposure() starts a timer (accelerated by t_accel).
  • exposure_status flips IDLE → WORKING → SUCCESS when the timer expires.
  • read_frame() returns the next frame whose exposure matches the current
    setting, cycling back to the start when the list is exhausted.
  • If no frame matches the current exposure_time, any available frame is
    returned (graceful fallback, same as the original MATLAB virtualCam).

Typical use::

    with VirtualCamera("/data/test_frames", camera_id=3, t_accel=10.0) as cam:
        cam.exposure_time = 2.0
        grabber = FrameGrabber(cam)
        while running:
            result = grabber.grab_frame(dark=False, flat=False, demosaic=False)
            if result.status == GrabStatus.SUCCESS:
                display(result.frame)
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .base import (
    Camera,
    CameraInfo,
    ConnectionError,
    ExposureStatus,
    Frame,
    FrameMeta,
    PersistentMeta,
)
from .catalog import scan_folder


class VirtualCamera(Camera):
    """
    Replays saved TIFF frames as if they came from live hardware.

    Parameters
    ----------
    folder :
        Directory containing canonically named .tif files.
    camera_id :
        If provided, only files from this camera_id are used.
        If None, all files in the folder are included and camera_id is
        inferred from the first file found.
    bayer_pattern :
        Bayer mosaic pattern string (e.g. 'RGGB'). Pass None for mono.
        Not encoded in the filename, so must be supplied here if needed.
    t_accel :
        Time acceleration factor. An exposure_time of 2.0 s with t_accel=10
        completes in 0.2 s of wall-clock time. Use large values (e.g. 1000)
        in automated tests so exposures finish in microseconds.
    """

    def __init__(
        self,
        folder: str | Path,
        camera_id: int | None = None,
        bayer_pattern: str | None = None,
        t_accel: float = 3.0,
    ) -> None:
        super().__init__()
        self._folder = Path(folder)
        self._filter_camera_id = camera_id
        self._bayer_pattern = bayer_pattern
        self._t_accel = t_accel

        # Public camera settings (mirrors ZwoAsiCamera API so FrameGrabber
        # can get/set exposure via hasattr checks)
        self.exposure_time: float = 1.0   # seconds
        self.gain: int = 0
        self.offset: int = 0

        self._file_table: pd.DataFrame = pd.DataFrame()
        self._cursors: dict[int, int] = {}   # exposure_us → next row index
        self._t_exp: float = float("inf")    # monotonic time when exposure completes
        self._image_shape: tuple = (0, 0)
        self._camera_id: int = camera_id or 0
        self._pixel_size_um: float = 0.0

    # -- connection lifecycle -----------------------------------------------

    def connect(self) -> None:
        table = scan_folder(self._folder)

        if self._filter_camera_id is not None:
            table = table[table.camera_id == self._filter_camera_id].reset_index(drop=True)

        if len(table) == 0:
            suffix = (
                f" for camera_id={self._filter_camera_id}"
                if self._filter_camera_id is not None
                else ""
            )
            raise ConnectionError(
                f"No canonical .tif files found in {self._folder}{suffix}. "
                "Files must follow the naming convention: "
                "C{{id}}_{{filter}}_{{exposure}}us_{{temp}}C_{{index}}.tif"
            )

        self._file_table = table.sort_values("frame_index").reset_index(drop=True)
        self._camera_id = int(table.camera_id.iloc[0])

        # Read first image once to learn sensor dimensions and extract metadata
        import json as _json
        import tifffile
        first_path = str(table.path.iloc[0])
        with tifffile.TiffFile(first_path) as tif:
            first = tif.asarray()
            desc = tif.pages[0].description or ""
        self._image_shape = first.shape
        try:
            saved = _json.loads(desc)
            self._pixel_size_um = float(saved.get("pixel_size_um", 0.0))
        except (ValueError, TypeError):
            self._pixel_size_um = 0.0

        self._cursors = {}
        self._t_exp = float("inf")
        self._connected = True

    def disconnect(self) -> None:
        self._file_table = pd.DataFrame()
        self._cursors = {}
        self._t_exp = float("inf")
        self.meta = PersistentMeta()
        self._connected = False

    # -- hardware info -------------------------------------------------------

    @property
    def info(self) -> CameraInfo:
        self._require_connected()
        h, w = self._image_shape[:2]
        return CameraInfo(
            model="VirtualCamera",
            usb_index=0,
            sensor_width_px=w,
            sensor_height_px=h,
            pixel_size_um=self._pixel_size_um,
            bit_depth=16,
            camera_id=self._camera_id,
            bayer_pattern=self._bayer_pattern,
        )

    # -- non-blocking imaging primitives ------------------------------------

    def start_exposure(self, is_dark: bool = False) -> None:
        self._require_connected()
        self._t_exp = time.monotonic() + self.exposure_time / self._t_accel

    @property
    def exposure_status(self) -> ExposureStatus:
        if self._t_exp == float("inf"):
            return ExposureStatus.IDLE
        if time.monotonic() >= self._t_exp:
            return ExposureStatus.SUCCESS
        return ExposureStatus.WORKING

    def read_frame(self) -> Frame:
        self._require_connected()

        # Match frames by exposure; fall back to full table if none match
        exp_us = round(self.exposure_time * 1_000_000)
        matching = self._file_table[self._file_table.exposure_us == exp_us]
        if len(matching) == 0:
            matching = self._file_table
            exp_us = -1   # sentinel so fallback set has its own cursor

        cursor = self._cursors.get(exp_us, 0)
        if cursor >= len(matching):
            cursor = 0
        self._cursors[exp_us] = cursor + 1

        row = matching.iloc[cursor].to_dict()

        import json as _json
        import tifffile
        with tifffile.TiffFile(str(row["path"])) as tif:
            data = tif.asarray()
            desc = tif.pages[0].description or ""
        try:
            saved = _json.loads(desc)
        except (ValueError, TypeError):
            saved = {}

        self._t_exp = float("inf")   # buffer consumed → IDLE
        return Frame(data=data, meta=self._build_meta(row, saved))

    def abort(self) -> None:
        self._t_exp = float("inf")

    # -- internals ----------------------------------------------------------

    def _build_meta(self, row: dict, saved: dict | None = None) -> FrameMeta:
        saved = saved or {}
        temp = row.get("temperature_c")
        if temp is None or (isinstance(temp, float) and math.isnan(temp)):
            temp = None

        return FrameMeta(
            camera_id=self._camera_id,
            camera_model="VirtualCamera",
            timestamp=datetime.now(tz=timezone.utc),
            exposure_seconds=float(row["exposure_s"]),
            gain=self.gain,
            offset=self.offset,
            binning=1,
            temperature_c=temp,
            bayer_pattern=self._bayer_pattern,
            filter_id=int(row["filter_id"]),
            pixel_size_um=self._pixel_size_um,
            telescope_description=self.meta.telescope_description,
            focal_length_mm=self.meta.focal_length_mm,
            RA=self.meta.RA,
            Dec=self.meta.Dec,
            Lat=self.meta.Lat,
            Lon=self.meta.Lon,
            roi_x=saved.get("roi_x"),
            roi_y=saved.get("roi_y"),
            roi_width=saved.get("roi_width"),
            roi_height=saved.get("roi_height"),
        )
