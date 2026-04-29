"""
ZWO ASI camera driver.

Requires the `zwoasi` package and the ZWO ASI SDK shared library:
  - macOS:  libASICamera2.dylib
  - Linux:  libASICamera2.so  (including Raspberry Pi)
  - SDK download: https://astronomy-imaging-camera.com/software-drivers

Provide the library path via the ASI_LIB environment variable, or pass it
explicitly to ZwoAsiCamera(library_path=...).

  pip install zwoasi

First-time setup
----------------
Each physical camera should have a unique ID (1–8) written to its EEPROM
once, so the software can identify it reliably regardless of USB port order::

    with ZwoAsiCamera(index=0) as cam:
        cam.set_camera_id(1)   # write once; persists until overwritten
"""

import ctypes
import os
import platform
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import zwoasi as asi
except ImportError as exc:
    raise ImportError(
        "zwoasi package not found. Install with: pip install zwoasi"
    ) from exc

from ..config import camera_config as _cam_cfg
from .base import (
    Camera,
    CameraError,
    CameraInfo,
    ConfigurationError,
    ConnectionError,
    ExposureError,
    ExposureStatus,
    Frame,
    FrameMeta,
    PersistentMeta,
)

_STATUS_MAP: dict[int, ExposureStatus] = {
    asi.ASI_EXP_IDLE:    ExposureStatus.IDLE,
    asi.ASI_EXP_WORKING: ExposureStatus.WORKING,
    asi.ASI_EXP_SUCCESS: ExposureStatus.SUCCESS,
    asi.ASI_EXP_FAILED:  ExposureStatus.FAILED,
}

_BAYER_MAP: dict[int, str] = {
    0: "RGGB",
    1: "BGGR",
    2: "GRBG",
    3: "GBRG",
}

class FlipMode(Enum):
    """Mirror of ASI_FLIP_STATUS in the ZWO SDK."""
    NONE  = 0
    HORIZ = 1
    VERT  = 2
    BOTH  = 3


# The user-assigned camera number lives in the last byte of the 8-byte EEPROM ID.
_CAMERA_ID_BYTE = 7


def list_cameras(library_path: str | None = None) -> list[dict]:
    """
    Return info dicts for all connected ASI cameras, sorted by camera_id.

    Cameras whose EEPROM ID has not been set (camera_id == 0) appear last.
    """
    _init_sdk(library_path)
    cameras = []
    for i in range(asi.get_num_cameras()):
        cam = asi.Camera(i)
        hw_id = _read_camera_id(cam)
        prop = cam.get_camera_property()
        cameras.append({"usb_index": i, "camera_id": hw_id, "model": prop["Name"]})
        cam.close()
    return sorted(cameras, key=lambda c: (c["camera_id"] == 0, c["camera_id"]))


_SDK_DIR = Path(__file__).parent / "asi_sdk" / "lib"

_PLATFORM_SUBDIR: dict[tuple[str, str], tuple[str, str]] = {
    ("darwin", "arm64"):  ("mac_arm64", "libASICamera2.dylib"),
    ("darwin", "x86_64"): ("mac",       "libASICamera2.dylib"),
    ("linux",  "aarch64"):("armv8",     "libASICamera2.so"),
    ("linux",  "x86_64"): ("x64",       "libASICamera2.so"),
}


def _find_asi_library() -> str | None:
    """Return path to the bundled ASI SDK library for the current platform, or None."""
    sys_key = sys.platform.split("-")[0]   # "darwin" or "linux"
    arch = platform.machine().lower()
    entry = _PLATFORM_SUBDIR.get((sys_key, arch))
    if entry is None:
        return None
    subdir, filename = entry
    lib = _SDK_DIR / subdir / filename
    return str(lib) if lib.exists() else None


def _init_sdk(library_path: str | None) -> None:
    path = (
        library_path
        or os.environ.get("ASI_LIB")
        or os.environ.get("ASI_SDK_PATH")
        or _find_asi_library()
    )
    if not path:
        raise CameraError(
            "ZWO ASI SDK library not found. "
            "Set ASI_LIB to the library path, or place the SDK in "
            f"{_SDK_DIR.parent / 'asi_sdk'}."
        )
    if not Path(path).exists():
        raise CameraError(f"ASI SDK library not found at: {path}")
    asi.init(path)



def _read_camera_id(cam: "asi.Camera") -> int:
    """Read the user-assigned ID from the last byte of the camera's EEPROM."""
    try:
        id_struct = asi._ASI_ID()
        asi.zwolib.ASIGetID(cam.id, id_struct)
        # id_struct.id is a c_char array — read raw bytes via the memory buffer
        # before zwoasi decodes it as a null-terminated string.
        raw = (ctypes.c_ubyte * 8).from_buffer_copy(id_struct)
        return int(raw[_CAMERA_ID_BYTE])
    except Exception:
        return 0


class ZwoAsiCamera(Camera):
    """
    Driver for ZWO ASI cameras via the `zwoasi` Python package.

    Parameters
    ----------
    index:
        Zero-based USB enumeration index. Use list_cameras() to discover
        which physical camera is at which index, then sort by camera_id
        for a stable ordering.
    library_path:
        Path to the ASI SDK shared library. Falls back to ASI_LIB env var.
    """

    def __init__(self, index: int = 0, library_path: str | None = None) -> None:
        super().__init__()
        self._index = index
        self._library_path = library_path
        self._cam: asi.Camera | None = None
        self._info: CameraInfo | None = None
        self._meta_snapshot: FrameMeta | None = None

    # -- identity (available without connecting) -----------------------------

    @property
    def index(self) -> int:
        """Zero-based USB enumeration index passed at construction."""
        return self._index

    @property
    def library_path(self) -> str | None:
        """SDK library path passed at construction (or None to use ASI_LIB env var)."""
        return self._library_path

    # -- connection lifecycle ------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        try:
            _init_sdk(self._library_path)
            num = asi.get_num_cameras()
            if num == 0:
                raise ConnectionError("No ZWO ASI cameras detected.")
            if self._index >= num:
                raise ConnectionError(
                    f"Camera index {self._index} requested but only {num} camera(s) found."
                )
            self._cam = asi.Camera(self._index)
            self._cam.set_image_type(asi.ASI_IMG_RAW16)
            self._cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, 70)
            self._cam.set_control_value(asi.ASI_EXPOSURE, 100_000)  # 0.1 s default
            self._info = self._build_info()
            self._connected = True
            self._apply_camera_config()
        except asi.ZWO_Error as exc:
            raise ConnectionError(
                f"Failed to connect to ASI camera {self._index}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if not self._connected or self._cam is None:
            return
        try:
            self._cam.close()
        except asi.ZWO_Error:
            pass
        finally:
            self._cam = None
            self._connected = False
            self.meta = PersistentMeta()

    # -- hardware info -------------------------------------------------------

    @property
    def info(self) -> CameraInfo:
        self._require_connected()
        return self._info  # type: ignore[return-value]

    def _build_info(self) -> CameraInfo:
        assert self._cam is not None
        p = self._cam.get_camera_property()
        is_color = bool(p.get("IsColorCam"))
        bayer = _BAYER_MAP.get(p.get("BayerPattern", -1)) if is_color else None
        return CameraInfo(
            model=p["Name"],
            usb_index=self._index,
            sensor_width_px=p["MaxWidth"],
            sensor_height_px=p["MaxHeight"],
            pixel_size_um=p["PixelSize"],
            bit_depth=p["BitDepth"],
            camera_id=_read_camera_id(self._cam),
            has_cooler=bool(p["IsCoolerCam"]),
            bayer_pattern=bayer,
        )

    # -- camera ID management -----------------------------------------------

    def set_camera_id(self, camera_id: int) -> None:
        """
        Write a user-assigned ID (1–8) to the camera's EEPROM.

        This is a one-time setup step. The value persists across power cycles
        and USB reconnects. Call list_cameras() afterwards to verify.
        """
        self._require_connected()
        assert self._cam is not None
        if not (1 <= camera_id <= 8):
            raise ConfigurationError("camera_id must be between 1 and 8.")
        try:
            id_bytes = bytearray(8)
            id_bytes[_CAMERA_ID_BYTE] = camera_id
            self._cam.set_id(bytes(id_bytes))
            # Refresh info so the new ID is reflected immediately
            self._info = self._build_info()
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to write camera ID: {exc}") from exc

    # -- ROI and flip controls -----------------------------------------------

    def get_roi(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) of the current region of interest."""
        self._require_connected()
        assert self._cam is not None
        x, y, width, height = self._cam.get_roi()
        return x, y, width, height

    def set_roi(
        self,
        *,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
        bin: int | None = None,
    ) -> None:
        """
        Set region of interest and start position.

        Unspecified width/height default to full sensor size divided by the
        active (or requested) bin factor.  Width must be a multiple of 8;
        height a multiple of 2 (SDK constraint).  Start position must be
        even on both axes to preserve Bayer mosaic color order.
        """
        self._require_connected()
        assert self._cam is not None
        _, _, current_bin, _ = self._cam.get_roi_format()
        if bin is None:
            bin = current_bin
        if width is None:
            width = (self._info.sensor_width_px // bin) & ~7   # type: ignore[union-attr]
        if height is None:
            height = (self._info.sensor_height_px // bin) & ~1  # type: ignore[union-attr]
        if width % 8 != 0:
            raise ConfigurationError(f"ROI width must be a multiple of 8 (got {width}).")
        if height % 2 != 0:
            raise ConfigurationError(f"ROI height must be a multiple of 2 (got {height}).")
        if x % 2 != 0:
            raise ConfigurationError(f"ROI start x must be a multiple of 2 (got {x}).")
        if y % 2 != 0:
            raise ConfigurationError(f"ROI start y must be a multiple of 2 (got {y}).")
        try:
            self._cam.set_roi(width=width, height=height, bins=bin)
            self._cam.set_roi_start_position(x, y)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set ROI: {exc}") from exc

    @property
    def flip(self) -> FlipMode:
        """Current image flip setting."""
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_FLIP)
            return FlipMode(val)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read flip mode: {exc}") from exc

    @flip.setter
    def flip(self, mode: FlipMode) -> None:
        self._require_connected()
        assert self._cam is not None
        try:
            self._cam.set_control_value(asi.ASI_FLIP, mode.value)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set flip mode: {exc}") from exc

    # -- camera state properties ---------------------------------------------

    @property
    def bayer(self) -> str | None:
        """Bayer pattern string (e.g. 'RGGB'), or None for mono cameras."""
        self._require_connected()
        return self._info.bayer_pattern  # type: ignore[union-attr]

    @property
    def exposure_time(self) -> float:
        """Current exposure time in seconds."""
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_EXPOSURE)
            return val / 1_000_000
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read exposure time: {exc}") from exc

    @exposure_time.setter
    def exposure_time(self, seconds: float) -> None:
        self._require_connected()
        assert self._cam is not None
        try:
            self._cam.set_control_value(asi.ASI_EXPOSURE, int(seconds * 1_000_000))
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set exposure time: {exc}") from exc

    @property
    def gain(self) -> int:
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_GAIN)
            return val
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read gain: {exc}") from exc

    @gain.setter
    def gain(self, value: int) -> None:
        self._require_connected()
        assert self._cam is not None
        try:
            self._cam.set_control_value(asi.ASI_GAIN, value)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set gain: {exc}") from exc

    @property
    def offset(self) -> int:
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_BRIGHTNESS)
            return val
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read offset: {exc}") from exc

    @offset.setter
    def offset(self, value: int) -> None:
        self._require_connected()
        assert self._cam is not None
        try:
            self._cam.set_control_value(asi.ASI_BRIGHTNESS, value)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set offset: {exc}") from exc

    @property
    def bin(self) -> int:
        """Current binning factor."""
        self._require_connected()
        return self._current_binning()

    @bin.setter
    def bin(self, value: int) -> None:
        """Set binning factor, resetting ROI to full sensor at the new bin size."""
        self.set_roi(bin=value)

    @property
    def image_width(self) -> int:
        """Current ROI width in pixels."""
        self._require_connected()
        assert self._cam is not None
        _, _, width, _ = self._cam.get_roi()
        return width

    @image_width.setter
    def image_width(self, value: int) -> None:
        x, y, _, h = self.get_roi()
        self.set_roi(x=x, y=y, width=value, height=h)

    @property
    def image_height(self) -> int:
        """Current ROI height in pixels."""
        self._require_connected()
        assert self._cam is not None
        _, _, _, height = self._cam.get_roi()
        return height

    @image_height.setter
    def image_height(self, value: int) -> None:
        x, y, w, _ = self.get_roi()
        self.set_roi(x=x, y=y, width=w, height=value)

    @property
    def target_temp(self) -> float:
        """Cooler target temperature in °C."""
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_TARGET_TEMP)
            return float(val)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read target temperature: {exc}") from exc

    @target_temp.setter
    def target_temp(self, celsius: float) -> None:
        self._require_connected()
        assert self._cam is not None
        if not self._info.has_cooler:  # type: ignore[union-attr]
            raise ConfigurationError(f"{self._info.model} has no cooler.")  # type: ignore[union-attr]
        try:
            self._cam.set_control_value(asi.ASI_TARGET_TEMP, int(celsius))
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set target temperature: {exc}") from exc

    @property
    def cooler_on(self) -> bool:
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_COOLER_ON)
            return bool(val)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read cooler state: {exc}") from exc

    @cooler_on.setter
    def cooler_on(self, enabled: bool) -> None:
        self._require_connected()
        assert self._cam is not None
        if not self._info.has_cooler:  # type: ignore[union-attr]
            raise ConfigurationError(f"{self._info.model} has no cooler.")  # type: ignore[union-attr]
        try:
            self._cam.set_control_value(asi.ASI_COOLER_ON, 1 if enabled else 0)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set cooler state: {exc}") from exc

    @property
    def fan_on(self) -> bool:
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_FAN_ON)
            return bool(val)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read fan state: {exc}") from exc

    @fan_on.setter
    def fan_on(self, enabled: bool) -> None:
        self._require_connected()
        assert self._cam is not None
        try:
            self._cam.set_control_value(asi.ASI_FAN_ON, 1 if enabled else 0)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set fan state: {exc}") from exc

    # -- non-blocking imaging primitives ------------------------------------

    def start_exposure(self, is_dark: bool = False) -> None:
        """
        Begin an exposure using the current camera settings.

        All parameters (exposure time, gain, ROI, etc.) must be set via the
        corresponding properties before calling this. On success, a metadata
        snapshot of all current settings is stored and returned with the frame
        from read_frame().
        """
        self._require_connected()
        assert self._cam is not None
        snapshot = self._build_meta_snapshot()
        try:
            self._cam.start_exposure(is_dark)
            self._meta_snapshot = snapshot
        except asi.ZWO_Error as exc:
            raise ExposureError(f"Failed to start exposure: {exc}") from exc

    @property
    def exposure_status(self) -> ExposureStatus:
        self._require_connected()
        assert self._cam is not None
        try:
            raw = self._cam.get_exposure_status()
            return _STATUS_MAP.get(raw, ExposureStatus.FAILED)
        except asi.ZWO_Error as exc:
            raise ExposureError(f"Failed to read exposure status: {exc}") from exc

    def read_frame(self) -> Frame:
        """
        Collect completed frame data. Metadata is the snapshot taken at
        start_exposure() time, guaranteeing it matches the exposure conditions.
        """
        self._require_connected()
        assert self._cam is not None
        if self._meta_snapshot is None:
            raise ExposureError("read_frame() called before any exposure was started.")
        try:
            raw = self._cam.get_data_after_exposure(None)
            roi_w = self._meta_snapshot.roi_width
            roi_h = self._meta_snapshot.roi_height
            data = np.frombuffer(raw, dtype=np.uint16).reshape(roi_h, roi_w)
            return Frame(data=data, meta=self._meta_snapshot)
        except asi.ZWO_Error as exc:
            raise ExposureError(f"Failed to read frame data: {exc}") from exc

    def abort(self) -> None:
        if self._cam is not None and self._connected:
            try:
                self._cam.stop_exposure()
            except asi.ZWO_Error as exc:
                raise ExposureError(f"Abort failed: {exc}") from exc

    # -- temperature ---------------------------------------------------------

    def get_temperature(self) -> float | None:
        if not self._connected or self._cam is None:
            return None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_TEMPERATURE)
            return val / 10.0
        except asi.ZWO_Error:
            return None

    def set_target_temperature(self, celsius: float) -> None:
        self._require_connected()
        assert self._cam is not None
        if not self._info.has_cooler:  # type: ignore[union-attr]
            raise ConfigurationError(f"{self._info.model} has no cooler.")  # type: ignore[union-attr]
        try:
            self._cam.set_control_value(asi.ASI_TARGET_TEMP, int(celsius))
            self._cam.set_control_value(asi.ASI_COOLER_ON, 1)
            self._cam.set_control_value(asi.ASI_FAN_ON, 1)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set temperature: {exc}") from exc

    @property
    def bandwidth_overload(self) -> int:
        """USB bandwidth limit (0–100). Lower values reduce dropped frames on slower buses."""
        self._require_connected()
        assert self._cam is not None
        try:
            val, _ = self._cam.get_control_value(asi.ASI_BANDWIDTHOVERLOAD)
            return val
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to read bandwidth overload: {exc}") from exc

    @bandwidth_overload.setter
    def bandwidth_overload(self, value: int) -> None:
        self._require_connected()
        assert self._cam is not None
        try:
            self._cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, value)
        except asi.ZWO_Error as exc:
            raise ConfigurationError(f"Failed to set bandwidth overload: {exc}") from exc

    # -- internal helpers ----------------------------------------------------

    def _build_meta_snapshot(self) -> FrameMeta:
        """Read all current camera settings and return them as a FrameMeta."""
        assert self._cam is not None
        info = self._info  # type: ignore[union-attr]
        roi_x, roi_y, roi_w, roi_h = self._cam.get_roi()
        exp_us, _ = self._cam.get_control_value(asi.ASI_EXPOSURE)
        gain_val, _ = self._cam.get_control_value(asi.ASI_GAIN)
        offset_val, _ = self._cam.get_control_value(asi.ASI_BRIGHTNESS)
        flip_val, _ = self._cam.get_control_value(asi.ASI_FLIP)
        return FrameMeta(
            camera_id=info.camera_id,
            camera_model=info.model,
            timestamp=datetime.now(tz=timezone.utc),
            exposure_seconds=exp_us / 1_000_000,
            gain=gain_val,
            offset=offset_val,
            binning=self._current_binning(),
            temperature_c=self.get_temperature(),
            bayer_pattern=info.bayer_pattern,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_width=roi_w,
            roi_height=roi_h,
            flip=FlipMode(flip_val).name,
            FOV=self.meta.FOV,
            RA=self.meta.RA,
            Dec=self.meta.Dec,
            Lat=self.meta.Lat,
            Lon=self.meta.Lon,
            Filter=self.meta.Filter,
        )

    def _apply_camera_config(self) -> None:
        """Apply gain, offset, pattern, and flip from cameras.yaml if present."""
        assert self._info is not None
        try:
            cfg = _cam_cfg.load()
        except FileNotFoundError:
            return   # no config file — silently skip
        settings = cfg.resolve(self._info.model, self._info.camera_id)
        if settings.gain is not None:
            self.gain = settings.gain
        if settings.offset is not None:
            self.offset = settings.offset
        if settings.flip is not None:
            self.flip = FlipMode[settings.flip]
        # Store pattern on info so FrameGrabber can pick it up via cam.info.bayer_pattern
        if settings.pattern is not None:
            from dataclasses import replace as _replace
            self._info = _replace(self._info, bayer_pattern=settings.pattern)

    def _current_binning(self) -> int:
        assert self._cam is not None
        _, _, bins, _ = self._cam.get_roi_format()
        return bins
