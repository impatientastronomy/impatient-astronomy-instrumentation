from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
import time

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CameraError(Exception):
    """Base for all camera errors."""

class ConnectionError(CameraError):
    """Raised when a connect/disconnect operation fails."""

class ExposureError(CameraError):
    """Raised when an exposure fails or is aborted."""

class ConfigurationError(CameraError):
    """Raised when an invalid setting is applied to the camera."""


# ---------------------------------------------------------------------------
# Exposure status
# ---------------------------------------------------------------------------

class ExposureStatus(Enum):
    IDLE    = auto()
    WORKING = auto()
    SUCCESS = auto()
    FAILED  = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraInfo:
    """
    Static hardware properties — populated once during connect(), never changes.

    usb_index:
        The order in which USB enumerated this camera. Unstable — can differ
        between power cycles. Use camera_id for reliable identification.
    camera_id:
        User-assigned number (1–8) written to camera EEPROM. Stable across
        USB reconnects. 0 means the ID has not been set yet.
    bayer_pattern:
        Colour filter arrangement (e.g. 'RGGB'). None for mono cameras.
    """
    model: str
    usb_index: int
    sensor_width_px: int
    sensor_height_px: int
    pixel_size_um: float
    bit_depth: int
    camera_id: int = 0
    has_cooler: bool = False
    bayer_pattern: str | None = None



@dataclass
class PersistentMeta:
    """
    Observation context that persists on a camera instance across frames.

    Set fields directly on the camera (cam.meta.FOV = 2.3) and they will
    appear in the metadata of every subsequent frame until changed.
    Cleared automatically when the camera is disconnected.
    """
    FOV:    float = 0.0    # field of view in degrees
    RA:     float = 0.0    # right ascension in degrees
    Dec:    float = 0.0    # declination in degrees
    Lat:    float = 0.0    # observer latitude in degrees
    Lon:    float = 0.0    # observer longitude in degrees
    Filter: str   = "none"


@dataclass
class FrameMeta:
    """
    Everything worth knowing about where a frame came from.

    Stored alongside the pixel data so a frame is self-describing. Can also
    be embedded into the first row of the image itself (see meta.py) so the
    provenance survives even if the frame is saved as a bare TIFF.

    extras:
        Open-ended dict for fields we haven't thought of yet. Add to it
        freely; existing code that doesn't know about a key will ignore it.
    """
    camera_id: int
    camera_model: str
    timestamp: datetime
    exposure_seconds: float
    gain: int
    offset: int
    binning: int
    temperature_c: float | None = None
    bayer_pattern: str | None = None
    roi_x: int | None = None
    roi_y: int | None = None
    roi_width: int | None = None
    roi_height: int | None = None
    flip: str | None = None
    FOV:    float = 0.0
    RA:     float = 0.0
    Dec:    float = 0.0
    Lat:    float = 0.0
    Lon:    float = 0.0
    Filter: str   = "none"
    extras: dict = field(default_factory=dict)


@dataclass
class Frame:
    """A single captured image plus everything needed to interpret it."""
    data: np.ndarray    # 2-D uint16 (mono or raw Bayer), or 3-D after debayer
    meta: FrameMeta


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Camera(ABC):
    """
    Contract that every camera driver must fulfill.

    Set camera parameters via properties before capturing::

        cam.exposure_time = 2.0
        cam.gain = 100

    There are two ways to capture frames:

    1. Blocking — good for scripts, tests, and the sky survey::

        with MyCamera() as cam:
            frame = cam.expose()

    2. Non-blocking — for live loops with multiple cameras or a responsive UI.
       Call the three primitives separately::

        cam.start_exposure()              # returns immediately; snapshots metadata
        while cam.exposure_status == ExposureStatus.WORKING:
            ...                           # update UI, poll other cameras, etc.
        frame = cam.read_frame()          # collect when ready

       In practice, FrameGrabber manages this loop for you.
    """

    def __init__(self) -> None:
        self._connected = False
        self.meta = PersistentMeta()

    # -- connection lifecycle ------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Open the device and prepare it for use."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the device."""

    def __enter__(self) -> "Camera":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # -- hardware info -------------------------------------------------------

    @property
    @abstractmethod
    def info(self) -> CameraInfo:
        """Return static hardware properties (available after connect)."""

    # -- non-blocking imaging primitives ------------------------------------

    @abstractmethod
    def start_exposure(self, is_dark: bool = False) -> None:
        """Begin an exposure. Returns immediately — does not wait."""

    @property
    @abstractmethod
    def exposure_status(self) -> ExposureStatus:
        """Current exposure state. Safe to call at any time."""

    @abstractmethod
    def read_frame(self) -> Frame:
        """
        Collect the completed frame and return it.
        Only call when exposure_status is SUCCESS.
        """

    @abstractmethod
    def abort(self) -> None:
        """Cancel an in-progress exposure immediately."""

    # -- blocking convenience -----------------------------------------------

    def expose(
        self,
        is_dark: bool = False,
        timeout: float = 30.0,
        poll_interval: float = 0.05,
    ) -> Frame:
        """
        Blocking single-shot capture built on the three non-blocking primitives.

        Suitable for tests, scripts, and use cases that don't need a live loop.
        Not suitable for multi-camera parallel capture.
        """
        self._require_connected()
        self.start_exposure(is_dark)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.exposure_status
            if status == ExposureStatus.SUCCESS:
                return self.read_frame()
            if status == ExposureStatus.FAILED:
                raise ExposureError("Camera reported exposure failure.")
            if status == ExposureStatus.IDLE:
                raise ExposureError("Exposure ended unexpectedly.")
            time.sleep(poll_interval)
        raise ExposureError(f"Exposure timed out after {timeout:.1f} s.")

    # -- temperature (optional) ---------------------------------------------

    def get_temperature(self) -> float | None:
        """Return sensor temperature in °C, or None if unsupported."""
        return None

    def set_target_temperature(self, celsius: float) -> None:
        """Set cooler target. Raises ConfigurationError if unsupported."""
        raise ConfigurationError(f"{self.__class__.__name__} has no cooler.")

    # -- internal -----------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise ConnectionError(
                f"{self.__class__.__name__} is not connected. Call connect() first."
            )
