"""
Configuration loader for configuration.yaml.

configuration.yaml is the single user-facing file for configuring the full
instrument setup: observer location, calibration data path, filters, and
per-camera telescope/optics/display settings.

Cameras section format
-----------------------
Each entry under ``cameras:`` is keyed by a descriptive name.  Multiple
entries may share the same ``id`` (different telescope configs for the same
physical camera)::

    cameras:
      primary:
        id: 3
        telescope_description: Vixen_600mm
        ...
      sct_config:
        id: 3
        telescope_description: SCT_2000mm
        ...
      second_camera:
        id: 7
        ...

The old single-block format (``cameras: {id: 3, config_1: {...}, ...}``) is
still accepted for backward compatibility.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configuration.yaml"

_FLIP_NAMES  = {0: "NONE", 1: "HORIZ", 2: "VERT", 3: "BOTH"}
_VALID_PATTERNS = {"NONE", "RGGB", "BGGR", "GRBG", "GBRG"}


@dataclass
class HotspotConfig:
    """Settings for the guest WiFi hotspot served on the second interface."""
    ssid:      str = "AstroEye"
    password:  str = "stargazer"
    interface: str = "wlan1"
    ip:        str = "192.168.10.1"
    port:      int = 8080

    @property
    def wifi_qr_data(self) -> str:
        """WiFi QR payload understood by iOS and Android camera apps."""
        return f"WIFI:S:{self.ssid};T:WPA;P:{self.password};;"

    @property
    def gallery_url(self) -> str:
        return f"http://{self.ip}:{self.port}" if self.port != 80 else f"http://{self.ip}"


@dataclass
class SkyMapConfig:
    """Parameters for the wide-angle virtual sky map view."""
    fov_min:     float = 10.0
    fov_default: float = 20.0
    fov_max:     float = 60.0


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Resolved settings for one camera + telescope combination."""
    # Telescope / optics
    telescope_description: str   = ""
    focal_length_mm:       float = 0.0
    mount_driver:          str   = ""      # python module name under astrocore.mount
    mount_type:            str   = "Alt-Az"
    # Camera hardware
    gain:        int | None        = None
    data_offset: int | None        = None   # moves readout values above zero
    pattern:     str | None        = None   # Bayer pattern; None for mono
    bin:         int               = 1
    flip:        int               = 0      # 0=NONE 1=HORIZ 2=VERT 3=BOTH
    cam_size:    tuple[int, int]   = (-1, -1)   # (width, height); -1 = sensor max
    cam_offset:  tuple[int, int]   = (-1, -1)   # (x, y) ROI start; -1 = auto-centre
    # Display / alignment (parsed but not yet consumed by the UI)
    image_offset:    tuple[int, int] = (0, 0)   # display pixels, shifts image
    image_scale:     float           = 1.0
    image_rotation:  float           = 0.0
    overlay_style:   str             = "style1"
    overlay_offset:  tuple[int, int] = (0, 0)   # display pixels, shifts overlay
    overlay_scale:   float           = 1.0
    overlay_rotation: float          = 0.0
    overlay_flip:    int             = 0

    @property
    def flip_name(self) -> str:
        """Flip as a driver-friendly string ('NONE', 'HORIZ', 'VERT', 'BOTH')."""
        return _FLIP_NAMES.get(self.flip, "NONE")

    def effective_roi(
        self,
        sensor_width_px: int,
        sensor_height_px: int,
    ) -> tuple[int, int, int, int]:
        """
        Return (x, y, width, height) with -1 values resolved against the sensor.

        cam_size values of -1 are replaced with the sensor dimension.
        cam_offset values of -1 are replaced with the centred position.
        """
        w = self.cam_size[0]   if self.cam_size[0]   > 0 else sensor_width_px
        h = self.cam_size[1]   if self.cam_size[1]   > 0 else sensor_height_px
        x = self.cam_offset[0] if self.cam_offset[0] >= 0 else (sensor_width_px  - w) // 2
        y = self.cam_offset[1] if self.cam_offset[1] >= 0 else (sensor_height_px - h) // 2
        return x, y, w, h


# ---------------------------------------------------------------------------
# FOV utility
# ---------------------------------------------------------------------------

def compute_hfov(
    focal_length_mm: float,
    pixel_size_um:   float,
    image_width_px:  int,
) -> float:
    """
    Horizontal field of view in degrees.

    Uses the thin-lens formula: hfov = 2 * arctan(sensor_width / (2 * f))
    where sensor_width = pixel_size_um * image_width_px / 1000.
    Returns 0.0 if any argument is non-positive.
    """
    if focal_length_mm <= 0 or pixel_size_um <= 0 or image_width_px <= 0:
        return 0.0
    sensor_width_mm = pixel_size_um * image_width_px / 1000.0
    return 2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * focal_length_mm)))


# ---------------------------------------------------------------------------
# Top-level Configuration object
# ---------------------------------------------------------------------------

class Configuration:
    """Parsed contents of configuration.yaml."""

    def __init__(self, raw: dict, path: Path) -> None:
        self._path = path
        self.latitude:            float      = float(raw.get("latitude",  0.0))
        self.longitude:           float      = float(raw.get("longitude", 0.0))
        self.preferred_camera_id: int | None = raw.get("preferred_camera_id")
        self.max_zoom: float = float(raw.get("max_zoom", 5.0))

        raw_cal = raw.get("cal_path")
        self.cal_path: Path | None = Path(str(raw_cal)) if raw_cal else None

        raw_rec = raw.get("record_path")
        self.record_path: Path | None = Path(str(raw_rec)) if raw_rec else None

        raw_img = raw.get("image_path")
        self.image_path: Path | None = (
            Path(str(raw_img)).expanduser() if raw_img else None
        )

        raw_hs = raw.get("hotspot") or {}
        self.hotspot = HotspotConfig(
            ssid      = str(raw_hs.get("ssid",      "AstroEye")),
            password  = str(raw_hs.get("password",  "stargazer")),
            interface = str(raw_hs.get("interface", "wlan1")),
            ip        = str(raw_hs.get("ip",        "192.168.10.1")),
            port      = int(raw_hs.get("port",      8080)),
        )

        raw_filters = raw.get("filters") or {}
        self.filters: dict[int, str] = {int(k): str(v) for k, v in raw_filters.items()}

        raw_sm = raw.get("sky_map") or {}
        fov_list = raw_sm.get("map_fov", [10, 20, 60])
        self.sky_map = SkyMapConfig(
            fov_min     = float(fov_list[0]) if len(fov_list) > 0 else 10.0,
            fov_default = float(fov_list[1]) if len(fov_list) > 1 else 20.0,
            fov_max     = float(fov_list[2]) if len(fov_list) > 2 else 60.0,
        )

        self._blocks: list[_CameraBlock] = _parse_cameras(raw.get("cameras") or {})
        self._validate()

    # -- camera / config access ----------------------------------------------

    def camera_ids(self) -> list[int]:
        """Camera IDs present in configuration.yaml."""
        return [b.camera_id for b in self._blocks]

    def config_names(self, camera_id: int) -> list[str]:
        """Config names for a camera, e.g. ['primary', 'sct_config']."""
        block = self._block(camera_id)
        return list(block.configs) if block else []

    def telescope_descriptions(self, camera_id: int) -> list[tuple[str, str]]:
        """Return [(config_name, telescope_description), …] for a camera."""
        block = self._block(camera_id)
        return block.list_configs() if block else []

    def get_config(
        self,
        camera_id: int,
        config_name: str | None = None,
    ) -> CameraConfig:
        """
        Resolved settings for a camera.

        config_name=None selects the first config listed for that camera.
        Returns a default CameraConfig if camera_id is not in the file.
        """
        block = self._block(camera_id)
        return block.get_config(config_name) if block else CameraConfig()

    # -- internal ------------------------------------------------------------

    def _block(self, camera_id: int) -> "_CameraBlock | None":
        for b in self._blocks:
            if b.camera_id == camera_id:
                return b
        return None

    def _validate(self) -> None:
        for block in self._blocks:
            for name, raw in block.configs.items():
                pat = raw.get("pattern")
                if pat and pat not in _VALID_PATTERNS:
                    raise ValueError(
                        f"configuration.yaml camera {block.camera_id} "
                        f"{name!r}.pattern={pat!r} "
                        f"must be one of {sorted(_VALID_PATTERNS)}."
                    )
                flip = raw.get("flip", 0)
                if flip not in _FLIP_NAMES:
                    raise ValueError(
                        f"configuration.yaml camera {block.camera_id} "
                        f"{name!r}.flip={flip!r} must be 0–3."
                    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _CameraBlock:
    def __init__(self, camera_id: int, configs: dict[str, dict]) -> None:
        self.camera_id = camera_id
        self.configs: dict[str, dict] = configs

    def list_configs(self) -> list[tuple[str, str]]:
        return [
            (name, cfg.get("telescope_description", name))
            for name, cfg in self.configs.items()
        ]

    def get_config(self, name: str | None = None) -> CameraConfig:
        if name is None:
            name = next(iter(self.configs), None)
        if name is None:
            return CameraConfig()
        return _parse_camera_config(self.configs.get(name, {}))


def _parse_camera_config(raw: dict) -> CameraConfig:
    # cam_size — new name; fall back to old image_width / image_height
    cam_size_raw = raw.get("cam_size")
    if cam_size_raw and len(cam_size_raw) == 2:
        cam_w, cam_h = int(cam_size_raw[0]), int(cam_size_raw[1])
    else:
        cam_w = int(raw.get("image_width",  -1))
        cam_h = int(raw.get("image_height", -1))

    # cam_offset — new name; fall back to old x_offset / y_offset
    cam_offset_raw = raw.get("cam_offset")
    if cam_offset_raw and len(cam_offset_raw) == 2:
        cam_x, cam_y = int(cam_offset_raw[0]), int(cam_offset_raw[1])
    else:
        cam_x = int(raw.get("x_offset", -1))
        cam_y = int(raw.get("y_offset", -1))

    image_offset_raw   = raw.get("image_offset",   [0, 0])
    overlay_offset_raw = raw.get("overlay_offset", [0, 0])

    return CameraConfig(
        telescope_description = str(raw.get("telescope_description", "")),
        focal_length_mm       = float(raw.get("focal_length_mm", 0.0)),
        mount_driver          = str(raw.get("mount_driver", "")),
        mount_type            = str(raw.get("mount_type", "Alt-Az")),
        gain                  = raw.get("gain"),
        data_offset           = raw.get("data_offset", raw.get("offset")),  # new name, fallback to old
        pattern               = raw.get("pattern") or None,
        bin                   = int(raw.get("bin", 1)),
        flip                  = int(raw.get("flip", 0)),
        cam_size              = (cam_w, cam_h),
        cam_offset            = (cam_x, cam_y),
        image_offset          = (int(image_offset_raw[0]),   int(image_offset_raw[1])),
        image_scale           = float(raw.get("image_scale",    1.0)),
        image_rotation        = float(raw.get("image_rotation", 0.0)),
        overlay_style         = str(raw.get("overlay_style", "style1")),
        overlay_offset        = (int(overlay_offset_raw[0]), int(overlay_offset_raw[1])),
        overlay_scale         = float(raw.get("overlay_scale",    1.0)),
        overlay_rotation      = float(raw.get("overlay_rotation", 0.0)),
        overlay_flip          = int(raw.get("overlay_flip", 0)),
    )


def _parse_cameras(raw: object) -> list[_CameraBlock]:
    if not isinstance(raw, dict):
        return []

    # Old format: 'id' key at the cameras level, config_N sub-dicts alongside it.
    if "id" in raw:
        cam_id  = int(raw.get("id", 0))
        configs = {
            k: (v or {})
            for k, v in raw.items()
            if k not in ("id", "ID") and isinstance(v, dict)
        }
        return [_CameraBlock(cam_id, configs)]

    # New format: each key is a camera name; the value dict contains 'id'.
    by_id: dict[int, dict[str, dict]] = {}
    for name, cam_raw in raw.items():
        if not isinstance(cam_raw, dict):
            continue
        cam_id = int(cam_raw.get("id", 0))
        by_id.setdefault(cam_id, {})[name] = cam_raw

    return [_CameraBlock(cam_id, cfgs) for cam_id, cfgs in by_id.items()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load(config_path: str | Path | None = None) -> Configuration:
    """
    Load and return the configuration.  Raises FileNotFoundError if not found.

    Search order:
      1. config_path argument
      2. CAMERA_CONFIG environment variable
      3. configuration.yaml at the repo root
    """
    path = Path(
        config_path
        or os.environ.get("CAMERA_CONFIG", "")
        or _DEFAULT_CONFIG_PATH
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration not found at {path}. "
            "Create configuration.yaml or set CAMERA_CONFIG."
        )
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Configuration(raw, path)
