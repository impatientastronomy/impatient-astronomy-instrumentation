"""
Configuration loader for configuration.yaml.

configuration.yaml is the single user-facing file for configuring the full
instrument setup: observer location, calibration data path, filters, and
per-camera telescope/optics/display settings.

Each camera may have multiple named configs (e.g. "config 1" for a 600 mm
refractor and "config 2" for an SCT at 2000 mm).  The application picks the
active config at runtime; the config name and telescope_description appear in
the Telescope menu.

Cameras section — two supported formats
----------------------------------------
Single camera (current format)::

    cameras:
      ID: 3
      config 1:
        telescope_description: Vixen_600mm
        ...
      config 2:
        telescope_description: SCT_2000
        ...

Multiple cameras (list format, for future multi-camera setups)::

    cameras:
      - ID: 3
        config 1: ...
      - ID: 7
        config 1: ...
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configuration.yaml"

_FLIP_NAMES  = {0: "NONE", 1: "HORIZ", 2: "VERT", 3: "BOTH"}
_VALID_PATTERNS = {"NONE", "RGGB", "BGGR", "GRBG", "GBRG"}


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
    # Camera
    gain:         int | None = None
    offset:       int | None = None
    pattern:      str | None = None   # Bayer pattern; None for mono
    bin:          int        = 1
    flip:         int        = 0      # 0=NONE 1=HORIZ 2=VERT 3=BOTH
    image_width:  int        = -1     # sensor ROI width;  -1 = sensor max
    image_height: int        = -1     # sensor ROI height; -1 = sensor max
    x_offset:     int        = -1     # ROI start column; -1 = auto-centre
    y_offset:     int        = -1     # ROI start row;    -1 = auto-centre
    # Display / overlay
    overlay_style:  str   = "style1"
    overlay_flip:   int   = 0
    image_rotation: float = 0.0

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

        x_offset / y_offset of -1 are replaced with the centred position.
        image_width / image_height of -1 are replaced with the sensor dimension.
        """
        w = self.image_width  if self.image_width  > 0 else sensor_width_px
        h = self.image_height if self.image_height > 0 else sensor_height_px
        x = self.x_offset if self.x_offset >= 0 else (sensor_width_px  - w) // 2
        y = self.y_offset if self.y_offset >= 0 else (sensor_height_px - h) // 2
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

        raw_cal = raw.get("cal_path")
        self.cal_path: Path | None = Path(str(raw_cal)) if raw_cal else None

        raw_rec = raw.get("record_path")
        self.record_path: Path | None = Path(str(raw_rec)) if raw_rec else None

        raw_filters = raw.get("filters") or {}
        self.filters: dict[int, str] = {int(k): str(v) for k, v in raw_filters.items()}

        self._blocks: list[_CameraBlock] = _parse_cameras(raw.get("cameras") or {})
        self._validate()

    # -- camera / config access ----------------------------------------------

    def camera_ids(self) -> list[int]:
        """Camera IDs present in configuration.yaml."""
        return [b.camera_id for b in self._blocks]

    def config_names(self, camera_id: int) -> list[str]:
        """Config names for a camera, e.g. ['config 1', 'config 2']."""
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
    def __init__(self, raw: dict) -> None:
        self.camera_id: int = int(raw.get("ID", 0))
        self.configs: dict[str, dict] = {
            k: (v or {})
            for k, v in raw.items()
            if k != "ID" and isinstance(v, dict)
        }

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
    return CameraConfig(
        telescope_description = str(raw.get("telescope_description", "")),
        focal_length_mm       = float(raw.get("focal_length_mm", 0.0)),
        mount_driver          = str(raw.get("mount_driver", "")),
        mount_type            = str(raw.get("mount_type", "Alt-Az")),
        gain                  = raw.get("gain"),
        offset                = raw.get("offset"),
        pattern               = raw.get("pattern") or None,
        bin                   = int(raw.get("bin", 1)),
        flip                  = int(raw.get("flip", 0)),
        image_width           = int(raw.get("image_width",  -1)),
        image_height          = int(raw.get("image_height", -1)),
        x_offset              = int(raw.get("x_offset", -1)),
        y_offset              = int(raw.get("y_offset", -1)),
        overlay_style         = str(raw.get("overlay_style", "style1")),
        overlay_flip          = int(raw.get("overlay_flip", 0)),
        image_rotation        = float(raw.get("image_rotation", 0.0)),
    )


def _parse_cameras(raw: object) -> list[_CameraBlock]:
    if isinstance(raw, dict):
        return [_CameraBlock(raw)]
    if isinstance(raw, list):
        return [_CameraBlock(c) for c in raw if isinstance(c, dict)]
    return []


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
