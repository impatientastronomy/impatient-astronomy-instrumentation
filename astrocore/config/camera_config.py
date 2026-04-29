"""
Camera configuration loader.

Reads cameras.yaml and returns per-camera settings so callers never have to
hard-code gain, offset, Bayer pattern, or flip mode.

Search order for the config file:
  1. Path passed explicitly to load()
  2. CAMERA_CONFIG env var
  3. cameras.yaml next to the repo root (two levels up from this file)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Repo root is three levels up: astrocore/config/camera_config.py
_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "cameras.yaml"

_VALID_FLIPS    = {"NONE", "HORIZ", "VERT", "BOTH"}
_VALID_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


@dataclass
class CameraSettings:
    """Resolved settings for one physical camera."""
    gain:        int | None = None
    offset:      int | None = None
    pattern:     str | None = None   # Bayer pattern, None for mono
    flip:        str | None = None   # FlipMode name, None → leave as-is


def load(config_path: str | Path | None = None) -> "_CameraConfig":
    """Load and return the camera config. Raises FileNotFoundError if not found."""
    path = Path(
        config_path
        or os.environ.get("CAMERA_CONFIG", "")
        or _DEFAULT_CONFIG_PATH
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Camera config not found at {path}. "
            "Create cameras.yaml or set CAMERA_CONFIG."
        )
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return _CameraConfig(raw, path)


class _CameraConfig:
    def __init__(self, raw: dict, path: Path) -> None:
        self._models:  dict = raw.get("models", {}) or {}
        self._cameras: dict = raw.get("cameras", {}) or {}
        self._path = path
        raw_cal = raw.get("cal_path")
        self.cal_path: Path | None = Path(raw_cal) if raw_cal else None
        self._validate()

    def _validate(self) -> None:
        for model, vals in self._models.items():
            if not isinstance(vals, dict):
                raise ValueError(f"cameras.yaml: models.{model!r} must be a mapping.")
        for cam_id, vals in self._cameras.items():
            if vals is None:
                continue
            flip = (vals or {}).get("flip")
            if flip and flip not in _VALID_FLIPS:
                raise ValueError(
                    f"cameras.yaml: cameras.{cam_id}.flip={flip!r} "
                    f"must be one of {sorted(_VALID_FLIPS)}."
                )
            pat = (vals or {}).get("pattern")
            if pat and pat not in _VALID_PATTERNS:
                raise ValueError(
                    f"cameras.yaml: cameras.{cam_id}.pattern={pat!r} "
                    f"must be one of {sorted(_VALID_PATTERNS)}."
                )

    def resolve(self, model: str, camera_id: int) -> CameraSettings:
        """
        Return merged settings for a camera: model defaults overridden by
        per-camera values. Fields absent in both sources are left as None.
        """
        model_vals  = self._models.get(model, {}) or {}
        camera_vals = self._cameras.get(camera_id, {}) or {}

        return CameraSettings(
            gain    = camera_vals.get("gain",    model_vals.get("gain")),
            offset  = camera_vals.get("offset",  model_vals.get("offset")),
            pattern = camera_vals.get("pattern", model_vals.get("pattern")),
            flip    = camera_vals.get("flip",    model_vals.get("flip")),
        )
