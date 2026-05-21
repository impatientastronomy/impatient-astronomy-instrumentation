"""
OverlayStyle — per-object-type rendering settings for the sky overlay.

Loaded from overlay_style.yaml at startup; the style name comes from
CameraConfig.overlay_style.  Falls back to built-in defaults if the file
or style name is not found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_VALID_SHAPES = {"dot", "circle", "circle_cross", "cross"}


@dataclass
class FeatureStyle:
    """Rendering settings for horizon, FOV box, or constellation lines."""
    show:  bool              = True
    color: tuple[int, int, int] = (255, 255, 255)
    alpha: int               = 180

    def rgba(self) -> tuple[int, int, int, int]:
        return (*self.color, self.alpha)


@dataclass
class ObjectTypeStyle:
    """Rendering settings for one object type (star, galaxy, nebula, cluster)."""
    color:     tuple[int, int, int] = (255, 255, 255)
    alpha:     int   = 210
    mag_limit: float = 6.5
    max_count: int   = 200
    size_min:  int   = 1
    size_max:  int   = 40
    shape:     str   = "circle_cross"

    def rgba(self) -> tuple[int, int, int, int]:
        return (*self.color, self.alpha)


@dataclass
class OverlayStyle:
    """Complete rendering style for all object types."""
    label_mag_limit: float = 4.0
    star:    ObjectTypeStyle = field(default_factory=lambda: ObjectTypeStyle(
        color=(255, 245, 220), alpha=210, mag_limit=6.5,
        max_count=300, size_min=1, size_max=5, shape="dot"))
    galaxy:  ObjectTypeStyle = field(default_factory=lambda: ObjectTypeStyle(
        color=(80, 200, 255), alpha=210, mag_limit=10.0,
        max_count=50, size_min=4, size_max=40, shape="circle_cross"))
    nebula:  ObjectTypeStyle = field(default_factory=lambda: ObjectTypeStyle(
        color=(255, 80, 80), alpha=210, mag_limit=10.0,
        max_count=30, size_min=4, size_max=40, shape="circle_cross"))
    cluster: ObjectTypeStyle = field(default_factory=lambda: ObjectTypeStyle(
        color=(80, 220, 80), alpha=210, mag_limit=8.0,
        max_count=30, size_min=4, size_max=40, shape="circle_cross"))
    horizon:       FeatureStyle = field(default_factory=lambda: FeatureStyle(
        show=True,  color=(255, 160, 0),   alpha=180))
    fov_box:       FeatureStyle = field(default_factory=lambda: FeatureStyle(
        show=True,  color=(255, 255, 255), alpha=200))
    constellations: FeatureStyle = field(default_factory=lambda: FeatureStyle(
        show=True,  color=(80, 100, 220),  alpha=160))

    def for_type(self, obj_type: str) -> ObjectTypeStyle:
        """Return the style for a given ObjType string, defaulting to star."""
        return getattr(self, obj_type, self.star)


_DEFAULT_STYLE = OverlayStyle()


def load_overlay_style(path: str | Path, style_name: str) -> OverlayStyle:
    """
    Load a named style from overlay_style.yaml.

    Returns the built-in default style if the file is missing, unreadable,
    or the requested style_name is not defined.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        return OverlayStyle()

    block = raw.get(style_name)
    if not isinstance(block, dict):
        return OverlayStyle()

    return OverlayStyle(
        label_mag_limit = float(block.get("label_mag_limit", 4.0)),
        star    = _parse_type_style(block.get("star"),    _DEFAULT_STYLE.star),
        galaxy  = _parse_type_style(block.get("galaxy"),  _DEFAULT_STYLE.galaxy),
        nebula  = _parse_type_style(block.get("nebula"),  _DEFAULT_STYLE.nebula),
        cluster = _parse_type_style(block.get("cluster"), _DEFAULT_STYLE.cluster),
        horizon        = _parse_feature_style(block.get("horizon"),        _DEFAULT_STYLE.horizon),
        fov_box        = _parse_feature_style(block.get("fov_box"),        _DEFAULT_STYLE.fov_box),
        constellations = _parse_feature_style(block.get("constellations"), _DEFAULT_STYLE.constellations),
    )


def _parse_feature_style(raw: object, default: FeatureStyle) -> FeatureStyle:
    if not isinstance(raw, dict):
        return default
    color_raw = raw.get("color", list(default.color))
    return FeatureStyle(
        show  = bool(raw.get("show",  default.show)),
        color = (int(color_raw[0]), int(color_raw[1]), int(color_raw[2])),
        alpha = int(raw.get("alpha", default.alpha)),
    )


def _parse_type_style(raw: object, default: ObjectTypeStyle) -> ObjectTypeStyle:
    if not isinstance(raw, dict):
        return default
    color_raw = raw.get("color", list(default.color))
    shape = str(raw.get("shape", default.shape))
    if shape not in _VALID_SHAPES:
        shape = default.shape
    return ObjectTypeStyle(
        color     = (int(color_raw[0]), int(color_raw[1]), int(color_raw[2])),
        alpha     = int(raw.get("alpha",     default.alpha)),
        mag_limit = float(raw.get("mag_limit", default.mag_limit)),
        max_count = int(raw.get("max_count",  default.max_count)),
        size_min  = int(raw.get("size_min",   default.size_min)),
        size_max  = int(raw.get("size_max",   default.size_max)),
        shape     = shape,
    )
