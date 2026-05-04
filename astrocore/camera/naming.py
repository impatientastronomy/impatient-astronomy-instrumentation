"""
Image filename generation and parsing for the canonical frame naming convention.

Format: Cam{id}Bin{bin}Flip{flip}Filt{filter_id}_{exposure}us_{temp}C_{index:03d}.tif

Example: Cam3Bin1Flip0Filt3_2e6us_21C_001.tif
  Cam3   — camera_id 3
  Bin1   — 1×1 binning
  Flip0  — no flip (0=NONE, 1=HORIZ, 2=VERT, 3=BOTH)
  Filt3  — filter slot 3 (e.g. Ha — defined in cameras.yaml)
  2e6us  — 2,000,000 µs = 2 s
  21C    — sensor temperature rounded to nearest °C; 'm' prefix for negative; 'unk' if None
  001    — frame index, 1-based, zero-padded to 3 digits
"""

from __future__ import annotations

import re

from .base import FrameMeta

_PATTERN = re.compile(
    r"^Cam(\d+)Bin(\d+)Flip(\d+)Filt(\d+)_(\d+(?:e\d+)?)us_(m?\d+|unk)C_(\d{3})\.tif$"
)

# Maps flip name (as stored in FrameMeta.flip) to integer for filename
_FLIP_TO_INT: dict[str, int] = {"NONE": 0, "HORIZ": 1, "VERT": 2, "BOTH": 3}


def frame_filename(meta: FrameMeta, index: int) -> str:
    """Build a canonical .tif filename for a captured frame."""
    flip_int = _FLIP_TO_INT.get(meta.flip or "NONE", 0)
    prefix = (
        f"Cam{meta.camera_id}"
        f"Bin{meta.binning}"
        f"Flip{flip_int}"
        f"Filt{meta.filter_id}"
    )
    exp  = _format_exposure(meta.exposure_seconds)
    temp = _format_temperature(meta.temperature_c)
    return f"{prefix}_{exp}_{temp}_{index:03d}.tif"


def _format_exposure(seconds: float) -> str:
    """
    Express exposure time as compact SI-aligned scientific-notation microseconds.

    Tries e6 (seconds range) then e3 (milliseconds range), then falls back
    to raw integer microseconds for values that don't divide evenly.
    """
    us = round(seconds * 1_000_000)
    if us == 0:
        return "0us"
    for exp in (6, 3):
        divisor = 10 ** exp
        if us % divisor == 0:
            return f"{us // divisor}e{exp}us"
    return f"{us}us"


def _format_temperature(temp_c: float | None) -> str:
    """Round temperature to nearest integer degree. Negative uses 'm' prefix."""
    if temp_c is None:
        return "unkC"
    t = round(temp_c)
    if t < 0:
        return f"m{abs(t)}C"
    return f"{t}C"


# ---------------------------------------------------------------------------
# Parsing (inverse of the format functions above)
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict | None:
    """
    Parse a canonical frame filename into its components.

    Returns a dict with keys:
        camera_id (int), bin (int), flip (int), filter_id (int),
        exposure_us (int), exposure_s (float),
        temperature_c (float | None), frame_index (int)

    Returns None if the filename does not match the naming convention.
    """
    m = _PATTERN.match(filename)
    if not m:
        return None

    exposure_us = _parse_exposure_us(m.group(5))
    return {
        "camera_id":     int(m.group(1)),
        "bin":           int(m.group(2)),
        "flip":          int(m.group(3)),
        "filter_id":     int(m.group(4)),
        "exposure_us":   exposure_us,
        "exposure_s":    exposure_us / 1_000_000,
        "temperature_c": _parse_temperature(m.group(6)),
        "frame_index":   int(m.group(7)),
    }


def _parse_exposure_us(s: str) -> int:
    """Parse '2e6' → 2,000,000  or  '1500' → 1,500."""
    if "e" in s:
        mantissa, exp = s.split("e")
        return int(mantissa) * (10 ** int(exp))
    return int(s)


def _parse_temperature(s: str) -> float | None:
    """Parse '22' → 22.0, 'm10' → -10.0, 'unk' → None."""
    if s == "unk":
        return None
    if s.startswith("m"):
        return -float(s[1:])
    return float(s)
