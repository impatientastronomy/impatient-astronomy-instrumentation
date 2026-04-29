"""
Image filename generation and parsing for the canonical frame naming convention.

Format: C{camera_id}_{filter}_{exposure}_{temp}_{index:03d}.tif

Example: C3_Ha_1250e3us_22C_001.tif
  C3       — camera_id 3
  Ha       — filter name
  1250e3us — 1,250,000 µs = 1.25 s (SI-aligned scientific notation)
  22C      — sensor temperature rounded to nearest °C; 'm' prefix for negative; 'unk' if None
  001      — frame index, zero-padded to 3 digits
"""

from __future__ import annotations

import re

from .base import FrameMeta

_PATTERN = re.compile(
    r"^C(\d+)_([^_]+)_(\d+(?:e\d+)?)us_(m?\d+|unk)C_(\d{3})\.tif$"
)


def frame_filename(meta: FrameMeta, index: int) -> str:
    """Build a canonical .tif filename for a captured frame."""
    cam  = f"C{meta.camera_id}"
    filt = meta.Filter
    exp  = _format_exposure(meta.exposure_seconds)
    temp = _format_temperature(meta.temperature_c)
    idx  = f"{index:03d}"
    return f"{cam}_{filt}_{exp}_{temp}_{idx}.tif"


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
        camera_id (int), filter (str), exposure_us (int), exposure_s (float),
        temperature_c (float | None), index (int)

    Returns None if the filename does not match the naming convention.
    """
    m = _PATTERN.match(filename)
    if not m:
        return None

    exposure_us = _parse_exposure_us(m.group(3))
    return {
        "camera_id":     int(m.group(1)),
        "filter_name":   m.group(2),
        "exposure_us":   exposure_us,
        "exposure_s":    exposure_us / 1_000_000,
        "temperature_c": _parse_temperature(m.group(4)),
        "frame_index":   int(m.group(5)),
    }


def _parse_exposure_us(s: str) -> int:
    """Parse '1250e3' → 1,250,000  or  '1500' → 1,500."""
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
