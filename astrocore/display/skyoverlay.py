"""
Sky map overlay for the digital eyepiece.

Loads a star/DSO catalog and renders an RGBA overlay image aligned to the
current telescope pointing using a gnomonic (TAN) projection.

Public API
----------
load_catalog(path)           -- load skyChart.csv, return a CatalogEntry list
compute_overlay(...)         -- build an RGBA numpy array for blitting over the camera frame
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from astrocore.mount.coord import _jd, _lst_deg, altaz_to_radec


# ---------------------------------------------------------------------------
# Catalog types
# ---------------------------------------------------------------------------

class ObjType:
    STAR    = "star"
    GALAXY  = "galaxy"
    NEBULA  = "nebula"
    CLUSTER = "cluster"


# RGBA colors for each object type
_COLORS: dict[str, tuple[int, int, int, int]] = {
    ObjType.STAR:    (255, 245, 220, 210),  # warm white
    ObjType.GALAXY:  ( 80, 200, 255, 210),  # cyan-blue
    ObjType.NEBULA:  (255,  80,  80, 210),  # red
    ObjType.CLUSTER: ( 80, 220,  80, 210),  # green
}


@dataclass
class CatalogEntry:
    ra_deg:  float
    dec_deg: float
    mag:     float
    size_arcmin: float
    obj_type: str       # ObjType constant
    name:     str       # primary name (first in semicolon list)


def _classify(type_str: str) -> str:
    t = type_str.lower()
    if t.startswith("star"):
        return ObjType.STAR
    if "galaxy" in t:
        return ObjType.GALAXY
    if "nebul" in t or "neb" in t:
        return ObjType.NEBULA
    if "cluster" in t:
        return ObjType.CLUSTER
    return ObjType.STAR


def _primary_name(name_field: str) -> str:
    parts = [p.strip() for p in name_field.split(";") if p.strip()]
    return parts[0] if parts else ""


def load_catalog(path: str) -> list[CatalogEntry]:
    """Load skyChart.csv and return a list of CatalogEntry objects."""
    entries: list[CatalogEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entries.append(CatalogEntry(
                    ra_deg       = float(row["RA"]),
                    dec_deg      = float(row["DEC"]),
                    mag          = float(row["MAG"]),
                    size_arcmin  = float(row["SIZE"]) if row["SIZE"] else 0.0,
                    obj_type     = _classify(row["TYPE"]),
                    name         = _primary_name(row["NAME"]),
                ))
            except (ValueError, KeyError):
                continue
    return entries


# ---------------------------------------------------------------------------
# Gnomonic projection helpers
# ---------------------------------------------------------------------------

def _gnomonic_project(
    alt_rad:    np.ndarray,   # shape (N,)
    az_rad:     np.ndarray,   # shape (N,)
    center_alt: float,        # radians
    center_az:  float,        # radians
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project (alt, az) onto the tangent plane at (center_alt, center_az).

    Returns (x, y) in radian-scale tangent-plane coordinates.
    x increases to the right (East),  y increases upward (North on sky).
    Points behind the tangent plane (cos_c <= 0) are masked as NaN.
    """
    # Unit vectors in local horizontal (ENU) frame
    # East  = (cos(az), -sin(az), 0)  — but we only need dot products
    # North = (-sin(alt)*sin(az), -sin(alt)*cos(az), cos(alt))
    # Up    = (cos(alt)*cos(az),   cos(alt)*sin(az), sin(alt))
    #
    # Dot product with unit vector at center to get cos(angular distance c):
    cos_c = (
        np.sin(center_alt) * np.sin(alt_rad)
        + np.cos(center_alt) * np.cos(alt_rad) * np.cos(az_rad - center_az)
    )

    # Tangent-plane coords: x = East component / cos_c, y = North component / cos_c
    x = np.cos(alt_rad) * np.sin(az_rad - center_az) / cos_c
    # North-up on sky means increasing altitude away from horizon, and
    # azimuth wraps around; the "northward" component in the tangent plane:
    y = (
        np.cos(center_alt) * np.sin(alt_rad)
        - np.sin(center_alt) * np.cos(alt_rad) * np.cos(az_rad - center_az)
    ) / cos_c

    # Mask points behind the tangent plane
    mask = cos_c <= 0.0
    x[mask] = np.nan
    y[mask] = np.nan

    return x, y


# ---------------------------------------------------------------------------
# Marker drawing (into a uint8 RGBA array)
# ---------------------------------------------------------------------------

def _draw_circle(
    img: np.ndarray,
    cx: int, cy: int,
    radius: int,
    color: tuple[int, int, int, int],
    fill: bool = False,
) -> None:
    """Draw a circle (filled or outline) into an RGBA array.  Clips to bounds."""
    h, w = img.shape[:2]
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dx * dx + dy * dy
            px, py = cx + dx, cy + dy
            if px < 0 or px >= w or py < 0 or py >= h:
                continue
            if fill:
                if d2 <= r2:
                    img[py, px] = color
            else:
                if (radius - 1) ** 2 < d2 <= r2:
                    img[py, px] = color


def _draw_cross(
    img: np.ndarray,
    cx: int, cy: int,
    half: int,
    color: tuple[int, int, int, int],
) -> None:
    h, w = img.shape[:2]
    for dx in range(-half, half + 1):
        px = cx + dx
        if 0 <= px < w and 0 <= cy < h:
            img[cy, px] = color
    for dy in range(-half, half + 1):
        py = cy + dy
        if 0 <= cx < w and 0 <= py < h:
            img[py, cx] = color


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_overlay(
    catalog:    Sequence[CatalogEntry],
    fov_deg:    float,
    alt_deg:    float,
    az_deg:     float,
    image_shape: tuple[int, int],          # (height, width)
    lat_deg:    float,
    lon_deg:    float,
    t:          datetime | None = None,
    mag_limit:  float = 6.5,
    label_limit: float = 4.0,
) -> tuple[np.ndarray, list[dict]]:
    """
    Build an RGBA sky overlay aligned to the telescope pointing.

    Parameters
    ----------
    catalog      : output of load_catalog()
    fov_deg      : horizontal field of view in degrees
    alt_deg      : telescope pointing altitude (degrees)
    az_deg       : telescope pointing azimuth (degrees, N=0 E=90)
    image_shape  : (height, width) of the camera frame in pixels
    lat_deg      : observer latitude  (+N)
    lon_deg      : observer longitude (+E)
    t            : UTC datetime (defaults to now)
    mag_limit    : faintest magnitude to render
    label_limit  : faintest magnitude to label with a name

    Returns
    -------
    overlay : np.ndarray shape (H, W, 4) uint8 — transparent RGBA image
    table   : list of dicts with keys 'name', 'mag', 'type', 'px', 'py'
              for all objects rendered (for hover/info display)
    """
    h, w = image_shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    fov_rad      = math.radians(fov_deg)
    center_alt   = math.radians(alt_deg)
    center_az    = math.radians(az_deg)

    # Scale factor: pixels per radian in tangent plane
    # The full horizontal FOV spans fov_rad radians → w pixels
    scale = w / fov_rad   # pixels per radian

    # Build full arrays from catalog
    ra_arr   = np.array([e.ra_deg      for e in catalog])
    dec_arr  = np.array([e.dec_deg     for e in catalog])
    mag_arr  = np.array([e.mag         for e in catalog])
    size_arr = np.array([e.size_arcmin for e in catalog])

    # --- Stage 1: magnitude filter ---
    mag_mask = mag_arr <= mag_limit
    ra_arr   = ra_arr[mag_mask]
    dec_arr  = dec_arr[mag_mask]
    mag_arr  = mag_arr[mag_mask]
    size_arr = size_arr[mag_mask]
    entries  = [e for e, m in zip(catalog, mag_mask) if m]

    if len(entries) == 0:
        return overlay, []

    # --- Stage 2: angular-separation prefilter in RA/Dec space ---
    # Convert telescope pointing to RA/Dec so we can cull in sky coordinates.
    # Add a diagonal margin so objects near frame corners survive.
    diag_fov_rad = fov_rad * math.sqrt(2) * 0.55   # half-diagonal + 10% margin
    center_ra_h, center_dec_deg = altaz_to_radec(alt_deg, az_deg, lat_deg, lon_deg, t=t)
    center_ra_rad  = math.radians(center_ra_h * 15.0)
    center_dec_rad = math.radians(center_dec_deg)

    ra_rad   = np.radians(ra_arr)
    dec_rad  = np.radians(dec_arr)
    cos_sep  = (
        np.sin(center_dec_rad) * np.sin(dec_rad)
        + np.cos(center_dec_rad) * np.cos(dec_rad) * np.cos(ra_rad - center_ra_rad)
    )
    cos_sep  = np.clip(cos_sep, -1.0, 1.0)
    sep_mask = cos_sep >= math.cos(diag_fov_rad)

    ra_arr   = ra_arr[sep_mask]
    dec_arr  = dec_arr[sep_mask]
    mag_arr  = mag_arr[sep_mask]
    size_arr = size_arr[sep_mask]
    entries  = [e for e, m in zip(entries, sep_mask) if m]

    if len(entries) == 0:
        return overlay, []

    # --- Stage 3: vectorized RA/Dec → Alt/Az ---
    # Inline the coord.py math with numpy arrays; avoids Python-loop overhead.
    from datetime import timezone as _tz
    t_now = t if t is not None else datetime.now(tz=_tz.utc)
    jd    = _jd(t_now)
    lst_deg = _lst_deg(jd, lon_deg)
    lat_rad = math.radians(lat_deg)

    ha_arr  = np.radians((lst_deg - ra_arr) % 360.0)   # ra_arr is in degrees
    dec_rad = np.radians(dec_arr)

    alt_arr = np.arcsin(
        np.sin(dec_rad) * math.sin(lat_rad)
        + np.cos(dec_rad) * math.cos(lat_rad) * np.cos(ha_arr)
    )
    az_arr = np.arctan2(
        -np.sin(ha_arr) * np.cos(dec_rad),
        np.sin(dec_rad) * math.cos(lat_rad)
        - np.cos(dec_rad) * math.sin(lat_rad) * np.cos(ha_arr),
    ) % (2 * math.pi)

    # Gnomonic projection → tangent-plane (x_tan, y_tan) in radians
    x_tan, y_tan = _gnomonic_project(alt_arr, az_arr, center_alt, center_az)

    # Convert to pixel coords: origin at image center
    # x_tan > 0 → East → right in image  (+x)
    # y_tan > 0 → North-up on sky → up in image → smaller row index (-y)
    cx_pix = w // 2 + x_tan * scale   # float pixels
    cy_pix = h // 2 - y_tan * scale

    # Clip to frame (with margin so markers on edge still partially show)
    margin = 30
    visible = (
        np.isfinite(cx_pix) & np.isfinite(cy_pix)
        & (cx_pix >= -margin) & (cx_pix < w + margin)
        & (cy_pix >= -margin) & (cy_pix < h + margin)
    )

    table: list[dict] = []

    for idx in np.where(visible)[0]:
        entry  = entries[idx]
        px     = int(round(float(cx_pix[idx])))
        py     = int(round(float(cy_pix[idx])))
        color  = _COLORS.get(entry.obj_type, _COLORS[ObjType.STAR])
        mag    = float(mag_arr[idx])
        size_am = float(size_arr[idx])

        if entry.obj_type == ObjType.STAR:
            # Radius: mag -1.5→5px, mag 6.5→1px
            radius = max(1, round(5 - (mag + 1.5) * (4 / 8.0)))
            _draw_circle(overlay, px, py, radius, color, fill=True)
        else:
            # DSO: use angular size if available, else fixed 6px
            if size_am > 0:
                size_rad = math.radians(size_am / 60.0)
                radius = max(4, int(round(size_rad * scale * 0.5)))
                radius = min(radius, 40)
            else:
                radius = 6
            _draw_circle(overlay, px, py, radius, color, fill=False)
            _draw_cross(overlay, px, py, radius + 3, color)

        table.append({
            "name": entry.name,
            "mag":  mag,
            "type": entry.obj_type,
            "px":   px,
            "py":   py,
        })

    return overlay, table
