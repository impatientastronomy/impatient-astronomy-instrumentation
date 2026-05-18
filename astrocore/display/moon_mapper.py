"""
Moon feature overlay for the digital eyepiece.

Optical libration is computed via Meeus *Astronomical Algorithms* Ch.53,
using a truncated Ch.47 lunar position series (20 terms, ~0.2° accuracy —
sufficient for feature labeling).  No external dependencies required.

Public API
----------
load_moon_catalog(path)       -- load moon_features.csv → list[MoonFeature]
moon_libration(t)             -- (l', b', P) in degrees
compute_moon_overlay(...)     -- RGBA overlay + feature table
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

from astrocore.mount.coord import _jd


# Inclination of Moon's equatorial plane to the ecliptic (Meeus constant)
_I = math.radians(1.5424)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MoonFeature:
    name: str
    feature_type: str       # 'crater' | 'mare' | 'mons' | 'vallis' | 'rima' | 'other'
    lat_deg: float          # selenographic latitude (+N)
    lon_deg: float          # selenographic longitude (+E, IAU convention)
    diameter_km: float      # 0 if unknown/not applicable


_FEATURE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "crater": (230, 210, 170, 210),
    "mare":   (100, 170, 255, 160),
    "mons":   (150, 220, 120, 190),
    "vallis": (255, 180,  80, 190),
    "rima":   (255, 140, 140, 190),
    "other":  (200, 200, 200, 170),
}
_DEFAULT_COLOR = (200, 200, 200, 170)


def _normalize_type(t: str) -> str:
    t = t.lower().strip()
    for alias, mapped in (
        ("oceanus", "mare"), ("sinus", "mare"), ("lacus", "mare"),
        ("palus", "mare"),   ("montes", "mons"), ("mountain", "mons"),
        ("valley", "vallis"), ("rimae", "rima"), ("rille", "rima"),
        ("rupes", "vallis"),
    ):
        if alias in t:
            return mapped
    for key in ("crater", "mare", "mons", "vallis", "rima"):
        if key in t:
            return key
    return "other"


def load_moon_catalog(path: str) -> list[MoonFeature]:
    """Load moon_features.csv and return a list of MoonFeature objects."""
    features: list[MoonFeature] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features.append(MoonFeature(
                    name         = row["NAME"].strip(),
                    feature_type = _normalize_type(row.get("TYPE", "other")),
                    lat_deg      = float(row["LAT"]),
                    lon_deg      = float(row["LON"]),
                    diameter_km  = float(row["DIAMETER_KM"]) if row.get("DIAMETER_KM") else 0.0,
                ))
            except (ValueError, KeyError):
                continue
    return features


# ---------------------------------------------------------------------------
# Meeus Ch.47/53 — Moon position and libration
# ---------------------------------------------------------------------------

def _moon_ecliptic(T: float) -> tuple[float, float]:
    """
    Geocentric ecliptic longitude λ (degrees) and latitude β (degrees).
    Truncated Meeus Ch.47 series, dominant ~20 terms, ~0.2° accuracy.
    """
    D  = (297.8501921 + 445267.1114034 * T) % 360
    M  = (357.5291092 +  35999.0502909 * T) % 360
    Mp = (134.9633964 + 477198.8676313 * T) % 360
    F  = ( 93.2720950 + 483202.0175233 * T) % 360
    L0 = (218.3165463 + 481267.8813417 * T) % 360

    Dr, Mr, Mpr, Fr = (math.radians(x) for x in (D, M, Mp, F))

    # Longitude sum — coefficients in units of 1e-6 degrees (Meeus Table 47.A)
    Sl = (
          6288774 * math.sin(Mpr)
        + 1274027 * math.sin(2*Dr - Mpr)
        +  658314 * math.sin(2*Dr)
        +  213618 * math.sin(2*Mpr)
        -  185116 * math.sin(Mr)
        -  114332 * math.sin(2*Fr)
        +   58793 * math.sin(2*Dr - 2*Mpr)
        +   57066 * math.sin(2*Dr - Mr - Mpr)
        +   53322 * math.sin(2*Dr + Mpr)
        +   45758 * math.sin(2*Dr - Mr)
        -   40923 * math.sin(Mr - Mpr)
        -   34720 * math.sin(Dr)
        -   30383 * math.sin(Mr + Mpr)
        +   15327 * math.sin(2*Dr - 2*Fr)
        -   12528 * math.sin(Mpr + 2*Fr)
        +   10980 * math.sin(Mpr - 2*Fr)
        +   10675 * math.sin(4*Dr - Mpr)
        +   10034 * math.sin(3*Mpr)
        +    8548 * math.sin(4*Dr - 2*Mpr)
        -    7888 * math.sin(2*Dr + Mr - Mpr)
    )
    lam = (L0 + Sl * 1e-6) % 360   # degrees

    # Latitude sum — coefficients in units of 1e-6 degrees (Meeus Table 47.B)
    Sb = (
          5128122 * math.sin(Fr)
        +  280602 * math.sin(Mpr + Fr)
        +  277693 * math.sin(Mpr - Fr)
        +  173237 * math.sin(2*Dr - Fr)
        +   55413 * math.sin(2*Dr - Mpr + Fr)
        +   46271 * math.sin(2*Dr - Mpr - Fr)
        +   32573 * math.sin(2*Dr + Fr)
        +   17198 * math.sin(2*Mpr + Fr)
        +    9266 * math.sin(2*Dr + Mpr - Fr)
        +    8822 * math.sin(2*Mpr - Fr)
        +    8216 * math.sin(2*Dr - Mr - Fr)
        +    4324 * math.sin(2*Dr - 2*Mpr - Fr)
        +    4200 * math.sin(2*Dr + Mpr + Fr)
        -    3359 * math.sin(2*Dr + Mr - Fr)
    )
    beta = Sb * 1e-6   # degrees

    return lam, beta


def moon_libration(t: datetime | None = None) -> tuple[float, float, float]:
    """
    Compute optical libration parameters using Meeus *Astronomical Algorithms* Ch.53.

    Parameters
    ----------
    t : UTC datetime (defaults to now)

    Returns
    -------
    l_prime        : libration in longitude (degrees; + = eastern limb exposed)
    b_prime        : libration in latitude  (degrees; + = northern limb exposed)
    position_angle : position angle of Moon's N pole from celestial N (degrees, +E)
    """
    if t is None:
        t = datetime.now(tz=timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    T = (_jd(t) - 2451545.0) / 36525.0

    lam_deg, beta_deg = _moon_ecliptic(T)
    lam  = math.radians(lam_deg)
    beta = math.radians(beta_deg)

    Om_deg = (125.0445479 - 1934.1362608 * T) % 360
    Om     = math.radians(Om_deg)

    # Nutation in longitude — dominant term (arcsec → radians)
    dpsi = math.radians(-17.20 / 3600.0 * math.sin(Om))

    # Moon's argument of latitude
    F_rad = math.radians((93.2720950 + 483202.0175233 * T) % 360)

    # Obliquity of ecliptic (Meeus Ch.22, simplified)
    eps = math.radians(23.439291 - 0.013004 * T)

    # W = lambda - Delta_psi - Omega  (Meeus Eq.53.1)
    W = lam - dpsi - Om

    # Optical libration in longitude (Meeus Eq.53.2)
    # A' = arctan2(sin W cos β cos I − sin β sin I,  cos W cos β)
    A_prime = math.atan2(
        math.sin(W) * math.cos(beta) * math.cos(_I) - math.sin(beta) * math.sin(_I),
        math.cos(W) * math.cos(beta),
    )
    l_prime = math.degrees(A_prime - F_rad)
    l_prime = (l_prime + 180.0) % 360.0 - 180.0   # normalise to (-180, 180]

    # Optical libration in latitude (Meeus Eq.53.3)
    sin_b = (-math.sin(W) * math.cos(beta) * math.sin(_I)
             - math.sin(beta) * math.cos(_I))
    b_prime = math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))

    # Position angle of Moon's axis (Meeus Eq.53.4/53.5)
    # Physical libration terms are sub-0.04° — omitted for simplicity
    V   = Om + dpsi
    chi = math.atan2(
        math.sin(_I) * math.sin(V),
        math.sin(eps) * math.cos(_I) - math.cos(eps) * math.sin(_I) * math.cos(V),
    )
    position_angle = math.degrees(chi)

    return l_prime, b_prime, position_angle


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_pixel(img: np.ndarray, cx: int, cy: int,
               color: tuple[int, int, int, int]) -> None:
    """Write a 3×3 dot, clipping to image bounds."""
    h, w = img.shape[:2]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px, py = cx + dx, cy + dy
            if 0 <= px < w and 0 <= py < h:
                img[py, px] = color


def _draw_ring(img: np.ndarray, cx: int, cy: int, radius: int,
               color: tuple[int, int, int, int]) -> None:
    """Draw a thin circle outline, clipping to bounds."""
    h, w = img.shape[:2]
    r2_outer = radius * radius
    r2_inner = (radius - 1) * (radius - 1)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dx * dx + dy * dy
            if r2_inner < d2 <= r2_outer:
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = color


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

#: Mean angular radius of the Moon in degrees (used to compute disk size from FOV)
MOON_ANGULAR_RADIUS_DEG = 0.259


def compute_moon_overlay(
    features: Sequence[MoonFeature],
    moon_cx: float,
    moon_cy: float,
    moon_r: float,
    image_shape: tuple[int, int],       # (height, width)
    t: datetime | None = None,
    north_angle_deg: float = 0.0,       # degrees CW from image-up to celestial north
    min_diameter_km: float = 0.0,       # skip features smaller than this
    limb_fraction: float = 0.93,        # hide features within this fractional radius of limb
) -> tuple[np.ndarray, list[dict]]:
    """
    Build an RGBA moon feature overlay.

    Parameters
    ----------
    features         : output of load_moon_catalog()
    moon_cx          : Moon disk centre x in pixels
    moon_cy          : Moon disk centre y in pixels
    moon_r           : Moon disk radius in pixels
    image_shape      : (height, width) of the camera frame
    t                : UTC datetime (defaults to now)
    north_angle_deg  : angle from image up to celestial north, CW positive
    min_diameter_km  : filter features smaller than this (useful at small moon_r)
    limb_fraction    : features beyond this fraction of moon_r (near limb) are hidden

    Returns
    -------
    overlay : np.ndarray (H, W, 4) uint8 — transparent RGBA
    table   : list of dicts with keys 'name', 'type', 'px', 'py', 'diameter_km'
    """
    h, w = image_shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    if moon_r <= 0:
        return overlay, []

    l_prime, b_prime, P_axis = moon_libration(t)
    lp = math.radians(l_prime)
    bp = math.radians(b_prime)

    # Rotation from Moon-disk frame (east/north) to image frame (right/down).
    # P_axis degrees CCW from celestial north to Moon's north; north_angle_deg CW
    # from image-up to celestial north.  Combined angle in image frame:
    theta = math.radians(north_angle_deg + P_axis)
    ct, st = math.cos(theta), math.sin(theta)

    table: list[dict] = []

    for feat in features:
        if min_diameter_km > 0 and 0 < feat.diameter_km < min_diameter_km:
            continue

        lat = math.radians(feat.lat_deg)
        lon = math.radians(feat.lon_deg)

        # Orthographic projection onto sub-Earth plane.
        # z > 0 → near side; z <= 0 → far side (hidden).
        z = (math.cos(lat) * math.cos(lon - lp) * math.cos(bp)
             + math.sin(lat) * math.sin(bp))
        if z <= 0:
            continue

        x_disk = math.cos(lat) * math.sin(lon - lp)
        y_disk = (math.sin(lat) * math.cos(bp)
                  - math.cos(lat) * math.sin(bp) * math.cos(lon - lp))

        rho = math.sqrt(x_disk ** 2 + y_disk ** 2)
        if rho > limb_fraction:
            continue

        # Rotate from Moon's east/north to image right/down:
        #   east → +x (right),  north → -y (up, but y increases downward)
        pixel_dx = x_disk * ct + y_disk * st
        pixel_dy = x_disk * st - y_disk * ct

        px = int(round(moon_cx + pixel_dx * moon_r))
        py = int(round(moon_cy + pixel_dy * moon_r))

        if px < 0 or px >= w or py < 0 or py >= h:
            continue

        color = _FEATURE_COLORS.get(feat.feature_type, _DEFAULT_COLOR)

        if feat.feature_type == "mare":
            _put_pixel(overlay, px, py, (color[0], color[1], color[2], 80))
        else:
            _put_pixel(overlay, px, py, color)

        table.append({
            "name":        feat.name,
            "type":        feat.feature_type,
            "px":          px,
            "py":          py,
            "diameter_km": feat.diameter_km,
        })

    return overlay, table
