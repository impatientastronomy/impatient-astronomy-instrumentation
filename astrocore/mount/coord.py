"""
Astronomical coordinate transformations: RA/Dec ↔ Alt/Az.

Pure functions, no state, stdlib only.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def radec_to_altaz(
    ra_hours: float,
    dec_deg: float,
    lat_deg: float,
    lon_deg: float,
    t: datetime | None = None,
) -> tuple[float, float]:
    """
    Convert equatorial coordinates to horizontal coordinates.

    Parameters
    ----------
    ra_hours : Right Ascension in decimal hours
    dec_deg  : Declination in degrees
    lat_deg  : Observer latitude in degrees  (+N)
    lon_deg  : Observer longitude in degrees (+E, West is negative)
    t        : UTC datetime; defaults to now

    Returns
    -------
    (alt_deg, az_deg) — altitude and azimuth in degrees.
    Azimuth convention: N=0, E=90, [0, 360).
    """
    if t is None:
        t = datetime.now(tz=timezone.utc)

    lst = _lst_deg(_jd(t), lon_deg)
    ha  = math.radians((lst - ra_hours * 15.0) % 360.0)
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)

    alt = math.asin(
        math.sin(dec) * math.sin(lat)
        + math.cos(dec) * math.cos(lat) * math.cos(ha)
    )
    az = math.atan2(
        -math.sin(ha) * math.cos(dec),
        math.sin(dec) * math.cos(lat) - math.cos(dec) * math.sin(lat) * math.cos(ha),
    )

    return math.degrees(alt), math.degrees(az) % 360.0


def altaz_to_radec(
    alt_deg: float,
    az_deg: float,
    lat_deg: float,
    lon_deg: float,
    t: datetime | None = None,
) -> tuple[float, float]:
    """
    Convert horizontal coordinates to equatorial coordinates.

    Parameters
    ----------
    alt_deg : Altitude in degrees
    az_deg  : Azimuth in degrees (N=0, E=90)
    lat_deg : Observer latitude in degrees  (+N)
    lon_deg : Observer longitude in degrees (+E, West is negative)
    t       : UTC datetime; defaults to now

    Returns
    -------
    (ra_hours, dec_deg)
    """
    if t is None:
        t = datetime.now(tz=timezone.utc)

    alt = math.radians(alt_deg)
    az  = math.radians(az_deg)
    lat = math.radians(lat_deg)

    dec = math.asin(
        math.sin(alt) * math.sin(lat)
        + math.cos(alt) * math.cos(lat) * math.cos(az)
    )
    sin_ha = -math.sin(az) * math.cos(alt) / math.cos(dec)
    cos_ha = (math.sin(alt) - math.sin(lat) * math.sin(dec)) / (
        math.cos(lat) * math.cos(dec)
    )
    ha_deg = math.degrees(math.atan2(sin_ha, cos_ha)) % 360.0

    lst    = _lst_deg(_jd(t), lon_deg)
    ra_deg = (lst - ha_deg) % 360.0

    return ra_deg / 15.0, math.degrees(dec)


# ── internal helpers ──────────────────────────────────────────────────────────

def _jd(t: datetime) -> float:
    """Julian Date for a UTC datetime."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return 2451545.0 + (t - j2000).total_seconds() / 86400.0


def _gst_deg(jd: float) -> float:
    """Greenwich Mean Sidereal Time in degrees (IAU polynomial approximation)."""
    T = (jd - 2451545.0) / 36525.0
    gst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T ** 2
        - T ** 3 / 38710000.0
    )
    return gst % 360.0


def _lst_deg(jd: float, lon_deg: float) -> float:
    """Local Sidereal Time in degrees."""
    return (_gst_deg(jd) + lon_deg) % 360.0
