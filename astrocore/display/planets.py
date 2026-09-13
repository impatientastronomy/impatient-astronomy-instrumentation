"""
Low-precision planetary positions for the digital eyepiece overlay.

Uses the Standish/JPL Keplerian elements (valid 1800-2050 AD) with a
two-body Kepler solution per planet -- no external ephemeris dependency,
~1 arcmin accuracy in position and a few tenths of a magnitude in
brightness.  Same accuracy tier as astrocore.display.moon_mapper: fine for
"is this planet in the current field of view" marking, not for slewing.

Public API
----------
PLANET_NAMES           -- the 7 non-Earth major planets, Mercury..Neptune
planet_radec(name, t)  -- (ra_hours, dec_deg, mag) for one planet
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from astrocore.mount.coord import _jd

PLANET_NAMES: tuple[str, ...] = (
    "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
)

# Keplerian elements at J2000.0 and their rates per Julian century
# (Standish 1994, "Keplerian Elements for Approximate Positions of the
# Major Planets", valid 1800-2050 AD).
#   name: (a, a_dot, e, e_dot, i, i_dot, L, L_dot, peri, peri_dot, node, node_dot)
#   a in AU; e dimensionless; all angles in degrees; rates are per century.
_ELEMENTS: dict[str, tuple[float, ...]] = {
    "Mercury": (0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
                252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081),
    "Venus":   (0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
                181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418),
    "Earth":   (1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
                100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0),
    "Mars":    (1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
                -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343),
    "Jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
                34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106),
    "Saturn":  (9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
                49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794),
    "Uranus":  (19.18916464, -0.00196176, 0.04725744, -0.00004397, 0.77263783, -0.00242939,
                313.23810451, 428.48202785, 170.95427630, 0.40805281, 74.01692503, 0.04240589),
    "Neptune": (30.06992276, 0.00026291, 0.00859048, 0.00005105, 1.77004347, 0.00035372,
                -55.12002969, 218.45945325, 44.96476227, -0.32241464, 131.78422574, -0.00508664),
}

# Approximate apparent-magnitude formulas (Meeus, *Astronomical Algorithms*
# Ch.41): mag = v0 + 5*log10(r*delta) + c1*a + c2*a**2 + c3*a**3, with
# r = heliocentric distance, delta = geocentric distance (both AU), and
# a = phase angle in degrees.  Saturn's ring-brightness term (+/-0.9 mag
# depending on ring tilt) is omitted -- fine for a coarse marker.
_MAG_TERMS: dict[str, tuple[float, float, float, float]] = {
    "Mercury": (-0.42, 0.0380, -0.000273, 0.000002),
    "Venus":   (-4.40, 0.0009, 0.000239, -0.00000065),
    "Mars":    (-1.52, 0.016, 0.0, 0.0),
    "Jupiter": (-9.40, 0.005, 0.0, 0.0),
    "Saturn":  (-8.88, 0.0, 0.0, 0.0),
    "Uranus":  (-7.19, 0.0, 0.0, 0.0),
    "Neptune": (-6.87, 0.0, 0.0, 0.0),
}


def _solve_kepler(m_rad: float, e: float, tol: float = 1e-9, max_iter: int = 30) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for E, via Newton's method."""
    E = m_rad if e < 0.8 else math.pi
    for _ in range(max_iter):
        dE = (E - e * math.sin(E) - m_rad) / (1 - e * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def _heliocentric_ecliptic(name: str, T: float) -> tuple[float, float, float]:
    """Heliocentric ecliptic rectangular coordinates (AU) at Julian-century T."""
    a0, a_dot, e0, e_dot, i0, i_dot, L0, L_dot, peri0, peri_dot, node0, node_dot = _ELEMENTS[name]

    a    = a0 + a_dot * T
    e    = e0 + e_dot * T
    incl = math.radians(i0 + i_dot * T)
    L    = (L0 + L_dot * T) % 360.0
    peri = (peri0 + peri_dot * T) % 360.0
    node = (node0 + node_dot * T) % 360.0
    omega = math.radians((peri - node) % 360.0)   # argument of perihelion

    M = (L - peri + 180.0) % 360.0 - 180.0        # mean anomaly, normalised to (-180, 180]
    E = _solve_kepler(math.radians(M), e)

    x_p = a * (math.cos(E) - e)
    y_p = a * math.sqrt(1 - e * e) * math.sin(E)

    node_r = math.radians(node)
    cw, sw = math.cos(omega), math.sin(omega)
    co, so = math.cos(node_r), math.sin(node_r)
    ci, si = math.cos(incl), math.sin(incl)

    x = (cw * co - sw * so * ci) * x_p + (-sw * co - cw * so * ci) * y_p
    y = (cw * so + sw * co * ci) * x_p + (-sw * so + cw * co * ci) * y_p
    z = (sw * si) * x_p + (cw * si) * y_p
    return x, y, z


def planet_radec(name: str, t: datetime | None = None) -> tuple[float, float, float]:
    """
    Geocentric apparent RA/Dec and approximate visual magnitude for one planet.

    Parameters
    ----------
    name : one of PLANET_NAMES
    t    : UTC datetime (defaults to now)

    Returns
    -------
    (ra_hours, dec_deg, mag)
    """
    if t is None:
        t = datetime.now(tz=timezone.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)

    T = (_jd(t) - 2451545.0) / 36525.0

    x_p, y_p, z_p = _heliocentric_ecliptic(name, T)
    x_e, y_e, z_e = _heliocentric_ecliptic("Earth", T)
    x_g, y_g, z_g = x_p - x_e, y_p - y_e, z_p - z_e

    delta = math.sqrt(x_g * x_g + y_g * y_g + z_g * z_g)
    lam   = math.atan2(y_g, x_g)
    beta  = math.atan2(z_g, math.sqrt(x_g * x_g + y_g * y_g))

    eps = math.radians(23.439291 - 0.013004 * T)

    ra = math.atan2(
        math.sin(lam) * math.cos(eps) - math.tan(beta) * math.sin(eps),
        math.cos(lam),
    )
    dec = math.asin(
        math.sin(beta) * math.cos(eps) + math.cos(beta) * math.sin(eps) * math.sin(lam)
    )
    ra_deg = math.degrees(ra) % 360.0
    ra_hours = ra_deg / 15.0
    dec_deg = math.degrees(dec)

    r = math.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
    R = math.sqrt(x_e * x_e + y_e * y_e + z_e * z_e)
    cos_alpha = (r * r + delta * delta - R * R) / (2 * r * delta)
    alpha = math.degrees(math.acos(max(-1.0, min(1.0, cos_alpha))))

    v0, c1, c2, c3 = _MAG_TERMS[name]
    mag = v0 + 5 * math.log10(r * delta) + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

    return ra_hours, dec_deg, mag
