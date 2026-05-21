"""
Sky map overlay for the digital eyepiece.

Loads a star/DSO catalog and renders an RGBA overlay image aligned to the
current telescope pointing using a gnomonic (TAN) projection.

Public API
----------
load_catalog(path)                -- load skyChart.csv, return a CatalogEntry list
load_constellation_lines(path)    -- load constellation_lines.csv, return list of (ra1,dec1,ra2,dec2)
compute_overlay(...)              -- build an RGBA numpy array for blitting over the camera frame
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from astrocore.mount.coord import _jd, _lst_deg, altaz_to_radec
from astrocore.display.overlay_style import OverlayStyle


# ---------------------------------------------------------------------------
# Catalog types
# ---------------------------------------------------------------------------

class ObjType:
    STAR    = "star"
    GALAXY  = "galaxy"
    NEBULA  = "nebula"
    CLUSTER = "cluster"


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


def load_constellation_lines(
    lines_path: str,
    catalog_path: str,
) -> list[tuple[str, float, float, float, float]]:
    """
    Load constellation line segments as (con, ra1_deg, dec1_deg, ra2_deg, dec2_deg) tuples.

    Reads HIP ID pairs and constellation abbreviation from lines_path
    (constellation_lines.csv, columns HIP1,HIP2,CON) and resolves each HIP ID
    to RA/Dec from catalog_path (skyChart.csv).
    Lines whose HIP IDs are not found in the catalog are silently skipped.
    Comment lines (starting with #) are ignored.
    """
    # Build HIP→(RA, Dec) map from the star catalog
    hip_map: dict[int, tuple[float, float]] = {}
    with open(catalog_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"HIP\s+(\d+)", row.get("NAME", ""))
            if m:
                try:
                    hip_map[int(m.group(1))] = (float(row["RA"]), float(row["DEC"]))
                except (ValueError, KeyError):
                    pass

    segments: list[tuple[str, float, float, float, float]] = []
    with open(lines_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                h1, h2 = int(row[0].strip()), int(row[1].strip())
                con = row[2].strip() if len(row) > 2 else ""
            except (IndexError, ValueError):
                continue
            if h1 in hip_map and h2 in hip_map:
                ra1, dec1 = hip_map[h1]
                ra2, dec2 = hip_map[h2]
                segments.append((con, ra1, dec1, ra2, dec2))
    return segments


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


def _draw_line(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color: tuple[int, int, int, int],
) -> None:
    """Bresenham line, clipped to image bounds."""
    h, w = img.shape[:2]
    dx = abs(x1 - x0);  sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            if x == x1:
                break
            err += dy
            x += sx
        if e2 <= dx:
            if y == y1:
                break
            err += dx
            y += sy


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
    mag_limit:  float = 6.5,       # kept for backward compat; ignored when style given
    label_limit: float = 4.0,      # kept for backward compat; ignored when style given
    style:      OverlayStyle | None = None,
    constellation_lines: list[tuple[float, float, float, float]] | None = None,
    cam_fov_h_deg: float | None = None,   # horizontal FOV of camera to draw as box
    cam_fov_v_deg: float | None = None,   # vertical FOV of camera to draw as box
    cam_alt_deg:   float | None = None,   # FOV box centre; defaults to view centre
    cam_az_deg:    float | None = None,
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

    if style is None:
        style = OverlayStyle()

    fov_rad      = math.radians(fov_deg)
    center_alt   = math.radians(alt_deg)
    center_az    = math.radians(az_deg)

    # Scale factor: pixels per radian in tangent plane
    scale = w / fov_rad   # pixels per radian

    # Build full arrays from catalog
    ra_arr   = np.array([e.ra_deg      for e in catalog])
    dec_arr  = np.array([e.dec_deg     for e in catalog])
    mag_arr  = np.array([e.mag         for e in catalog])
    size_arr = np.array([e.size_arcmin for e in catalog])
    type_arr = np.array([e.obj_type    for e in catalog])

    # --- Stage 1: per-type magnitude filter ---
    # Use each object's own type limit; keep the loosest limit for the
    # angular pre-filter so we don't discard objects prematurely.
    type_limits = np.array([style.for_type(t).mag_limit for t in type_arr])
    mag_mask = mag_arr <= type_limits
    ra_arr   = ra_arr[mag_mask]
    dec_arr  = dec_arr[mag_mask]
    mag_arr  = mag_arr[mag_mask]
    size_arr = size_arr[mag_mask]
    entries  = [e for e, m in zip(catalog, mag_mask) if m]

    if len(entries) == 0:
        return overlay, []

    # --- Stage 2: angular-separation prefilter in RA/Dec space ---
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
    from datetime import timezone as _tz
    t_now = t if t is not None else datetime.now(tz=_tz.utc)
    jd    = _jd(t_now)
    lst_deg = _lst_deg(jd, lon_deg)
    lat_rad = math.radians(lat_deg)

    ha_arr  = np.radians((lst_deg - ra_arr) % 360.0)
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

    x_tan, y_tan = _gnomonic_project(alt_arr, az_arr, center_alt, center_az)

    cx_pix = w // 2 + x_tan * scale
    cy_pix = h // 2 - y_tan * scale

    margin = 30
    visible = (
        np.isfinite(cx_pix) & np.isfinite(cy_pix)
        & (cx_pix >= -margin) & (cx_pix < w + margin)
        & (cy_pix >= -margin) & (cy_pix < h + margin)
    )

    # --- Stage 4: enforce per-type max_count (keep brightest) ---
    vis_indices = list(np.where(visible)[0])
    type_counts: dict[str, int] = {}
    # Sort visible indices by magnitude (brightest first) for fair culling
    vis_indices.sort(key=lambda i: float(mag_arr[i]))
    kept = []
    for i in vis_indices:
        ot = entries[i].obj_type
        limit = style.for_type(ot).max_count
        if type_counts.get(ot, 0) < limit:
            type_counts[ot] = type_counts.get(ot, 0) + 1
            kept.append(i)

    table: list[dict] = []

    for idx in kept:
        entry   = entries[idx]
        px      = int(round(float(cx_pix[idx])))
        py      = int(round(float(cy_pix[idx])))
        ts      = style.for_type(entry.obj_type)
        color   = ts.rgba()
        mag     = float(mag_arr[idx])
        size_am = float(size_arr[idx])

        if entry.obj_type == ObjType.STAR:
            # Scale radius linearly from size_max (brightest) to size_min (at mag_limit)
            bright_mag = -1.5
            t_frac = (mag - bright_mag) / max(ts.mag_limit - bright_mag, 1e-6)
            radius = max(ts.size_min,
                         min(ts.size_max, round(ts.size_max - t_frac * (ts.size_max - ts.size_min))))
        else:
            if size_am > 0:
                size_rad = math.radians(size_am / 60.0)
                radius = int(round(size_rad * scale * 0.5))
            else:
                radius = ts.size_min
            radius = max(ts.size_min, min(ts.size_max, radius))

        _draw_marker(overlay, px, py, radius, color, ts.shape)

        table.append({
            "name":   entry.name,
            "mag":    mag,
            "type":   entry.obj_type,
            "px":     px,
            "py":     py,
            "radius": radius,
            "ra_deg":  entry.ra_deg,
            "dec_deg": entry.dec_deg,
        })

    # --- Stage 5: horizon line ---
    if style.horizon.show:
        _draw_horizon(overlay, center_alt, center_az, scale, w, h, style.horizon.rgba())

    # --- Stage 6: constellation lines ---
    if style.constellations.show and constellation_lines:
        con_labels = _draw_constellation_lines(
            overlay, constellation_lines,
            center_alt, center_az, scale, w, h,
            lat_rad, lst_deg,
            style.constellations.rgba(),
        )
        for con_abbr, lx, ly in con_labels:
            table.append({
                "name":   _CONSTELLATION_NAMES.get(con_abbr, con_abbr),
                "mag":    0.0,
                "type":   "constellation_label",
                "px":     lx,
                "py":     ly,
                "radius": 0,
            })

    # --- Stage 7: camera FOV box ---
    if (style.fov_box.show
            and cam_fov_h_deg is not None and cam_fov_v_deg is not None
            and cam_fov_h_deg > 0 and cam_fov_v_deg > 0):
        box_alt = math.radians(cam_alt_deg if cam_alt_deg is not None else alt_deg)
        box_az  = math.radians(cam_az_deg  if cam_az_deg  is not None else az_deg)
        _draw_fov_box(
            overlay, center_alt, center_az, scale, w, h,
            box_alt, box_az,
            math.radians(cam_fov_h_deg), math.radians(cam_fov_v_deg),
            style.fov_box.rgba(),
        )

    return overlay, table


def _draw_marker(
    overlay: np.ndarray,
    px: int, py: int,
    radius: int,
    color: tuple[int, int, int, int],
    shape: str,
) -> None:
    if shape == "dot":
        _draw_circle(overlay, px, py, radius, color, fill=True)
    elif shape == "circle":
        _draw_circle(overlay, px, py, radius, color, fill=False)
    elif shape == "circle_cross":
        _draw_circle(overlay, px, py, radius, color, fill=False)
        _draw_cross(overlay, px, py, radius + 3, color)
    elif shape == "cross":
        _draw_cross(overlay, px, py, radius + 3, color)


# ---------------------------------------------------------------------------
# Horizon, FOV box, constellation line drawing
# ---------------------------------------------------------------------------

def _draw_horizon(
    img: np.ndarray,
    center_alt: float,   # radians
    center_az:  float,   # radians
    scale:      float,   # pixels per radian
    w: int, h: int,
    color: tuple[int, int, int, int],
) -> None:
    """Draw the alt=0 horizon as a polyline across the overlay."""
    _HORIZON_STEPS = 360
    prev_px: int | None = None
    prev_py: int | None = None
    for i in range(_HORIZON_STEPS + 1):
        az = 2 * math.pi * i / _HORIZON_STEPS
        alt = np.array([0.0])
        az_a = np.array([az])
        x_t, y_t = _gnomonic_project(alt, az_a, center_alt, center_az)
        if not math.isfinite(x_t[0]) or not math.isfinite(y_t[0]):
            prev_px = prev_py = None
            continue
        px = int(round(w / 2 + x_t[0] * scale))
        py = int(round(h / 2 - y_t[0] * scale))
        margin = max(w, h)
        if abs(px - w // 2) > margin or abs(py - h // 2) > margin:
            prev_px = prev_py = None
            continue
        if prev_px is not None:
            _draw_line(img, prev_px, prev_py, px, py, color)
        prev_px, prev_py = px, py


def _altaz_to_pixel(
    alt_deg: float, az_deg: float,
    center_alt: float, center_az: float,
    scale: float, w: int, h: int,
) -> tuple[int, int] | None:
    alt_a = np.array([math.radians(alt_deg)])
    az_a  = np.array([math.radians(az_deg)])
    x_t, y_t = _gnomonic_project(alt_a, az_a, center_alt, center_az)
    if not math.isfinite(x_t[0]) or not math.isfinite(y_t[0]):
        return None
    px = int(round(w / 2 + x_t[0] * scale))
    py = int(round(h / 2 - y_t[0] * scale))
    return px, py


def _draw_fov_box(
    img: np.ndarray,
    center_alt: float,      # radians — view/projection centre (tangent plane origin)
    center_az:  float,      # radians
    scale:      float,
    w: int, h: int,
    box_center_alt: float,  # radians — where the box is centred (mount actual pointing)
    box_center_az:  float,  # radians
    fov_h_rad: float,       # camera horizontal FOV
    fov_v_rad: float,       # camera vertical FOV
    color: tuple[int, int, int, int],
) -> None:
    """Draw a rectangle on the overlay representing the camera's FOV."""
    half_h = math.degrees(fov_h_rad / 2)
    half_v = math.degrees(fov_v_rad / 2)
    ca_deg = math.degrees(box_center_alt)
    cz_deg = math.degrees(box_center_az)

    # Azimuth offsets shrink by cos(alt) on the sphere.  Divide by cos(alt) so
    # the projected corners span the true angular FOV width at all altitudes.
    cos_alt = math.cos(box_center_alt)
    half_h_az = half_h / max(cos_alt, 1e-4)   # guard against divide-by-zero near zenith

    corners_altaz = [
        (ca_deg + half_v, cz_deg - half_h_az),
        (ca_deg + half_v, cz_deg + half_h_az),
        (ca_deg - half_v, cz_deg + half_h_az),
        (ca_deg - half_v, cz_deg - half_h_az),
    ]
    pts = []
    for alt_c, az_c in corners_altaz:
        p = _altaz_to_pixel(alt_c, az_c, center_alt, center_az, scale, w, h)
        pts.append(p)

    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        if a is not None and b is not None:
            _draw_line(img, a[0], a[1], b[0], b[1], color)


_CONSTELLATION_NAMES: dict[str, str] = {
    "And": "Andromeda",   "Ant": "Antlia",      "Aps": "Apus",
    "Aqr": "Aquarius",    "Aql": "Aquila",       "Ara": "Ara",
    "Ari": "Aries",       "Aur": "Auriga",       "Boo": "Boötes",
    "CMa": "Canis Major", "CMi": "Canis Minor",  "Cap": "Capricornus",
    "Car": "Carina",      "Cas": "Cassiopeia",   "Cen": "Centaurus",
    "Cep": "Cepheus",     "Cet": "Cetus",        "Cha": "Chamaeleon",
    "Cir": "Circinus",    "Col": "Columba",      "Com": "Coma Ber.",
    "CrA": "Corona Aus.", "CrB": "Corona Bor.",  "Crv": "Corvus",
    "Crt": "Crater",      "Cru": "Crux",         "Cyg": "Cygnus",
    "Del": "Delphinus",   "Dor": "Dorado",       "Dra": "Draco",
    "Equ": "Equuleus",    "Eri": "Eridanus",     "For": "Fornax",
    "Gem": "Gemini",      "Gru": "Grus",         "Her": "Hercules",
    "Hor": "Horologium",  "Hya": "Hydra",        "Hyi": "Hydrus",
    "Ind": "Indus",       "Lac": "Lacerta",      "Leo": "Leo",
    "LMi": "Leo Minor",   "Lep": "Lepus",        "Lib": "Libra",
    "Lup": "Lupus",       "Lyn": "Lynx",         "Lyr": "Lyra",
    "Men": "Mensa",       "Mic": "Microscopium", "Mon": "Monoceros",
    "Mus": "Musca",       "Nor": "Norma",        "Oct": "Octans",
    "Oph": "Ophiuchus",   "Ori": "Orion",        "Pav": "Pavo",
    "Peg": "Pegasus",     "Per": "Perseus",      "Phe": "Phoenix",
    "Pic": "Pictor",      "PsA": "Piscis Aus.",  "Psc": "Pisces",
    "Pup": "Puppis",      "Pyx": "Pyxis",        "Ret": "Reticulum",
    "Scl": "Sculptor",    "Sco": "Scorpius",     "Sct": "Scutum",
    "Ser": "Serpens",     "Sex": "Sextans",      "Sge": "Sagitta",
    "Sgr": "Sagittarius", "Tau": "Taurus",       "Tel": "Telescopium",
    "TrA": "Tri. Aus.",   "Tri": "Triangulum",   "Tuc": "Tucana",
    "UMa": "Ursa Major",  "UMi": "Ursa Minor",   "Vel": "Vela",
    "Vir": "Virgo",       "Vol": "Volans",       "Vul": "Vulpecula",
}


def _draw_constellation_lines(
    img: np.ndarray,
    segments: list[tuple[str, float, float, float, float]],
    center_alt: float,   # radians
    center_az:  float,   # radians
    scale:      float,
    w: int, h: int,
    lat_rad: float,
    lst_deg: float,
    color: tuple[int, int, int, int],
) -> list[tuple[str, int, int]]:
    """Project and draw constellation line segments.

    Returns a list of (con_abbr, label_px, label_py) for each constellation
    that has at least one endpoint visible on screen.  The label position is
    the centroid of all on-screen projected endpoints.
    """
    margin = max(w, h) * 2   # allow lines that start/end off-screen

    # Per-constellation accumulator for label centroid
    con_pts: dict[str, list[tuple[int, int]]] = {}

    def _to_altaz(ra_deg: float, dec_deg_: float) -> tuple[float, float]:
        ha = math.radians((lst_deg - ra_deg) % 360.0)
        dec_r = math.radians(dec_deg_)
        alt = math.asin(
            math.sin(dec_r) * math.sin(lat_rad)
            + math.cos(dec_r) * math.cos(lat_rad) * math.cos(ha)
        )
        az = math.atan2(
            -math.sin(ha) * math.cos(dec_r),
            math.sin(dec_r) * math.cos(lat_rad)
            - math.cos(dec_r) * math.sin(lat_rad) * math.cos(ha),
        ) % (2 * math.pi)
        return alt, az

    for con, ra1_deg, dec1_deg, ra2_deg, dec2_deg in segments:
        alt1, az1 = _to_altaz(ra1_deg, dec1_deg)
        alt2, az2 = _to_altaz(ra2_deg, dec2_deg)

        x1_t, y1_t = _gnomonic_project(
            np.array([alt1]), np.array([az1]), center_alt, center_az)
        x2_t, y2_t = _gnomonic_project(
            np.array([alt2]), np.array([az2]), center_alt, center_az)

        if not (math.isfinite(x1_t[0]) and math.isfinite(y1_t[0])
                and math.isfinite(x2_t[0]) and math.isfinite(y2_t[0])):
            continue

        px1 = int(round(w / 2 + x1_t[0] * scale))
        py1 = int(round(h / 2 - y1_t[0] * scale))
        px2 = int(round(w / 2 + x2_t[0] * scale))
        py2 = int(round(h / 2 - y2_t[0] * scale))

        if (abs(px1 - w // 2) > margin and abs(px2 - w // 2) > margin):
            continue
        if (abs(py1 - h // 2) > margin and abs(py2 - h // 2) > margin):
            continue

        _draw_line(img, px1, py1, px2, py2, color)

        # Accumulate on-screen endpoints for label centroid
        if con:
            if con not in con_pts:
                con_pts[con] = []
            for px, py in ((px1, py1), (px2, py2)):
                if -margin < px < w + margin and -margin < py < h + margin:
                    con_pts[con].append((px, py))

    # Compute centroids; only keep those inside the image (with small margin)
    _LABEL_MARGIN = 20
    labels: list[tuple[str, int, int]] = []
    for con, pts in con_pts.items():
        if not pts:
            continue
        lx = int(sum(p[0] for p in pts) / len(pts))
        ly = int(sum(p[1] for p in pts) / len(pts))
        if (_LABEL_MARGIN <= lx < w - _LABEL_MARGIN
                and _LABEL_MARGIN <= ly < h - _LABEL_MARGIN):
            labels.append((con, lx, ly))

    return labels
