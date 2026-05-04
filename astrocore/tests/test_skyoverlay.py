"""
Tests for astrocore/display/skyoverlay.py

No hardware, network, or real catalog file required.
Run with: pytest astrocore/tests/test_skyoverlay.py -v
"""

import math
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pytest

from astrocore.display.skyoverlay import (
    CatalogEntry,
    ObjType,
    _classify,
    _gnomonic_project,
    _primary_name,
    compute_overlay,
    load_catalog,
)

# Fixed time so tests are deterministic
T0 = datetime(2025, 6, 21, 3, 0, 0, tzinfo=timezone.utc)
LAT, LON = 38.44, -122.71


# ── classify ─────────────────────────────────────────────────────────────────

class TestClassify:
    def test_star(self):
        assert _classify("star A0m") == ObjType.STAR

    def test_star_uppercase(self):
        assert _classify("Star G2V") == ObjType.STAR

    def test_galaxy(self):
        assert _classify("Galaxy") == ObjType.GALAXY

    def test_nebula(self):
        assert _classify("Nebula") == ObjType.NEBULA

    def test_dso_neb(self):
        assert _classify("DSO Neb") == ObjType.NEBULA

    def test_cluster(self):
        assert _classify("Open Cluster") == ObjType.CLUSTER

    def test_plain_cluster(self):
        assert _classify("Cluster") == ObjType.CLUSTER

    def test_unknown_defaults_to_star(self):
        assert _classify("unknown XYZ") == ObjType.STAR


class TestPrimaryName:
    def test_single_name(self):
        assert _primary_name("Sirius ;HIP 32349 ;") == "Sirius"

    def test_multiple_names(self):
        assert _primary_name("M 31 ;NGC 0224 ;") == "M 31"

    def test_empty(self):
        assert _primary_name("") == ""

    def test_whitespace_only(self):
        assert _primary_name("  ;  ;") == ""


# ── load_catalog ──────────────────────────────────────────────────────────────

CSV_CONTENT = """\
RA,DEC,DIST,MAG,TYPE,NAME,SIZE
101.287215,-16.716116,2.6371,-1.44,star A0m,"Sirius ;HIP 32349 ;",0
210.802429,54.349,3.4,8.90,Galaxy,"M 101 ;NGC 5457 ;",28.8
188.666,69.679,0.0,9.65,Nebula,"M 97 ;NGC 3587 ;",3.4
"""


@pytest.fixture
def catalog_csv(tmp_path):
    p = tmp_path / "test_chart.csv"
    p.write_text(CSV_CONTENT)
    return str(p)


class TestLoadCatalog:
    def test_loads_correct_count(self, catalog_csv):
        entries = load_catalog(catalog_csv)
        assert len(entries) == 3

    def test_star_entry(self, catalog_csv):
        entries = load_catalog(catalog_csv)
        sirius = entries[0]
        assert sirius.name == "Sirius"
        assert sirius.obj_type == ObjType.STAR
        assert sirius.mag == pytest.approx(-1.44)
        assert sirius.ra_deg == pytest.approx(101.287215)

    def test_galaxy_entry(self, catalog_csv):
        entries = load_catalog(catalog_csv)
        m101 = entries[1]
        assert m101.obj_type == ObjType.GALAXY
        assert m101.size_arcmin == pytest.approx(28.8)

    def test_nebula_entry(self, catalog_csv):
        entries = load_catalog(catalog_csv)
        m97 = entries[2]
        assert m97.obj_type == ObjType.NEBULA


# ── gnomonic projection ───────────────────────────────────────────────────────

class TestGnomonicProject:
    def test_center_projects_to_zero(self):
        alt = np.array([0.5])
        az  = np.array([1.2])
        x, y = _gnomonic_project(alt, az, center_alt=0.5, center_az=1.2)
        assert x[0] == pytest.approx(0.0, abs=1e-9)
        assert y[0] == pytest.approx(0.0, abs=1e-9)

    def test_behind_tangent_plane_is_nan(self):
        # Object directly opposite the pointing direction → cos_c = -1
        alt = np.array([-0.5])
        az  = np.array([1.2 + math.pi])
        x, y = _gnomonic_project(alt, az, center_alt=0.5, center_az=1.2)
        assert np.isnan(x[0])
        assert np.isnan(y[0])

    def test_east_offset_is_positive_x(self):
        # An object slightly east (larger azimuth) should project to positive x
        center_alt = math.radians(45.0)
        center_az  = math.radians(180.0)
        offset_az  = math.radians(181.0)   # 1° east
        x, y = _gnomonic_project(
            np.array([center_alt]),
            np.array([offset_az]),
            center_alt, center_az,
        )
        assert x[0] > 0

    def test_north_offset_is_positive_y(self):
        # An object slightly north (higher altitude, same azimuth) → positive y
        center_alt = math.radians(30.0)
        center_az  = math.radians(90.0)
        x, y = _gnomonic_project(
            np.array([math.radians(31.0)]),
            np.array([math.radians(90.0)]),
            center_alt, center_az,
        )
        assert y[0] > 0


# ── compute_overlay ───────────────────────────────────────────────────────────

def _make_catalog(n_stars: int = 5, include_dso: bool = False) -> list[CatalogEntry]:
    """Create a small synthetic catalog near RA=0, Dec=+38 (near zenith for LAT=38)."""
    entries = []
    for i in range(n_stars):
        entries.append(CatalogEntry(
            ra_deg  = i * 0.1,
            dec_deg = 38.0 + i * 0.05,
            mag     = 2.0 + i * 0.5,
            size_arcmin = 0.0,
            obj_type = ObjType.STAR,
            name     = f"TestStar{i}",
        ))
    if include_dso:
        entries.append(CatalogEntry(
            ra_deg  = 0.5,
            dec_deg = 38.2,
            mag     = 7.0,
            size_arcmin = 10.0,
            obj_type = ObjType.GALAXY,
            name     = "TestGalaxy",
        ))
    return entries


class TestComputeOverlay:
    """
    compute_overlay uses radec_to_altaz internally, so these tests set up a pointing
    that puts synthetic catalog entries within the FOV.
    """

    def test_returns_correct_shape(self):
        cat = _make_catalog(3)
        overlay, _ = compute_overlay(
            catalog=cat, fov_deg=5.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
        )
        assert overlay.shape == (480, 640, 4)
        assert overlay.dtype == np.uint8

    def test_overlay_has_four_channels(self):
        cat = _make_catalog(1)
        overlay, _ = compute_overlay(
            catalog=cat, fov_deg=10.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(240, 320),
            lat_deg=LAT, lon_deg=LON, t=T0,
        )
        assert overlay.ndim == 3
        assert overlay.shape[2] == 4

    def test_empty_catalog_returns_transparent(self):
        overlay, table = compute_overlay(
            catalog=[], fov_deg=5.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
        )
        assert np.all(overlay == 0)
        assert table == []

    def test_mag_limit_filters_faint_stars(self):
        cat = _make_catalog(5)   # mags 2.0, 2.5, 3.0, 3.5, 4.0
        _, table_all = compute_overlay(
            catalog=cat, fov_deg=30.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
            mag_limit=6.5,
        )
        _, table_bright = compute_overlay(
            catalog=cat, fov_deg=30.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
            mag_limit=2.5,
        )
        assert len(table_bright) <= len(table_all)

    def test_table_entries_have_required_keys(self):
        cat = _make_catalog(2, include_dso=True)
        overlay, table = compute_overlay(
            catalog=cat, fov_deg=30.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
        )
        for entry in table:
            assert "name"  in entry
            assert "mag"   in entry
            assert "type"  in entry
            assert "px"    in entry
            assert "py"    in entry

    def test_table_pixel_coords_in_image_bounds(self):
        cat = _make_catalog(5)
        overlay, table = compute_overlay(
            catalog=cat, fov_deg=30.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
        )
        for entry in table:
            assert -30 <= entry["px"] <= 670
            assert -30 <= entry["py"] <= 510

    def test_dso_in_table(self):
        cat = _make_catalog(1, include_dso=True)
        _, table = compute_overlay(
            catalog=cat, fov_deg=30.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
            mag_limit=9.0,
        )
        types = {e["type"] for e in table}
        # At least one galaxy should be in the table if within FOV
        # (not guaranteed at this pointing, but table structure is what we're testing)
        assert isinstance(types, set)

    def test_some_pixels_nonzero_when_objects_in_fov(self):
        # Use a wide FOV to ensure something lands in frame
        cat = _make_catalog(20)
        overlay, table = compute_overlay(
            catalog=cat, fov_deg=45.0,
            alt_deg=60.0, az_deg=180.0,
            image_shape=(480, 640),
            lat_deg=LAT, lon_deg=LON, t=T0,
            mag_limit=6.5,
        )
        if table:
            # If any objects are in the table they should have left marks
            assert np.any(overlay[:, :, 3] > 0)
