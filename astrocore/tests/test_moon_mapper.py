"""
Tests for astrocore/display/moon_mapper.py

No hardware, network, or physical Moon required.
Run with: pytest astrocore/tests/test_moon_mapper.py -v
"""

import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from astrocore.display.moon_mapper import (
    MoonFeature,
    _moon_ecliptic,
    _normalize_type,
    compute_moon_overlay,
    load_moon_catalog,
    moon_libration,
    MOON_ANGULAR_RADIUS_DEG,
)

# ── _normalize_type ───────────────────────────────────────────────────────────

class TestNormalizeType:
    def test_crater(self):       assert _normalize_type("Crater")  == "crater"
    def test_mare(self):         assert _normalize_type("Mare")    == "mare"
    def test_oceanus(self):      assert _normalize_type("Oceanus") == "mare"
    def test_sinus(self):        assert _normalize_type("Sinus")   == "mare"
    def test_lacus(self):        assert _normalize_type("Lacus")   == "mare"
    def test_palus(self):        assert _normalize_type("Palus")   == "mare"
    def test_montes(self):       assert _normalize_type("Montes")  == "mons"
    def test_mons(self):         assert _normalize_type("Mons")    == "mons"
    def test_vallis(self):       assert _normalize_type("Vallis")  == "vallis"
    def test_rima(self):         assert _normalize_type("Rima")    == "rima"
    def test_rimae(self):        assert _normalize_type("Rimae")   == "rima"
    def test_rupes(self):        assert _normalize_type("Rupes")   == "vallis"
    def test_unknown(self):      assert _normalize_type("XYZ")     == "other"
    def test_case_insensitive(self): assert _normalize_type("CRATER") == "crater"


# ── _moon_ecliptic ────────────────────────────────────────────────────────────

class TestMoonEcliptic:
    def test_returns_two_floats(self):
        lam, beta = _moon_ecliptic(0.0)
        assert isinstance(lam, float)
        assert isinstance(beta, float)

    def test_longitude_in_range(self):
        for T in (-0.5, 0.0, 0.5, 1.0):
            lam, _ = _moon_ecliptic(T)
            assert 0.0 <= lam < 360.0

    def test_latitude_in_range(self):
        # Moon's ecliptic latitude never exceeds ~6°
        for T in (-0.5, 0.0, 0.5, 1.0):
            _, beta = _moon_ecliptic(T)
            assert -7.0 < beta < 7.0

    def test_longitude_near_j2000(self):
        # At J2000.0 (T=0), Moon should be near 218.3° (its mean longitude)
        lam, _ = _moon_ecliptic(0.0)
        # Allow ±10° for periodic terms
        assert 200.0 < lam < 240.0 or (lam < 10.0 or lam > 350.0)   # near 218 or wrapped

    def test_j2000_longitude_approximate(self):
        lam, _ = _moon_ecliptic(0.0)
        # Mean longitude at T=0 is 218.3°; true longitude should be within ~10°
        diff = min(abs(lam - 218.3), 360 - abs(lam - 218.3))
        assert diff < 12.0


# ── moon_libration ────────────────────────────────────────────────────────────

T_TEST = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class TestMoonLibration:
    def test_returns_three_floats(self):
        l, b, P = moon_libration(T_TEST)
        assert isinstance(l, float)
        assert isinstance(b, float)
        assert isinstance(P, float)

    def test_libration_longitude_range(self):
        # Optical libration in longitude never exceeds ±8°
        l, _, _ = moon_libration(T_TEST)
        assert -9.0 < l < 9.0

    def test_libration_latitude_range(self):
        # Optical libration in latitude never exceeds ±7°
        _, b, _ = moon_libration(T_TEST)
        assert -8.0 < b < 8.0

    def test_position_angle_range(self):
        # Position angle is in (-180, 180]
        _, _, P = moon_libration(T_TEST)
        assert -180.0 < P <= 180.0

    def test_defaults_to_now(self):
        # Just verify it doesn't raise
        l, b, P = moon_libration()
        assert -9.0 < l < 9.0
        assert -8.0 < b < 8.0

    def test_naive_datetime_treated_as_utc(self):
        t_naive = datetime(2024, 6, 15, 12, 0, 0)
        t_aware = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        l1, b1, P1 = moon_libration(t_naive)
        l2, b2, P2 = moon_libration(t_aware)
        assert abs(l1 - l2) < 1e-9
        assert abs(b1 - b2) < 1e-9

    def test_libration_varies_over_month(self):
        # Sample 10 dates a few days apart — libration should not be constant
        from datetime import timedelta
        t0 = datetime(2024, 3, 1, tzinfo=timezone.utc)
        ls = [moon_libration(t0 + timedelta(days=3*i))[0] for i in range(10)]
        assert max(ls) - min(ls) > 1.0   # must vary by at least 1°

    def test_known_approximate_libration(self):
        # On 2024-01-25 (near full moon), libration in longitude should be
        # within the physical range.  We can't hard-code the exact value without
        # a reference, but we verify the result is sensibly bounded.
        t = datetime(2024, 1, 25, 0, 0, tzinfo=timezone.utc)
        l, b, P = moon_libration(t)
        assert -9.0 < l < 9.0
        assert -8.0 < b < 8.0


# ── load_moon_catalog ─────────────────────────────────────────────────────────

_CSV = """\
NAME,TYPE,LAT,LON,DIAMETER_KM
Tycho,Crater,-43.3,-11.2,85
Mare Imbrium,Mare,32.8,-15.6,1123
Montes Apenninus,Montes,19.9,-3.2,600
Vallis Alpes,Vallis,49.0,3.2,166
Rima Hyginus,Rima,7.4,7.8,219
"""


@pytest.fixture
def small_catalog(tmp_path):
    p = tmp_path / "moon.csv"
    p.write_text(_CSV)
    return str(p)


class TestLoadMoonCatalog:
    def test_count(self, small_catalog):
        feats = load_moon_catalog(small_catalog)
        assert len(feats) == 5

    def test_crater(self, small_catalog):
        feats = load_moon_catalog(small_catalog)
        tycho = feats[0]
        assert tycho.name == "Tycho"
        assert tycho.feature_type == "crater"
        assert tycho.lat_deg == pytest.approx(-43.3)
        assert tycho.lon_deg == pytest.approx(-11.2)
        assert tycho.diameter_km == pytest.approx(85.0)

    def test_mare_type(self, small_catalog):
        feats = load_moon_catalog(small_catalog)
        assert feats[1].feature_type == "mare"

    def test_montes_type(self, small_catalog):
        feats = load_moon_catalog(small_catalog)
        assert feats[2].feature_type == "mons"

    def test_vallis_type(self, small_catalog):
        feats = load_moon_catalog(small_catalog)
        assert feats[3].feature_type == "vallis"

    def test_rima_type(self, small_catalog):
        feats = load_moon_catalog(small_catalog)
        assert feats[4].feature_type == "rima"

    def test_skips_bad_rows(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("NAME,TYPE,LAT,LON,DIAMETER_KM\nGood,Crater,-10.0,5.0,50\nBad,Crater,not_a_number,5.0,50\n")
        feats = load_moon_catalog(str(p))
        assert len(feats) == 1


# ── compute_moon_overlay ──────────────────────────────────────────────────────

def _simple_catalog() -> list[MoonFeature]:
    return [
        MoonFeature("Copernicus", "crater",  9.7, -20.1, 93),
        MoonFeature("Tycho",      "crater", -43.3, -11.2, 85),
        MoonFeature("Mare Imbrium", "mare", 32.8, -15.6, 1123),
        MoonFeature("Grimaldi",   "crater",  -5.2, -68.3, 222),  # near west limb
    ]


class TestComputeMoonOverlay:
    T0 = datetime(2024, 6, 21, 3, 0, 0, tzinfo=timezone.utc)

    def test_returns_correct_shape(self):
        feats = _simple_catalog()
        ov, _ = compute_moon_overlay(feats, 320, 240, 200,
                                     image_shape=(480, 640), t=self.T0)
        assert ov.shape == (480, 640, 4)
        assert ov.dtype == np.uint8

    def test_empty_features(self):
        ov, table = compute_moon_overlay([], 320, 240, 200,
                                         image_shape=(480, 640), t=self.T0)
        assert np.all(ov == 0)
        assert table == []

    def test_zero_radius_returns_empty(self):
        feats = _simple_catalog()
        ov, table = compute_moon_overlay(feats, 320, 240, 0,
                                         image_shape=(480, 640), t=self.T0)
        assert np.all(ov == 0)
        assert table == []

    def test_table_has_required_keys(self):
        feats = _simple_catalog()
        _, table = compute_moon_overlay(feats, 320, 240, 200,
                                        image_shape=(480, 640), t=self.T0)
        for entry in table:
            assert "name"        in entry
            assert "type"        in entry
            assert "px"          in entry
            assert "py"          in entry
            assert "diameter_km" in entry

    def test_pixels_within_moon_disk(self):
        feats = _simple_catalog()
        _, table = compute_moon_overlay(feats, 320, 240, 200,
                                        image_shape=(480, 640), t=self.T0)
        for entry in table:
            dx = entry["px"] - 320
            dy = entry["py"] - 240
            dist = math.sqrt(dx*dx + dy*dy)
            assert dist <= 200 * 1.01   # within disk radius (1% tolerance for rounding)

    def test_nonzero_pixels_painted(self):
        feats = [MoonFeature("Copernicus", "crater", 9.7, -20.1, 93)]
        ov, table = compute_moon_overlay(feats, 320, 240, 200,
                                         image_shape=(480, 640), t=self.T0)
        if table:
            assert np.any(ov[:, :, 3] > 0)

    def test_min_diameter_filter(self):
        feats = _simple_catalog()
        _, table_all = compute_moon_overlay(feats, 320, 240, 200,
                                            image_shape=(480, 640), t=self.T0,
                                            min_diameter_km=0)
        _, table_large = compute_moon_overlay(feats, 320, 240, 200,
                                              image_shape=(480, 640), t=self.T0,
                                              min_diameter_km=500)
        assert len(table_large) <= len(table_all)

    def test_north_angle_shifts_feature_position(self):
        feats = [MoonFeature("Copernicus", "crater", 9.7, -20.1, 93)]
        _, t0 = compute_moon_overlay(feats, 320, 240, 200,
                                     image_shape=(480, 640), t=self.T0,
                                     north_angle_deg=0)
        _, t90 = compute_moon_overlay(feats, 320, 240, 200,
                                      image_shape=(480, 640), t=self.T0,
                                      north_angle_deg=90)
        if t0 and t90:
            # Rotating the image frame by 90° should move the pixel position
            assert (t0[0]["px"] != t90[0]["px"]) or (t0[0]["py"] != t90[0]["py"])

    def test_far_side_feature_not_rendered(self):
        # A feature at lon=180° (far side) should never appear
        feats = [MoonFeature("FarSide", "crater", 0.0, 180.0, 100)]
        _, table = compute_moon_overlay(feats, 320, 240, 200,
                                        image_shape=(480, 640), t=self.T0)
        assert table == []

    def test_catalog_integration(self):
        catalog_path = Path(__file__).parent.parent / "mount" / "moon_features.csv"
        if not catalog_path.exists():
            pytest.skip("moon_features.csv not present")
        feats = load_moon_catalog(str(catalog_path))
        assert len(feats) > 100
        _, table = compute_moon_overlay(feats, 320, 240, 200,
                                        image_shape=(480, 640), t=self.T0)
        assert len(table) > 20   # plenty of features on the near side


# ── constants ────────────────────────────────────────────────────────────────

def test_moon_angular_radius_reasonable():
    assert 0.24 < MOON_ANGULAR_RADIUS_DEG < 0.28
