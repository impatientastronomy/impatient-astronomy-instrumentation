"""
Tests for astrocore/mount — coord transforms and LX200 codec.

No hardware or network required.
Run with: pytest astrocore/tests/test_mount.py -v
"""

import math
from datetime import datetime, timezone

import pytest

from astrocore.mount.coord import altaz_to_radec, radec_to_altaz
from astrocore.mount.lx200 import _dec_str, _parse_dec, _parse_ra, _ra_str

# Observer location used across tests
LAT = 38.44
LON = -122.71

# Fixed timestamp so tests are deterministic
T0 = datetime(2025, 6, 21, 3, 0, 0, tzinfo=timezone.utc)   # near summer solstice, 3 AM UTC


# ── LX200 codec ──────────────────────────────────────────────────────────────

class TestRaCodec:
    def test_roundtrip_whole_hours(self):
        assert _parse_ra(_ra_str(6.0)) == pytest.approx(6.0, abs=1 / 3600)

    def test_roundtrip_fractional(self):
        assert _parse_ra(_ra_str(14.5678)) == pytest.approx(14.5678, abs=1 / 3600)

    def test_zero(self):
        assert _parse_ra(_ra_str(0.0)) == pytest.approx(0.0, abs=1e-9)

    def test_near_24h(self):
        assert _parse_ra(_ra_str(23.9997)) == pytest.approx(23.9997, abs=1 / 3600)

    def test_format_is_hhmmss(self):
        s = _ra_str(6.5)   # 6h 30m 00s
        assert s == "06:30:00"

    def test_format_with_minutes(self):
        s = _ra_str(0.5)   # 0h 30m 00s
        assert s == "00:30:00"


class TestDecCodec:
    def test_roundtrip_positive(self):
        assert _parse_dec(_dec_str(45.25)) == pytest.approx(45.25, abs=1 / 3600)

    def test_roundtrip_negative(self):
        assert _parse_dec(_dec_str(-30.75)) == pytest.approx(-30.75, abs=1 / 3600)

    def test_zero(self):
        assert _parse_dec(_dec_str(0.0)) == pytest.approx(0.0, abs=1e-9)

    def test_positive_sign(self):
        assert _dec_str(10.0).startswith("+")

    def test_negative_sign(self):
        assert _dec_str(-10.0).startswith("-")

    def test_format_uses_asterisk(self):
        s = _dec_str(45.5)   # 45° 30' 00"
        assert s == "+45*30:00"

    def test_south_celestial_pole(self):
        assert _parse_dec(_dec_str(-90.0)) == pytest.approx(-90.0, abs=1 / 3600)


# ── Coordinate transforms ─────────────────────────────────────────────────────

class TestCoordRoundtrip:
    """altaz_to_radec(radec_to_altaz(ra, dec)) should recover the original coords."""

    def _roundtrip(self, ra: float, dec: float) -> None:
        alt, az = radec_to_altaz(ra, dec, LAT, LON, t=T0)
        ra2, dec2 = altaz_to_radec(alt, az, LAT, LON, t=T0)
        assert ra2 == pytest.approx(ra, abs=1e-6)
        assert dec2 == pytest.approx(dec, abs=1e-6)

    def test_typical_target(self):
        self._roundtrip(ra=5.5, dec=20.0)     # near Orion

    def test_near_zenith(self):
        self._roundtrip(ra=14.0, dec=40.0)

    def test_equator(self):
        self._roundtrip(ra=0.0, dec=0.0)

    def test_negative_declination(self):
        self._roundtrip(ra=6.75, dec=-16.7)   # Sirius-ish


class TestRadecToAltaz:
    def test_altitude_in_range(self):
        alt, _ = radec_to_altaz(5.5, 20.0, LAT, LON, t=T0)
        assert -90.0 <= alt <= 90.0

    def test_azimuth_in_range(self):
        _, az = radec_to_altaz(5.5, 20.0, LAT, LON, t=T0)
        assert 0.0 <= az < 360.0

    def test_zenith_altitude(self):
        # A star exactly at the zenith has alt = 90.
        # At the zenith: dec = lat, HA = 0 → ra = LST.
        from astrocore.mount.coord import _jd, _lst_deg
        lst = _lst_deg(_jd(T0), LON)
        ra_zenith = lst / 15.0
        alt, _ = radec_to_altaz(ra_zenith, LAT, LAT, LON, t=T0)
        assert alt == pytest.approx(90.0, abs=0.01)

    def test_defaults_to_now(self):
        # Just verifies it doesn't raise when t=None
        alt, az = radec_to_altaz(5.5, 20.0, LAT, LON)
        assert -90.0 <= alt <= 90.0
        assert 0.0 <= az < 360.0
