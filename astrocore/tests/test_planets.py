"""
Tests for astrocore/display/planets.py

No hardware, network, or external ephemeris required.
Run with: pytest astrocore/tests/test_planets.py -v
"""

from datetime import datetime, timedelta, timezone

import pytest

from astrocore.display.planets import PLANET_NAMES, planet_radec

T_TEST = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class TestPlanetNames:
    def test_seven_planets(self):
        assert len(PLANET_NAMES) == 7

    def test_no_earth_or_pluto(self):
        assert "Earth" not in PLANET_NAMES
        assert "Pluto" not in PLANET_NAMES


class TestPlanetRadec:
    @pytest.mark.parametrize("name", PLANET_NAMES)
    def test_returns_three_floats(self, name):
        ra_h, dec_deg, mag = planet_radec(name, T_TEST)
        assert isinstance(ra_h, float)
        assert isinstance(dec_deg, float)
        assert isinstance(mag, float)

    @pytest.mark.parametrize("name", PLANET_NAMES)
    def test_ra_in_range(self, name):
        ra_h, _, _ = planet_radec(name, T_TEST)
        assert 0.0 <= ra_h < 24.0

    @pytest.mark.parametrize("name", PLANET_NAMES)
    def test_dec_in_range(self, name):
        _, dec_deg, _ = planet_radec(name, T_TEST)
        assert -90.0 <= dec_deg <= 90.0

    @pytest.mark.parametrize("name", PLANET_NAMES)
    def test_magnitude_plausible(self, name):
        # Loose sanity bounds -- brightest (Venus, ~-4.9) to faintest
        # (Neptune, ~8.0) naked-eye-to-telescopic planet magnitudes.
        _, _, mag = planet_radec(name, T_TEST)
        assert -5.0 < mag < 9.0

    def test_unknown_planet_raises(self):
        with pytest.raises(KeyError):
            planet_radec("Pluto", T_TEST)

    def test_defaults_to_now(self):
        # Just verify it doesn't raise
        ra_h, dec_deg, mag = planet_radec("Jupiter")
        assert 0.0 <= ra_h < 24.0
        assert -90.0 <= dec_deg <= 90.0

    def test_naive_datetime_treated_as_utc(self):
        t_naive = datetime(2024, 6, 15, 12, 0, 0)
        t_aware = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        r1 = planet_radec("Mars", t_naive)
        r2 = planet_radec("Mars", t_aware)
        assert r1 == pytest.approx(r2)

    def test_position_varies_over_time(self):
        # Outer planets move slowly, but Mercury/Venus should visibly shift
        # in RA over a few months.
        t0 = T_TEST
        t1 = t0 + timedelta(days=90)
        ra0, _, _ = planet_radec("Mercury", t0)
        ra1, _, _ = planet_radec("Mercury", t1)
        assert abs(ra0 - ra1) > 0.1

    def test_venus_brighter_than_neptune(self):
        _, _, venus_mag = planet_radec("Venus", T_TEST)
        _, _, neptune_mag = planet_radec("Neptune", T_TEST)
        assert venus_mag < neptune_mag
