"""
Tests for astrocore/mount — coord transforms and LX200 codec.

No hardware or network required.
Run with: pytest astrocore/tests/test_mount.py -v
"""

import logging
import math
from datetime import datetime, timezone

import pytest

from astrocore.mount.coord import altaz_to_radec, radec_to_altaz
from astrocore.mount.lx200 import (
    Lx200Mount, _dec_str, _parse_dec, _parse_ra, _parse_site_coord, _ra_str,
)

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


# ── Lx200Mount socket behavior ───────────────────────────────────────────────
#
# FakeSocket models a real TCP stream: `stray` bytes are already sitting in
# the buffer and visible to recv() immediately (simulating a late/duplicate/
# unsolicited reply left over from a prior exchange); each entry in
# `responses` only becomes visible *after* the matching sendall() call,
# simulating a genuine request/response round trip.

class FakeSocket:
    def __init__(self, stray: bytes = b"", responses: list[bytes] | None = None):
        self._available = bytearray(stray)
        self._pending_responses = list(responses or [])
        self.sent: list[bytes] = []
        self._blocking = True
        self._timeout: float | None = None

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)
        if self._pending_responses:
            self._available += self._pending_responses.pop(0)

    def settimeout(self, t: float | None) -> None:
        self._timeout = t

    def gettimeout(self) -> float | None:
        return self._timeout

    def setblocking(self, flag: bool) -> None:
        self._blocking = flag

    def recv(self, bufsize: int) -> bytes:
        if not self._available:
            if self._blocking:
                raise OSError("FakeSocket: no data queued and blocking recv would hang")
            raise BlockingIOError()
        chunk = bytes(self._available[:bufsize])
        del self._available[:bufsize]
        return chunk

    def close(self) -> None:
        pass


def _mount_with_socket(sock: FakeSocket) -> Lx200Mount:
    mount = Lx200Mount("dummy-host")
    mount._sock = sock
    return mount


class TestDrainBeforeSend:
    def test_cmd_ignores_stray_bytes_ahead_of_its_own_response(self):
        # A stray reply ("13:19:38#") is already sitting in the buffer when
        # we go to send :GD# -- without draining first, _cmd() would return
        # the stray bytes instead of the real declination reply.
        sock  = FakeSocket(stray=b"13:19:38#", responses=[b"+42*10:05#"])
        mount = _mount_with_socket(sock)
        assert mount._cmd(":GD#") == "+42*10:05"

    def test_cmd1_ignores_stray_byte(self):
        sock  = FakeSocket(stray=b"X", responses=[b"1"])
        mount = _mount_with_socket(sock)
        assert mount._cmd1(":MS#") == "1"

    def test_cmdn_drains_before_sending(self):
        sock  = FakeSocket(stray=b"garbage")
        mount = _mount_with_socket(sock)
        mount._cmdn(":Q#")
        assert bytes(sock._available) == b""

    def test_drain_restores_prior_timeout(self):
        sock  = FakeSocket(responses=[b"+42*10:05#"])
        mount = _mount_with_socket(sock)
        sock.settimeout(7.5)
        mount._drain()
        assert sock.gettimeout() == 7.5


class TestQueryValidatedRetry:
    def test_get_ra_retries_past_a_misrouted_dec_reply(self, caplog):
        # First reply looks like a declination string (has '*') -- invalid
        # for :GR#. Second reply is a proper RA string.
        sock  = FakeSocket(responses=[b"+42*10:05#", b"13:19:38#"])
        mount = _mount_with_socket(sock)
        with caplog.at_level(logging.WARNING, logger="astrocore.mount.lx200"):
            ra = mount._get_ra()
        assert ra == pytest.approx(13 + 19 / 60 + 38 / 3600)
        assert any("Malformed :GR#" in r.message for r in caplog.records)

    def test_get_dec_retries_past_a_misrouted_ra_reply(self, caplog):
        sock  = FakeSocket(responses=[b"13:19:38#", b"+42*10:05#"])
        mount = _mount_with_socket(sock)
        with caplog.at_level(logging.WARNING, logger="astrocore.mount.lx200"):
            dec = mount._get_dec()
        assert dec == pytest.approx(42 + 10 / 60 + 5 / 3600)
        assert any("Malformed :GD#" in r.message for r in caplog.records)

    def test_get_ra_succeeds_immediately_on_valid_reply(self, caplog):
        sock  = FakeSocket(responses=[b"06:30:00#"])
        mount = _mount_with_socket(sock)
        with caplog.at_level(logging.WARNING, logger="astrocore.mount.lx200"):
            ra = mount._get_ra()
        assert ra == pytest.approx(6.5)
        assert not caplog.records


# ── _parse_site_coord ─────────────────────────────────────────────────────────

class TestParseSiteCoord:
    def test_degrees_minutes_no_seconds(self):
        assert _parse_site_coord("+37*51") == pytest.approx(37.85)

    def test_no_explicit_sign_is_positive(self):
        assert _parse_site_coord("37*51") == pytest.approx(37.85)

    def test_negative_sign(self):
        assert _parse_site_coord("-122*29") == pytest.approx(-122.4833333, abs=1e-6)

    def test_with_seconds(self):
        assert _parse_site_coord("-122*29:13") == pytest.approx(-122.4869444, abs=1e-6)

    def test_zero(self):
        assert _parse_site_coord("+00*00") == pytest.approx(0.0)


# ── Lx200Mount.site_location ────────────────────────────────────────────────

class TestSiteLocation:
    def test_reads_lat_and_converts_lon_west_to_east(self):
        # LX200 :Gg# is positive-west; site_location must return positive-east
        # to match astrocore.mount.coord's convention.
        sock  = FakeSocket(responses=[b"+37*51#", b"+122*29#"])
        mount = _mount_with_socket(sock)
        lat, lon = mount.site_location
        assert lat == pytest.approx(37.85)
        assert lon == pytest.approx(-122.4833333, abs=1e-6)

    def test_east_longitude_site_becomes_positive(self):
        # A site east of Greenwich reports a negative (west-convention) reply;
        # converted, it must come back positive.
        sock  = FakeSocket(responses=[b"+51*30#", b"-00*07#"])
        mount = _mount_with_socket(sock)
        lat, lon = mount.site_location
        assert lat == pytest.approx(51.5)
        assert lon == pytest.approx(0.1166667, abs=1e-6)

    def test_unconfigured_site_zero_zero_returns_none(self):
        sock  = FakeSocket(responses=[b"+00*00#", b"+00*00#"])
        mount = _mount_with_socket(sock)
        assert mount.site_location is None

    def test_malformed_reply_returns_none(self):
        sock  = FakeSocket(responses=[b"not a coordinate#", b"+122*29#"])
        mount = _mount_with_socket(sock)
        assert mount.site_location is None

    def test_not_connected_returns_none(self):
        mount = Lx200Mount("dummy-host")
        # _ensure() would try a real socket.connect() -- not connected, no
        # socket set, so the property must fail closed to None, not raise.
        assert mount._sock is None
        assert mount.site_location is None
