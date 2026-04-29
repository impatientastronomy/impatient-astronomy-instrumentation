"""
Tests for astrocore/camera/naming.py.

No hardware required.
Run with: pytest astrocore/tests/test_naming.py -v
"""

from datetime import datetime, timezone

import pytest

from astrocore.camera.base import FrameMeta
from astrocore.camera.naming import (
    _format_exposure,
    _format_temperature,
    frame_filename,
    parse_filename,
)


def _make_meta(**kwargs) -> FrameMeta:
    defaults = dict(
        camera_id=3,
        camera_model="ZWO ASI294MC Pro",
        timestamp=datetime.now(tz=timezone.utc),
        exposure_seconds=1.25,
        gain=100,
        offset=10,
        binning=1,
        temperature_c=22.0,
        Filter="Ha",
    )
    return FrameMeta(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# frame_filename
# ---------------------------------------------------------------------------

class TestFrameFilename:
    def test_canonical_example(self):
        assert frame_filename(_make_meta(), 1) == "C3_Ha_1250e3us_22C_001.tif"

    def test_index_zero_padded(self):
        meta = _make_meta()
        assert frame_filename(meta, 1).endswith("001.tif")
        assert frame_filename(meta, 42).endswith("042.tif")
        assert frame_filename(meta, 999).endswith("999.tif")

    def test_camera_id(self):
        assert frame_filename(_make_meta(camera_id=7), 1).startswith("C7_")

    def test_filter_name(self):
        assert "_OIII_" in frame_filename(_make_meta(Filter="OIII"), 1)

    def test_negative_temperature(self):
        assert "_m10C_" in frame_filename(_make_meta(temperature_c=-10.0), 1)

    def test_unknown_temperature(self):
        assert "_unkC_" in frame_filename(_make_meta(temperature_c=None), 1)

    def test_temperature_rounded(self):
        assert "_23C_" in frame_filename(_make_meta(temperature_c=22.7), 1)

    def test_extension_is_tif(self):
        assert frame_filename(_make_meta(), 1).endswith(".tif")


# ---------------------------------------------------------------------------
# _format_exposure
# ---------------------------------------------------------------------------

class TestFormatExposure:
    def test_1_25_seconds(self):
        assert _format_exposure(1.25) == "1250e3us"

    def test_2_seconds(self):
        assert _format_exposure(2.0) == "2e6us"

    def test_1_second(self):
        assert _format_exposure(1.0) == "1e6us"

    def test_0_1_seconds(self):
        assert _format_exposure(0.1) == "100e3us"

    def test_1_millisecond(self):
        assert _format_exposure(0.001) == "1e3us"

    def test_raw_microseconds_fallback(self):
        # 1500 µs is not divisible by 1000
        assert _format_exposure(0.0015) == "1500us"

    def test_zero(self):
        assert _format_exposure(0.0) == "0us"


# ---------------------------------------------------------------------------
# _format_temperature
# ---------------------------------------------------------------------------

class TestFormatTemperature:
    def test_positive(self):
        assert _format_temperature(22.0) == "22C"

    def test_zero(self):
        assert _format_temperature(0.0) == "0C"

    def test_negative(self):
        assert _format_temperature(-10.0) == "m10C"

    def test_none(self):
        assert _format_temperature(None) == "unkC"

    def test_rounds_up(self):
        assert _format_temperature(22.7) == "23C"

    def test_rounds_down(self):
        assert _format_temperature(22.3) == "22C"

    def test_negative_near_zero_rounds_to_zero(self):
        assert _format_temperature(-0.4) == "0C"

    def test_negative_rounds_away_from_zero(self):
        assert _format_temperature(-10.6) == "m11C"


# ---------------------------------------------------------------------------
# parse_filename (round-trip with frame_filename)
# ---------------------------------------------------------------------------

class TestParseFilename:
    def test_canonical_example(self):
        p = parse_filename("C3_Ha_1250e3us_22C_001.tif")
        assert p is not None
        assert p["camera_id"] == 3
        assert p["filter_name"] == "Ha"
        assert p["exposure_us"] == 1_250_000
        assert p["exposure_s"] == pytest.approx(1.25)
        assert p["temperature_c"] == 22.0
        assert p["frame_index"] == 1

    def test_negative_temperature(self):
        p = parse_filename("C1_none_2e6us_m10C_042.tif")
        assert p is not None
        assert p["temperature_c"] == -10.0

    def test_unknown_temperature(self):
        p = parse_filename("C2_OIII_100e3us_unkC_007.tif")
        assert p is not None
        assert p["temperature_c"] is None

    def test_raw_microseconds(self):
        p = parse_filename("C1_Lum_1500us_20C_001.tif")
        assert p is not None
        assert p["exposure_us"] == 1500

    def test_no_match_returns_none(self):
        assert parse_filename("random_file.tif") is None
        assert parse_filename("C3_Ha_1250e3us_22C_001.jpg") is None
        assert parse_filename("") is None

    def test_round_trip(self):
        meta = _make_meta()
        filename = frame_filename(meta, 5)
        p = parse_filename(filename)
        assert p is not None
        assert p["camera_id"] == meta.camera_id
        assert p["filter_name"] == meta.Filter
        assert p["exposure_s"] == pytest.approx(meta.exposure_seconds)
        assert p["temperature_c"] == round(meta.temperature_c)
        assert p["frame_index"] == 5
