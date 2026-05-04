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
        camera_model="ZWO ASI1600MM Pro",
        timestamp=datetime.now(tz=timezone.utc),
        exposure_seconds=2.0,
        gain=139,
        offset=21,
        binning=1,
        temperature_c=21.0,
        filter_id=3,    # Ha
        flip=None,
    )
    return FrameMeta(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# frame_filename
# ---------------------------------------------------------------------------

class TestFrameFilename:
    def test_canonical_example(self):
        assert frame_filename(_make_meta(), 1) == "Cam3Bin1Flip0Filt3_2e6us_21C_001.tif"

    def test_index_zero_padded(self):
        meta = _make_meta()
        assert frame_filename(meta, 1).endswith("001.tif")
        assert frame_filename(meta, 42).endswith("042.tif")
        assert frame_filename(meta, 999).endswith("999.tif")

    def test_camera_id(self):
        assert frame_filename(_make_meta(camera_id=7), 1).startswith("Cam7")

    def test_binning(self):
        assert "Bin2" in frame_filename(_make_meta(binning=2), 1)

    def test_flip_none_is_zero(self):
        assert "Flip0" in frame_filename(_make_meta(flip=None), 1)

    def test_flip_horiz(self):
        assert "Flip1" in frame_filename(_make_meta(flip="HORIZ"), 1)

    def test_flip_vert(self):
        assert "Flip2" in frame_filename(_make_meta(flip="VERT"), 1)

    def test_flip_both(self):
        assert "Flip3" in frame_filename(_make_meta(flip="BOTH"), 1)

    def test_filter_id(self):
        assert "Filt5" in frame_filename(_make_meta(filter_id=5), 1)

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
    def test_2_seconds(self):
        assert _format_exposure(2.0) == "2e6us"

    def test_1_25_seconds(self):
        assert _format_exposure(1.25) == "1250e3us"

    def test_1_second(self):
        assert _format_exposure(1.0) == "1e6us"

    def test_0_1_seconds(self):
        assert _format_exposure(0.1) == "100e3us"

    def test_1_millisecond(self):
        assert _format_exposure(0.001) == "1e3us"

    def test_raw_microseconds_fallback(self):
        assert _format_exposure(0.0015) == "1500us"

    def test_zero(self):
        assert _format_exposure(0.0) == "0us"


# ---------------------------------------------------------------------------
# _format_temperature
# ---------------------------------------------------------------------------

class TestFormatTemperature:
    def test_positive(self):
        assert _format_temperature(21.0) == "21C"

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
        p = parse_filename("Cam3Bin1Flip0Filt3_2e6us_21C_001.tif")
        assert p is not None
        assert p["camera_id"]   == 3
        assert p["bin"]         == 1
        assert p["flip"]        == 0
        assert p["filter_id"]   == 3
        assert p["exposure_us"] == 2_000_000
        assert p["exposure_s"]  == pytest.approx(2.0)
        assert p["temperature_c"] == 21.0
        assert p["frame_index"] == 1

    def test_binning_2(self):
        p = parse_filename("Cam3Bin2Flip0Filt0_1e6us_20C_001.tif")
        assert p["bin"] == 2

    def test_flip_horiz(self):
        p = parse_filename("Cam3Bin1Flip1Filt0_1e6us_20C_001.tif")
        assert p["flip"] == 1

    def test_negative_temperature(self):
        p = parse_filename("Cam1Bin1Flip0Filt0_2e6us_m10C_042.tif")
        assert p is not None
        assert p["temperature_c"] == -10.0

    def test_unknown_temperature(self):
        p = parse_filename("Cam2Bin1Flip0Filt5_100e3us_unkC_007.tif")
        assert p is not None
        assert p["temperature_c"] is None

    def test_raw_microseconds(self):
        p = parse_filename("Cam1Bin1Flip0Filt1_1500us_20C_001.tif")
        assert p is not None
        assert p["exposure_us"] == 1500

    def test_no_match_returns_none(self):
        assert parse_filename("random_file.tif") is None
        assert parse_filename("C3_Ha_1250e3us_22C_001.tif") is None   # old format
        assert parse_filename("Cam3Bin1Flip0Filt3_2e6us_21C_001.jpg") is None
        assert parse_filename("") is None

    def test_round_trip(self):
        meta = _make_meta()
        filename = frame_filename(meta, 5)
        p = parse_filename(filename)
        assert p is not None
        assert p["camera_id"]   == meta.camera_id
        assert p["bin"]         == meta.binning
        assert p["filter_id"]   == meta.filter_id
        assert p["exposure_s"]  == pytest.approx(meta.exposure_seconds)
        assert p["temperature_c"] == round(meta.temperature_c)
        assert p["frame_index"] == 5
