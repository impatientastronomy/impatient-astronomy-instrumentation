"""
Tests for astrocore/camera/meta.py — metadata embedding and extraction.

No hardware required.
Run with: pytest astrocore/tests/test_meta.py -v
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from astrocore.camera.base import Frame, FrameMeta
from astrocore.camera.meta import (
    embed_meta,
    extract_meta,
    _encode_row,
    _decode_row,
    _meta_to_json,
    _json_to_meta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(**kwargs) -> FrameMeta:
    defaults = dict(
        camera_id=2,
        camera_model="ZWO ASI294MC Pro",
        timestamp=datetime(2025, 6, 15, 22, 30, 0, tzinfo=timezone.utc),
        exposure_seconds=5.0,
        gain=100,
        offset=10,
        binning=1,
        temperature_c=-10.0,
        bayer_pattern="RGGB",
    )
    return FrameMeta(**{**defaults, **kwargs})


def _make_frame(height: int = 10, width: int = 200) -> Frame:
    return Frame(
        data=np.ones((height, width), dtype=np.uint16) * 32768,
        meta=_make_meta(),
    )


# ---------------------------------------------------------------------------
# Row encoding / decoding
# ---------------------------------------------------------------------------

class TestEncodeDecodeRow:
    def test_round_trip(self):
        text = '{"camera_id":2,"model":"FakeCam"}'
        row = _encode_row(text, width=100)
        assert _decode_row(row) == text

    def test_high_byte_is_zero(self):
        row = _encode_row("Hello", width=10)
        assert all((v >> 8) == 0 for v in row)

    def test_low_byte_is_ascii(self):
        row = _encode_row("Hi", width=10)
        assert row[0] == ord("H")
        assert row[1] == ord("i")

    def test_unused_pixels_are_zero(self):
        row = _encode_row("Hi", width=10)
        assert all(row[2:] == 0)

    def test_truncates_to_row_width(self):
        long_text = "A" * 500
        row = _encode_row(long_text, width=10)
        assert len(row) == 10
        assert all(v == ord("A") for v in row)

    def test_empty_string_produces_zero_row(self):
        row = _encode_row("", width=5)
        assert all(v == 0 for v in row)

    def test_decode_stops_at_null(self):
        row = np.zeros(10, dtype=np.uint16)
        row[0] = ord("A")
        row[1] = ord("B")
        # row[2] remains 0 (null terminator)
        row[3] = ord("C")
        assert _decode_row(row) == "AB"  # stops at null, C is not included


# ---------------------------------------------------------------------------
# JSON serialisation / deserialisation
# ---------------------------------------------------------------------------

class TestMetaJsonRoundTrip:
    def test_all_fields_survive_round_trip(self):
        meta = _make_meta()
        recovered = _json_to_meta(_meta_to_json(meta))
        assert recovered.camera_id == meta.camera_id
        assert recovered.camera_model == meta.camera_model
        assert recovered.exposure_seconds == meta.exposure_seconds
        assert recovered.gain == meta.gain
        assert recovered.temperature_c == meta.temperature_c
        assert recovered.bayer_pattern == meta.bayer_pattern

    def test_timestamp_round_trips_with_utc(self):
        meta = _make_meta()
        recovered = _json_to_meta(_meta_to_json(meta))
        assert recovered.timestamp == meta.timestamp
        assert recovered.timestamp.tzinfo is not None

    def test_extras_survive_round_trip(self):
        meta = _make_meta(extras={"target": "M42", "filter": "Ha"})
        recovered = _json_to_meta(_meta_to_json(meta))
        assert recovered.extras["target"] == "M42"
        assert recovered.extras["filter"] == "Ha"

    def test_none_fields_survive_round_trip(self):
        meta = _make_meta(temperature_c=None, bayer_pattern=None)
        recovered = _json_to_meta(_meta_to_json(meta))
        assert recovered.temperature_c is None
        assert recovered.bayer_pattern is None


# ---------------------------------------------------------------------------
# embed_meta
# ---------------------------------------------------------------------------

class TestEmbedMeta:
    def test_returns_new_frame(self):
        frame = _make_frame()
        result = embed_meta(frame)
        assert result is not frame

    def test_original_data_not_mutated(self):
        frame = _make_frame()
        original_first_row = frame.data[0, :].copy()
        embed_meta(frame)
        np.testing.assert_array_equal(frame.data[0, :], original_first_row)

    def test_first_row_high_bytes_are_zero(self):
        frame = _make_frame()
        result = embed_meta(frame)
        high_bytes = result.data[0, :] >> 8
        assert all(high_bytes == 0)

    def test_first_row_values_are_small(self):
        # Printable ASCII is 32–126, so max pixel value in first row ≤ 126
        frame = _make_frame()
        result = embed_meta(frame)
        assert result.data[0, :].max() <= 126

    def test_remaining_rows_unchanged(self):
        frame = _make_frame()
        result = embed_meta(frame)
        np.testing.assert_array_equal(result.data[1:, :], frame.data[1:, :])

    def test_meta_object_preserved(self):
        frame = _make_frame()
        result = embed_meta(frame)
        assert result.meta is frame.meta


# ---------------------------------------------------------------------------
# extract_meta
# ---------------------------------------------------------------------------

class TestExtractMeta:
    def test_round_trip_embed_then_extract(self):
        frame = _make_frame(width=500)  # wide enough for full JSON
        embedded = embed_meta(frame)
        recovered = extract_meta(embedded)
        assert recovered is not None
        assert recovered.camera_id == frame.meta.camera_id
        assert recovered.camera_model == frame.meta.camera_model
        assert recovered.exposure_seconds == frame.meta.exposure_seconds

    def test_returns_none_for_unembedded_frame(self):
        frame = _make_frame()
        # Raw frame with no metadata — first row is all high values (32768)
        assert extract_meta(frame) is None

    def test_returns_none_for_corrupt_first_row(self):
        frame = _make_frame()
        # Write garbage ASCII that isn't valid JSON
        frame.data[0, :5] = [ord(c) for c in "hello"]
        assert extract_meta(frame) is None

    def test_extras_survive_full_round_trip(self):
        meta = _make_meta(extras={"target": "NGC 1234"})
        frame = Frame(
            data=np.zeros((10, 500), dtype=np.uint16),
            meta=meta,
        )
        embedded = embed_meta(frame)
        recovered = extract_meta(embedded)
        assert recovered is not None
        assert recovered.extras["target"] == "NGC 1234"
