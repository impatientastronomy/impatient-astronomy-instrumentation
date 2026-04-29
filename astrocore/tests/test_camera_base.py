"""
Tests for astrocore/camera/base.py.

No hardware or SDK required — all pure Python.
Run with: pytest astrocore/tests/test_camera_base.py -v
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from astrocore.camera.base import (
    Camera,
    CameraError,
    CameraInfo,
    ConfigurationError,
    ConnectionError,
    ExposureError,
    ExposureStatus,
    Frame,
    FrameMeta,
)


# ---------------------------------------------------------------------------
# Minimal concrete Camera for testing the base class contract.
#
# Note: expose() is now a concrete method on Camera that calls the three
# primitives, so _FakeCamera only needs to implement those primitives.
# ---------------------------------------------------------------------------

def _make_meta(exposure_seconds: float = 1.0, gain: int = 0) -> FrameMeta:
    return FrameMeta(
        camera_id=1,
        camera_model="FakeCam",
        timestamp=datetime.now(tz=timezone.utc),
        exposure_seconds=exposure_seconds,
        gain=gain,
        offset=10,
        binning=1,
    )


class _FakeCamera(Camera):
    def __init__(self, has_cooler: bool = False) -> None:
        super().__init__()
        self._has_cooler = has_cooler
        self._status = ExposureStatus.IDLE
        self.exposure_seconds: float = 1.0
        self.gain: int = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def info(self) -> CameraInfo:
        self._require_connected()
        return CameraInfo(
            model="FakeCam",
            usb_index=0,
            sensor_width_px=100,
            sensor_height_px=100,
            pixel_size_um=4.63,
            bit_depth=16,
            camera_id=1,
            has_cooler=self._has_cooler,
        )

    def start_exposure(self, is_dark: bool = False) -> None:
        self._require_connected()
        self._status = ExposureStatus.SUCCESS  # instant for tests

    @property
    def exposure_status(self) -> ExposureStatus:
        return self._status

    def read_frame(self) -> Frame:
        self._require_connected()
        return Frame(
            data=np.zeros((100, 100), dtype=np.uint16),
            meta=_make_meta(self.exposure_seconds, self.gain),
        )

    def abort(self) -> None:
        self._status = ExposureStatus.IDLE


# ---------------------------------------------------------------------------
# CameraInfo
# ---------------------------------------------------------------------------

class TestCameraInfo:
    def test_frozen(self):
        info = _make_info()
        with pytest.raises(Exception):
            info.model = "Other"  # type: ignore[misc]

    def test_defaults(self):
        info = _make_info()
        assert info.camera_id == 0
        assert info.has_cooler is False
        assert info.bayer_pattern is None

    def test_usb_index_stored(self):
        info = CameraInfo(
            model="FakeCam", usb_index=2, sensor_width_px=100,
            sensor_height_px=100, pixel_size_um=3.76, bit_depth=12,
        )
        assert info.usb_index == 2


def _make_info(**kwargs) -> CameraInfo:
    defaults = dict(
        model="FakeCam", usb_index=0, sensor_width_px=100,
        sensor_height_px=100, pixel_size_um=4.63, bit_depth=16,
    )
    return CameraInfo(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# FrameMeta
# ---------------------------------------------------------------------------

class TestFrameMeta:
    def test_required_fields(self):
        meta = _make_frame_meta()
        assert meta.camera_id == 1
        assert meta.camera_model == "FakeCam"
        assert isinstance(meta.timestamp, datetime)

    def test_optional_fields_default_none(self):
        meta = _make_frame_meta()
        assert meta.temperature_c is None
        assert meta.bayer_pattern is None

    def test_extras_defaults_empty(self):
        meta = _make_frame_meta()
        assert meta.extras == {}

    def test_extras_accepts_arbitrary_data(self):
        meta = _make_frame_meta(extras={"target": "M42", "filter": "Ha"})
        assert meta.extras["target"] == "M42"


def _make_frame_meta(**kwargs) -> FrameMeta:
    defaults = dict(
        camera_id=1,
        camera_model="FakeCam",
        timestamp=datetime.now(tz=timezone.utc),
        exposure_seconds=2.0,
        gain=100,
        offset=10,
        binning=1,
    )
    return FrameMeta(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

class TestFrame:
    def test_frame_stores_meta(self):
        meta = _make_frame_meta(exposure_seconds=2.0)
        frame = Frame(data=np.zeros((10, 10), dtype=np.uint16), meta=meta)
        assert frame.meta.camera_id == 1
        assert frame.meta.exposure_seconds == 2.0

    def test_frame_data_is_numpy_array(self):
        frame = Frame(
            data=np.zeros((10, 10), dtype=np.uint16),
            meta=_make_frame_meta(),
        )
        assert isinstance(frame.data, np.ndarray)


# ---------------------------------------------------------------------------
# ExposureStatus
# ---------------------------------------------------------------------------

class TestExposureStatus:
    def test_all_states_exist(self):
        for name in ("IDLE", "WORKING", "SUCCESS", "FAILED"):
            assert hasattr(ExposureStatus, name)

    def test_states_are_distinct(self):
        states = [ExposureStatus.IDLE, ExposureStatus.WORKING,
                  ExposureStatus.SUCCESS, ExposureStatus.FAILED]
        assert len(set(states)) == 4


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_connection_error_is_camera_error(self):
        assert issubclass(ConnectionError, CameraError)

    def test_exposure_error_is_camera_error(self):
        assert issubclass(ExposureError, CameraError)

    def test_configuration_error_is_camera_error(self):
        assert issubclass(ConfigurationError, CameraError)

    def test_catch_all_works(self):
        with pytest.raises(CameraError):
            raise ExposureError("boom")


# ---------------------------------------------------------------------------
# Camera base class behaviour
# ---------------------------------------------------------------------------

class TestCameraBase:
    def test_context_manager_connects_and_disconnects(self):
        cam = _FakeCamera()
        with cam:
            assert cam._connected
        assert not cam._connected

    def test_context_manager_disconnects_on_exception(self):
        cam = _FakeCamera()
        with pytest.raises(ValueError):
            with cam:
                raise ValueError("mid-exposure error")
        assert not cam._connected

    def test_require_connected_raises_before_connect(self):
        cam = _FakeCamera()
        with pytest.raises(ConnectionError, match="not connected"):
            _ = cam.info

    def test_expose_convenience_returns_frame(self):
        with _FakeCamera() as cam:
            frame = cam.expose()
        assert isinstance(frame, Frame)
        assert isinstance(frame.data, np.ndarray)

    def test_expose_frame_meta_reflects_camera_state(self):
        with _FakeCamera() as cam:
            cam.exposure_seconds = 3.0
            cam.gain = 150
            frame = cam.expose()
        assert frame.meta.exposure_seconds == 3.0
        assert frame.meta.gain == 150

    def test_non_blocking_primitives(self):
        with _FakeCamera() as cam:
            assert cam.exposure_status == ExposureStatus.IDLE
            cam.start_exposure()
            assert cam.exposure_status == ExposureStatus.SUCCESS
            frame = cam.read_frame()
        assert isinstance(frame, Frame)

    def test_abort_resets_to_idle(self):
        with _FakeCamera() as cam:
            cam.start_exposure()
            cam.abort()
            assert cam.exposure_status == ExposureStatus.IDLE

    def test_no_cooler_raises_configuration_error(self):
        with _FakeCamera(has_cooler=False) as cam:
            with pytest.raises(ConfigurationError):
                cam.set_target_temperature(-10.0)

    def test_get_temperature_returns_none_by_default(self):
        with _FakeCamera() as cam:
            assert cam.get_temperature() is None
