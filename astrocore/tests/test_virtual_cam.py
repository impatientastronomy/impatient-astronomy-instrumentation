"""
Tests for astrocore/camera/virtual_cam.py.

Creates real TIFF files in tmp_path — no physical camera required.
Run with: pytest astrocore/tests/test_virtual_cam.py -v
"""

import math
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from astrocore.camera.base import ConnectionError, ExposureStatus, Frame
from astrocore.camera.virtual_cam import VirtualCamera


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_frame(folder: Path, name: str, shape=(64, 64), value=1000):
    data = np.full(shape, value, dtype=np.uint16)
    tifffile.imwrite(str(folder / name), data)


def _populate(folder: Path):
    """Write a small library of canonical-named frames."""
    _write_frame(folder, "C3_Ha_2e6us_20C_001.tif", value=100)
    _write_frame(folder, "C3_Ha_2e6us_20C_002.tif", value=200)
    _write_frame(folder, "C3_Ha_2e6us_20C_003.tif", value=300)
    _write_frame(folder, "C3_Ha_1e6us_20C_001.tif", value=400)   # different exposure


@pytest.fixture
def folder(tmp_path):
    _populate(tmp_path)
    return tmp_path


@pytest.fixture
def cam(folder):
    c = VirtualCamera(folder, camera_id=3, t_accel=1_000_000)
    c.connect()
    yield c
    c.disconnect()


# ---------------------------------------------------------------------------
# Construction and connection
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_populates_file_table(self, cam):
        assert len(cam._file_table) == 4

    def test_connect_filters_by_camera_id(self, tmp_path):
        _write_frame(tmp_path, "C1_Ha_2e6us_20C_001.tif")
        _write_frame(tmp_path, "C3_Ha_2e6us_20C_001.tif")
        with VirtualCamera(tmp_path, camera_id=1) as c:
            assert all(c._file_table.camera_id == 1)

    def test_connect_infers_camera_id_when_not_specified(self, folder):
        with VirtualCamera(folder) as c:
            assert c._camera_id == 3

    def test_connect_reads_image_shape(self, cam):
        assert cam._image_shape == (64, 64)

    def test_connect_empty_folder_raises(self, tmp_path):
        with pytest.raises(ConnectionError):
            VirtualCamera(tmp_path).connect()

    def test_connect_wrong_camera_id_raises(self, folder):
        with pytest.raises(ConnectionError):
            VirtualCamera(folder, camera_id=99).connect()

    def test_context_manager_connects_and_disconnects(self, folder):
        with VirtualCamera(folder, camera_id=3) as c:
            assert c._connected
        assert not c._connected

    def test_disconnect_clears_file_table(self, cam):
        cam.disconnect()
        assert len(cam._file_table) == 0


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_sensor_dimensions(self, cam):
        info = cam.info
        assert info.sensor_width_px == 64
        assert info.sensor_height_px == 64

    def test_info_camera_id(self, cam):
        assert cam.info.camera_id == 3

    def test_info_bayer_pattern(self, folder):
        with VirtualCamera(folder, camera_id=3, bayer_pattern="RGGB") as c:
            assert c.info.bayer_pattern == "RGGB"

    def test_info_requires_connection(self, folder):
        c = VirtualCamera(folder, camera_id=3)
        with pytest.raises(ConnectionError):
            _ = c.info


# ---------------------------------------------------------------------------
# Exposure state machine
# ---------------------------------------------------------------------------

class TestExposureStateMachine:
    def test_initial_status_is_idle(self, cam):
        assert cam.exposure_status == ExposureStatus.IDLE

    def test_start_exposure_transitions_to_working(self, folder):
        # Use slow t_accel so WORKING state is observable
        with VirtualCamera(folder, camera_id=3, t_accel=0.001) as c:
            c.exposure_time = 10.0      # 10s / 0.001 = 10000s wall-clock → definitely WORKING
            c.start_exposure()
            assert c.exposure_status == ExposureStatus.WORKING

    def test_exposure_completes_to_success(self, cam):
        # t_accel=1_000_000 → any exposure completes in microseconds
        cam.exposure_time = 1.0
        cam.start_exposure()
        time.sleep(0.01)
        assert cam.exposure_status == ExposureStatus.SUCCESS

    def test_abort_resets_to_idle(self, folder):
        with VirtualCamera(folder, camera_id=3, t_accel=0.001) as c:
            c.exposure_time = 10.0
            c.start_exposure()
            c.abort()
            assert c.exposure_status == ExposureStatus.IDLE

    def test_read_frame_resets_to_idle(self, cam):
        cam.exposure_time = 2.0
        cam.start_exposure()
        time.sleep(0.01)
        cam.read_frame()
        assert cam.exposure_status == ExposureStatus.IDLE


# ---------------------------------------------------------------------------
# Frame reading and cycling
# ---------------------------------------------------------------------------

class TestReadFrame:
    def test_read_frame_returns_frame(self, cam):
        cam.exposure_time = 2.0
        cam.start_exposure()
        time.sleep(0.01)
        frame = cam.read_frame()
        assert isinstance(frame, Frame)
        assert isinstance(frame.data, np.ndarray)

    def test_frames_cycle_through_matching_exposure(self, cam):
        cam.exposure_time = 2.0    # matches 2e6us files (values 100, 200, 300)
        values = []
        for _ in range(3):
            cam._t_exp = time.monotonic() - 1   # force SUCCESS
            values.append(cam.read_frame().data[0, 0])
        assert sorted(values) == [100, 200, 300]

    def test_cursor_wraps_around(self, cam):
        cam.exposure_time = 2.0
        for _ in range(3):
            cam._t_exp = time.monotonic() - 1
            cam.read_frame()
        # 4th read should wrap back to first frame (value=100)
        cam._t_exp = time.monotonic() - 1
        frame = cam.read_frame()
        assert frame.data[0, 0] == 100

    def test_falls_back_to_any_frame_when_no_exposure_match(self, cam):
        cam.exposure_time = 99.0   # no files match this exposure
        cam._t_exp = time.monotonic() - 1
        frame = cam.read_frame()
        assert isinstance(frame, Frame)   # got something back

    def test_frame_meta_reflects_camera_settings(self, cam):
        cam.exposure_time = 2.0
        cam.gain = 150
        cam._t_exp = time.monotonic() - 1
        frame = cam.read_frame()
        assert frame.meta.gain == 150
        assert frame.meta.camera_id == 3

    def test_frame_meta_temperature_from_filename(self, cam):
        cam.exposure_time = 2.0
        cam._t_exp = time.monotonic() - 1
        frame = cam.read_frame()
        assert frame.meta.temperature_c == pytest.approx(20.0)

    def test_frame_meta_unknown_temperature_is_none(self, tmp_path):
        _write_frame(tmp_path, "C3_Ha_2e6us_unkC_001.tif")
        with VirtualCamera(tmp_path, camera_id=3, t_accel=1_000_000) as c:
            c._t_exp = time.monotonic() - 1
            frame = c.read_frame()
            assert frame.meta.temperature_c is None


# ---------------------------------------------------------------------------
# Non-blocking exposure only available after connect
# ---------------------------------------------------------------------------

class TestRequiresConnection:
    def test_start_exposure_requires_connection(self, folder):
        c = VirtualCamera(folder, camera_id=3)
        with pytest.raises(ConnectionError):
            c.start_exposure()

    def test_read_frame_requires_connection(self, folder):
        c = VirtualCamera(folder, camera_id=3)
        with pytest.raises(ConnectionError):
            c.read_frame()
