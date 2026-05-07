"""
Tests for astrocore/camera/frame_grabber.py.

No hardware required — FakeCamera controls exposure state directly.
Run with: pytest astrocore/tests/test_frame_grabber.py -v
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from astrocore.camera.base import (
    Camera,
    CameraInfo,
    ExposureStatus,
    Frame,
    FrameMeta,
)
from astrocore.camera.frame_grabber import (
    FrameGrabber,
    GrabResult,
    GrabStatus,
    _apply_flat_field,
    _crop_to_roi,
    _fix_dead_pixels_mask,
    _is_array,
    _subtract_dark,
)


# ---------------------------------------------------------------------------
# FakeCamera
# ---------------------------------------------------------------------------

def _make_meta(exposure_seconds: float = 2.0, temperature_c: float = 20.0) -> FrameMeta:
    return FrameMeta(
        camera_id=1,
        camera_model="FakeCam",
        timestamp=datetime.now(tz=timezone.utc),
        exposure_seconds=exposure_seconds,
        gain=0,
        offset=10,
        binning=1,
        temperature_c=temperature_c,
        filter_id=3,
    )


class _FakeCamera(Camera):
    """Camera whose exposure_status can be forced directly in tests."""

    def __init__(self) -> None:
        super().__init__()
        self._status = ExposureStatus.IDLE
        self.exposure_seconds: float = 2.0
        self.start_count = 0
        self.abort_count = 0
        self.read_count = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def info(self) -> CameraInfo:
        return CameraInfo(
            model="FakeCam", usb_index=0,
            sensor_width_px=100, sensor_height_px=100,
            pixel_size_um=4.63, bit_depth=16,
        )

    def start_exposure(self, is_dark: bool = False) -> None:
        self._status = ExposureStatus.WORKING
        self.start_count += 1

    @property
    def exposure_status(self) -> ExposureStatus:
        return self._status

    def set_status(self, status: ExposureStatus) -> None:
        self._status = status

    def read_frame(self) -> Frame:
        self.read_count += 1
        self._status = ExposureStatus.IDLE
        return Frame(
            data=np.ones((100, 100), dtype=np.uint16) * 1000,
            meta=_make_meta(self.exposure_seconds),
        )

    def abort(self) -> None:
        self._status = ExposureStatus.IDLE
        self.abort_count += 1


@pytest.fixture
def camera():
    cam = _FakeCamera()
    cam.connect()
    return cam


@pytest.fixture
def grabber(camera):
    return FrameGrabber(camera)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestGrabFrameStateMachine:
    def test_idle_starts_exposure(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        result = grabber.grab_frame(dark=False, flat=False)
        assert result.status == GrabStatus.STARTED
        assert camera.start_count == 1

    def test_failed_restarts_exposure(self, grabber, camera):
        camera.set_status(ExposureStatus.FAILED)
        result = grabber.grab_frame(dark=False, flat=False)
        assert result.status == GrabStatus.STARTED
        assert camera.start_count == 1

    def test_working_returns_working(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=False)
        camera.set_status(ExposureStatus.WORKING)
        result = grabber.grab_frame(dark=False, flat=False)
        assert result.status == GrabStatus.WORKING

    def test_success_returns_frame(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=False, flat=False)
        assert result.status == GrabStatus.SUCCESS
        assert isinstance(result.frame.data, np.ndarray)

    def test_success_immediately_starts_next_exposure(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=False)       # STARTED
        camera.set_status(ExposureStatus.SUCCESS)
        grabber.grab_frame(dark=False, flat=False)       # SUCCESS + re-arms
        assert camera.start_count == 2

    def test_timeout_aborts_and_reports(self, grabber, camera):
        from datetime import timedelta
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=False)
        grabber._deadline = datetime.now(tz=timezone.utc) - timedelta(seconds=1)
        camera.set_status(ExposureStatus.WORKING)
        result = grabber.grab_frame(dark=False, flat=False)
        assert result.status == GrabStatus.TIMEOUT
        assert camera.abort_count == 1
        assert "timed out" in result.message


# ---------------------------------------------------------------------------
# Exposure change
# ---------------------------------------------------------------------------

class TestExposureChange:
    def test_new_exposure_us_aborts_and_restarts(self, grabber, camera):
        camera.set_status(ExposureStatus.WORKING)
        camera.exposure_seconds = 1.0
        result = grabber.grab_frame(exposure_us=2_000_000, dark=False, flat=False)
        assert result.status == GrabStatus.STARTED
        assert camera.abort_count == 1
        assert camera.start_count == 1

    def test_new_exposure_drains_completed_frame(self, grabber, camera):
        camera.set_status(ExposureStatus.SUCCESS)
        camera.exposure_seconds = 1.0
        grabber.grab_frame(exposure_us=2_000_000, dark=False, flat=False)
        assert camera.read_count == 1   # drained

    def test_new_exposure_invalidates_dark(self, grabber, camera):
        grabber.imDark = np.zeros((100, 100), dtype=np.uint16)
        camera.set_status(ExposureStatus.IDLE)
        camera.exposure_seconds = 1.0
        grabber.grab_frame(exposure_us=2_000_000, dark=False, flat=False)
        assert not _is_array(grabber.imDark)

    def test_same_exposure_does_not_restart(self, grabber, camera):
        camera.exposure_seconds = 2.0
        camera.set_status(ExposureStatus.WORKING)
        result = grabber.grab_frame(exposure_us=2_000_000, dark=False, flat=False)
        assert result.status == GrabStatus.WORKING
        assert camera.abort_count == 0

    def test_camera_exposure_updated(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        camera.exposure_seconds = 1.0
        grabber.grab_frame(exposure_us=3_000_000, dark=False, flat=False)
        assert camera.exposure_seconds == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_aborts_working_exposure(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=False)
        camera.set_status(ExposureStatus.WORKING)
        grabber.reset()
        assert camera.abort_count == 1
        assert grabber._deadline is None

    def test_reset_drains_completed_frame(self, grabber, camera):
        camera.set_status(ExposureStatus.SUCCESS)
        grabber.reset()
        assert camera.read_count == 1


# ---------------------------------------------------------------------------
# Dark pipeline integration
# ---------------------------------------------------------------------------

class TestDarkPipelineIntegration:
    def test_dark_subtracted_from_frame(self, grabber, camera):
        grabber.imDark = np.full((100, 100), 200, dtype=np.uint16)
        grabber.current_dark = "C1_Ha_2e6us_20C_001.tif"
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=True, flat=False, dpc=False, demosaic=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=True, flat=False, dpc=False, demosaic=False)
        assert result.status == GrabStatus.SUCCESS
        # FakeCamera returns 1000; dark is 200 → expected 800
        assert result.frame.data[0, 0] == 800

    def test_warns_and_continues_when_no_dark_found(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=True, flat=False, dpc=False, demosaic=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=True, flat=False, dpc=False, demosaic=False)
        assert result.status == GrabStatus.SUCCESS
        assert result.calibrated is False

    def test_calibrated_true_when_dark_succeeds(self, grabber, camera):
        grabber.imDark = np.full((100, 100), 200, dtype=np.uint16)
        grabber.current_dark = "C1_Ha_2e6us_20C_001.tif"
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=True, flat=False, dpc=False, demosaic=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=True, flat=False, dpc=False, demosaic=False)
        assert result.calibrated is True


# ---------------------------------------------------------------------------
# Flat pipeline integration
# ---------------------------------------------------------------------------

class TestFlatPipelineIntegration:
    def test_flat_applied_to_frame(self, grabber, camera):
        grabber.imFlat = np.full((100, 100), 2000, dtype=np.float32)
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=True, dpc=False, demosaic=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=False, flat=True, dpc=False, demosaic=False)
        assert result.status == GrabStatus.SUCCESS
        # FakeCamera returns 1000; imFlat=2000 → 1000 * 2000/1000 = 2000
        assert result.frame.data[0, 0] == 2000

    def test_warns_and_continues_when_no_flat_found(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=True, dpc=False, demosaic=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=False, flat=True, dpc=False, demosaic=False)
        assert result.status == GrabStatus.SUCCESS
        assert result.calibrated is False

    def test_calibrated_true_when_no_cal_requested(self, grabber, camera):
        camera.set_status(ExposureStatus.IDLE)
        grabber.grab_frame(dark=False, flat=False, dpc=False, demosaic=False)
        camera.set_status(ExposureStatus.SUCCESS)
        result = grabber.grab_frame(dark=False, flat=False, dpc=False, demosaic=False)
        assert result.calibrated is True


# ---------------------------------------------------------------------------
# _subtract_dark
# ---------------------------------------------------------------------------

class TestSubtractDark:
    def test_subtracts_dark(self):
        imR  = np.full((5, 5), 1000, dtype=np.uint16)
        dark = np.full((5, 5),  200, dtype=np.uint16)
        result = _subtract_dark(imR, dark)
        assert result[0, 0] == 800

    def test_clamps_to_zero(self):
        imR  = np.full((5, 5), 100, dtype=np.uint16)
        dark = np.full((5, 5), 500, dtype=np.uint16)
        result = _subtract_dark(imR, dark)
        assert result.min() == 0

    def test_preserves_uint16_dtype(self):
        imR  = np.full((5, 5), 1000, dtype=np.uint16)
        dark = np.full((5, 5),  200, dtype=np.uint16)
        assert _subtract_dark(imR, dark).dtype == np.uint16


# ---------------------------------------------------------------------------
# _apply_flat_field
# ---------------------------------------------------------------------------

class TestApplyFlatField:
    def test_unity_gain_unchanged(self):
        imR    = np.full((5, 5), 1000, dtype=np.uint16)
        imFlat = np.full((5, 5), 1000, dtype=np.float32)
        result = _apply_flat_field(imR, imFlat)
        assert result[0, 0] == 1000

    def test_double_gain(self):
        imR    = np.full((5, 5), 1000, dtype=np.uint16)
        imFlat = np.full((5, 5), 2000, dtype=np.float32)
        result = _apply_flat_field(imR, imFlat)
        assert result[0, 0] == 2000

    def test_half_gain(self):
        imR    = np.full((5, 5), 1000, dtype=np.uint16)
        imFlat = np.full((5, 5),  500, dtype=np.float32)
        result = _apply_flat_field(imR, imFlat)
        assert result[0, 0] == 500

    def test_clamps_to_65535(self):
        imR    = np.full((5, 5), 60000, dtype=np.uint16)
        imFlat = np.full((5, 5),  2000, dtype=np.float32)
        result = _apply_flat_field(imR, imFlat)
        assert result.max() == 65535

    def test_preserves_uint16_dtype(self):
        imR    = np.full((5, 5), 1000, dtype=np.uint16)
        imFlat = np.full((5, 5), 1000, dtype=np.float32)
        assert _apply_flat_field(imR, imFlat).dtype == np.uint16


# ---------------------------------------------------------------------------
# _fix_dead_pixels_mask  (2nd-neighbor, boolean mask)
# ---------------------------------------------------------------------------

class TestFixDeadPixelsMask:
    def test_replaces_dead_pixel_with_2nd_neighbor_mean(self):
        data = np.zeros((9, 9), dtype=np.uint16)
        data[4, 4] = 9999   # dead pixel
        data[2, 4] = 100    # 2nd neighbor above
        data[6, 4] = 200    # 2nd neighbor below
        data[4, 2] = 300    # 2nd neighbor left
        data[4, 6] = 400    # 2nd neighbor right
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        result = _fix_dead_pixels_mask(data, mask)
        assert result[4, 4] == pytest.approx(250)

    def test_good_pixels_unchanged(self):
        data = np.ones((9, 9), dtype=np.uint16) * 500
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        original = data.copy()
        result = _fix_dead_pixels_mask(data, mask)
        mask_inv = ~mask
        np.testing.assert_array_equal(result[mask_inv], original[mask_inv])

    def test_empty_mask_unchanged(self):
        data = np.ones((9, 9), dtype=np.uint16) * 500
        mask = np.zeros((9, 9), dtype=bool)
        result = _fix_dead_pixels_mask(data, mask)
        np.testing.assert_array_equal(result, data)

    def test_does_not_mutate_input(self):
        data = np.zeros((9, 9), dtype=np.uint16)
        data[4, 4] = 9999
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        original = data.copy()
        _fix_dead_pixels_mask(data, mask)
        np.testing.assert_array_equal(data, original)


# ---------------------------------------------------------------------------
# _crop_to_roi
# ---------------------------------------------------------------------------

class TestCropToRoi:
    def _meta(self, **kwargs):
        defaults = dict(
            camera_id=1, camera_model="FC",
            timestamp=datetime.now(tz=timezone.utc),
            exposure_seconds=1.0, gain=0, offset=0, binning=1,
            roi_x=None, roi_y=None, roi_width=None, roi_height=None,
        )
        return FrameMeta(**{**defaults, **kwargs})

    def test_no_roi_returns_full_array(self):
        data = np.ones((10, 10), dtype=np.uint16)
        meta = self._meta()
        result = _crop_to_roi(data, meta)
        assert result.shape == (10, 10)

    def test_crops_to_roi(self):
        data = np.arange(100, dtype=np.uint16).reshape(10, 10)
        meta = self._meta(roi_x=2, roi_y=3, roi_width=4, roi_height=2)
        result = _crop_to_roi(data, meta)
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result, data[3:5, 2:6])


# ---------------------------------------------------------------------------
# Dark table matching
# ---------------------------------------------------------------------------

class TestFindBestDarkRow:
    def _meta(self, camera_id=1, exposure_s=2.0, temperature_c=20.0):
        return _make_meta(exposure_seconds=exposure_s, temperature_c=temperature_c)

    def _make_dark_table(self, rows):
        return pd.DataFrame(rows, columns=[
            "filename", "path", "camera_id", "bin", "flip",
            "exposure_us", "exposure_s", "temperature_c", "frame_index",
        ])

    def test_returns_none_when_table_empty(self, grabber):
        assert grabber._find_best_dark_row(self._meta()) is None

    def test_matches_camera_bin_and_exposure(self, grabber):
        grabber.dark_table = self._make_dark_table([
            ("Cam1Bin1Flip0Filt0_2e6us_20C_001.tif", None, 1, 1, 0, 2_000_000, 2.0, 20.0, 1),
            ("Cam2Bin1Flip0Filt0_2e6us_20C_001.tif", None, 2, 1, 0, 2_000_000, 2.0, 20.0, 1),
        ])
        result = grabber._find_best_dark_row(self._meta())
        assert result["filename"] == "Cam1Bin1Flip0Filt0_2e6us_20C_001.tif"

    def test_bin_mismatch_returns_none(self, grabber):
        grabber.dark_table = self._make_dark_table([
            ("Cam1Bin2Flip0Filt0_2e6us_20C_001.tif", None, 1, 2, 0, 2_000_000, 2.0, 20.0, 1),
        ])
        # meta has binning=1, table has bin=2 → no match
        assert grabber._find_best_dark_row(self._meta()) is None

    def test_selects_closest_temperature(self, grabber):
        grabber.dark_table = self._make_dark_table([
            ("Cam1Bin1Flip0Filt0_2e6us_18C_001.tif", None, 1, 1, 0, 2_000_000, 2.0, 18.0, 1),
            ("Cam1Bin1Flip0Filt0_2e6us_22C_001.tif", None, 1, 1, 0, 2_000_000, 2.0, 22.0, 2),
        ])
        result = grabber._find_best_dark_row(self._meta(temperature_c=21.0))
        assert result["filename"] == "Cam1Bin1Flip0Filt0_2e6us_22C_001.tif"

    def test_falls_back_to_100us_dark(self, grabber):
        grabber.dark_table = self._make_dark_table([
            ("Cam1Bin1Flip0Filt0_100us_20C_001.tif", None, 1, 1, 0, 100, 0.0001, 20.0, 1),
        ])
        # Science exposure is 2s, no match → falls back to 100µs bias dark
        result = grabber._find_best_dark_row(self._meta(exposure_s=2.0))
        assert result["filename"] == "Cam1Bin1Flip0Filt0_100us_20C_001.tif"

    def test_returns_none_for_wrong_exposure_and_no_100us(self, grabber):
        grabber.dark_table = self._make_dark_table([
            ("Cam1Bin1Flip0Filt0_1e6us_20C_001.tif", None, 1, 1, 0, 1_000_000, 1.0, 20.0, 1),
        ])
        assert grabber._find_best_dark_row(self._meta(exposure_s=2.0)) is None
