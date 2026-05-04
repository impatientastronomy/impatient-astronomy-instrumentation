"""
Tests for astrocore/camera/zwo_asi.py.

Unit tests use mocks — no SDK or hardware required.
Integration tests (marked 'hardware') require a connected ASI camera and the SDK.

Run unit tests only:
    pytest astrocore/tests/test_zwo_asi.py -v -m "not hardware"

Run hardware tests (camera plugged in, SDK library available):
    pytest astrocore/tests/test_zwo_asi.py -v -m hardware --lib /path/to/libASICamera2.dylib
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock SDK factory
# ---------------------------------------------------------------------------

def _make_mock_sdk(
    num_cameras: int = 1,
    model: str = "ZWO ASI294MC Pro",
    width: int = 4144,
    height: int = 2822,
    pixel_size: float = 4.63,
    bit_depth: int = 16,
    has_cooler: bool = True,
    is_color: bool = True,
    bayer_pattern: int = 0,   # 0 = RGGB
    camera_id: int = 1,        # value stored in EEPROM last byte
):
    """Return a (sdk_mock, camera_mock) pair with sensible defaults."""
    cam_props = {
        "Name": model,
        "MaxWidth": width,
        "MaxHeight": height,
        "PixelSize": pixel_size,
        "BitDepth": bit_depth,
        "IsCoolerCam": has_cooler,
        "IsColorCam": is_color,
        "BayerPattern": bayer_pattern,
    }

    sdk = MagicMock()
    sdk.get_num_cameras.return_value = num_cameras
    sdk.get_camera_property.return_value = cam_props

    sdk.ASI_IMG_RAW16          = 2
    sdk.ASI_EXPOSURE           = 1
    sdk.ASI_GAIN               = 0
    sdk.ASI_BRIGHTNESS         = 11
    sdk.ASI_TEMPERATURE        = 8
    sdk.ASI_TARGET_TEMP        = 22
    sdk.ASI_COOLER_ON          = 13
    sdk.ASI_FAN_ON             = 18
    sdk.ASI_FLIP               = 9
    sdk.ASI_BANDWIDTHOVERLOAD  = 26
    sdk.ASI_COOLER_POWER_PERC  = 21
    sdk.ASI_EXP_IDLE           = 0
    sdk.ASI_EXP_WORKING        = 1
    sdk.ASI_EXP_SUCCESS        = 2
    sdk.ASI_EXP_FAILED         = 3
    sdk.ZWO_Error = Exception

    # EEPROM ID: last byte holds the user-assigned camera number
    eeprom = bytearray(8)
    eeprom[7] = camera_id
    mock_cam = MagicMock()
    mock_cam.get_camera_property.return_value = cam_props
    mock_cam.get_roi.return_value = (0, 0, width, height)           # (x, y, w, h)
    mock_cam.get_roi_format.return_value = (width, height, 1, sdk.ASI_IMG_RAW16)  # (w, h, bins, type)
    mock_cam.get_roi_start_position.return_value = (0, 0)
    mock_cam.get_exposure_status.return_value = sdk.ASI_EXP_SUCCESS
    mock_cam.get_data_after_exposure.return_value = np.zeros((height, width), dtype=np.uint16)
    mock_cam.get_id.return_value = bytes(eeprom)

    def _get_control_value(control_type):
        return {
            sdk.ASI_TEMPERATURE:       (200,       False),  # 20.0°C
            sdk.ASI_FLIP:              (0,         False),  # FlipMode.NONE
            sdk.ASI_BANDWIDTHOVERLOAD: (70,        False),
            sdk.ASI_EXPOSURE:          (2_000_000, False),  # 2.0 s
            sdk.ASI_GAIN:              (100,       False),
            sdk.ASI_BRIGHTNESS:        (10,        False),
            sdk.ASI_TARGET_TEMP:       (-15,       False),
            sdk.ASI_COOLER_ON:         (1,         False),
            sdk.ASI_FAN_ON:            (1,         False),
        }.get(control_type, (0, False))

    mock_cam.get_control_value.side_effect = _get_control_value

    sdk.Camera.return_value = mock_cam

    return sdk, mock_cam


@pytest.fixture
def mock_sdk_and_cam():
    """Fixture that patches zwoasi and reloads the driver module."""
    sdk, mock_cam = _make_mock_sdk()
    with patch.dict("sys.modules", {"zwoasi": sdk}):
        from importlib import reload
        import astrocore.camera.zwo_asi as module
        reload(module)
        # Bypass filesystem and ctypes calls that can't work without real hardware.
        # _read_camera_id returns 1 by default, matching _make_mock_sdk(camera_id=1).
        with patch.object(module, "_init_sdk"), \
             patch.object(module, "_read_camera_id", return_value=1):
            yield module, sdk, mock_cam


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TestConnect:
    def test_connect_opens_camera(self, mock_sdk_and_cam):
        module, sdk, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        cam.connect()
        assert cam._connected
        sdk.Camera.assert_called_once_with(0)

    def test_connect_twice_is_idempotent(self, mock_sdk_and_cam):
        module, sdk, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        cam.connect()
        cam.connect()
        assert sdk.Camera.call_count == 1

    def test_no_cameras_raises_connection_error(self, mock_sdk_and_cam):
        module, sdk, _ = mock_sdk_and_cam
        sdk.get_num_cameras.return_value = 0
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError, match="No ZWO ASI cameras"):
            cam.connect()

    def test_index_out_of_range_raises(self, mock_sdk_and_cam):
        module, sdk, _ = mock_sdk_and_cam
        sdk.get_num_cameras.return_value = 2
        cam = module.ZwoAsiCamera(index=5, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError, match="index 5"):
            cam.connect()

    def test_missing_lib_raises(self):
        # Test _init_sdk in isolation — no mock fixture needed.
        import astrocore.camera.zwo_asi as zwo_module
        from astrocore.camera.base import CameraError
        import os
        env = {k: v for k, v in os.environ.items() if k not in ("ASI_LIB", "ASI_SDK_PATH")}
        with patch.dict("os.environ", env, clear=True), \
             patch.object(zwo_module, "_find_asi_library", return_value=None):
            with pytest.raises(CameraError, match="ASI_LIB"):
                zwo_module._init_sdk(None)

    def test_context_manager_disconnects(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with cam:
            pass
        assert not cam._connected

    def test_context_manager_disconnects_on_exception(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(RuntimeError):
            with cam:
                raise RuntimeError("mid-session error")
        assert not cam._connected


# ---------------------------------------------------------------------------
# CameraInfo and camera_id
# ---------------------------------------------------------------------------

class TestCameraInfo:
    def test_info_fields(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            info = cam.info
        assert info.model == "ZWO ASI294MC Pro"
        assert info.sensor_width_px == 4144
        assert info.sensor_height_px == 2822
        assert info.pixel_size_um == 4.63
        assert info.bit_depth == 16
        assert info.has_cooler is True
        assert info.usb_index == 0

    def test_camera_id_read_from_eeprom(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.info.camera_id == 1  # set in _make_mock_sdk default

    def test_bayer_pattern_set_for_color_camera(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.info.bayer_pattern == "RGGB"

    def test_info_requires_connection(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError):
            _ = cam.info


class TestSetCameraId:
    def test_set_camera_id_writes_eeprom(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.set_camera_id(3)
        written = mock_cam.set_id.call_args[0][0]
        assert written[7] == 3

    def test_set_camera_id_out_of_range_raises(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError):
                cam.set_camera_id(9)
            with pytest.raises(module.ConfigurationError):
                cam.set_camera_id(0)

    def test_set_camera_id_requires_connection(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError):
            cam.set_camera_id(1)


# ---------------------------------------------------------------------------
# Non-blocking imaging primitives
# ---------------------------------------------------------------------------

class TestNonBlockingPrimitives:
    def test_start_exposure_calls_sdk(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.start_exposure()
        mock_cam.start_exposure.assert_called_once_with(False)

    def test_start_exposure_dark_frame(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.start_exposure(is_dark=True)
        mock_cam.start_exposure.assert_called_once_with(True)

    def test_exposure_status_maps_sdk_value(self, mock_sdk_and_cam):
        module, sdk, mock_cam = mock_sdk_and_cam
        from astrocore.camera.base import ExposureStatus
        mock_cam.get_exposure_status.return_value = sdk.ASI_EXP_WORKING
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.exposure_status == ExposureStatus.WORKING

    def test_read_frame_returns_correct_shape(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.start_exposure()
            frame = cam.read_frame()
        assert frame.data.shape == (2822, 4144)

    def test_read_frame_meta_populated(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.start_exposure()
            frame = cam.read_frame()
        assert frame.meta.camera_id == 1
        assert frame.meta.exposure_seconds == 2.0   # mock returns 2_000_000 µs
        assert frame.meta.gain == 100               # mock returns 100
        assert frame.meta.camera_model == "ZWO ASI294MC Pro"

    def test_connect_sets_default_exposure_time(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib"):
            pass
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_EXPOSURE, 100_000)  # 0.1s

    def test_read_frame_without_prior_exposure_raises(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ExposureError, match="before any exposure"):
                cam.read_frame()


# ---------------------------------------------------------------------------
# Blocking convenience expose()
# ---------------------------------------------------------------------------

class TestExpose:
    def test_expose_returns_frame(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            frame = cam.expose()
        assert frame.data.shape == (2822, 4144)
        assert frame.meta.gain == 100

    def test_expose_raises_on_failure(self, mock_sdk_and_cam):
        module, sdk, mock_cam = mock_sdk_and_cam
        mock_cam.get_exposure_status.return_value = sdk.ASI_EXP_FAILED
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ExposureError):
                cam.expose()

    def test_expose_requires_connection(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError):
            cam.expose()


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_get_temperature_converts_tenths(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        # Default side_effect already returns (200, False) for ASI_TEMPERATURE.
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.get_temperature() == 20.0

    def test_set_temperature_uncooled_raises(self, mock_sdk_and_cam):
        module, sdk, mock_cam = mock_sdk_and_cam
        mock_cam.get_camera_property.return_value = {
            **mock_cam.get_camera_property.return_value,
            "IsCoolerCam": False,
        }
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError, match="no cooler"):
                cam.set_target_temperature(-10.0)

    def test_set_temperature_cooled_sends_correct_values(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.set_target_temperature(-15.0)
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_TARGET_TEMP, -15)
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_COOLER_ON, 1)
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_FAN_ON, 1)


# ---------------------------------------------------------------------------
# Camera state properties
# ---------------------------------------------------------------------------

class TestCameraStateProperties:
    def test_index_and_library_path_without_connecting(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=2, library_path="/fake/lib.dylib")
        assert cam.index == 2
        assert cam.library_path == "/fake/lib.dylib"

    def test_bayer_returns_pattern(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.bayer == "RGGB"

    def test_exposure_time_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.exposure_time == 2.0

    def test_exposure_time_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.exposure_time = 5.0
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_EXPOSURE, 5_000_000)

    def test_gain_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.gain == 100

    def test_gain_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.gain = 200
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_GAIN, 200)

    def test_offset_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.offset == 10

    def test_offset_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.offset = 20
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_BRIGHTNESS, 20)

    def test_bin_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.bin == 1

    def test_bin_set_resets_roi_to_full_sensor(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.bin = 2
        mock_cam.set_roi.assert_called_with(width=2072, height=1410, bins=2)

    def test_image_width_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.image_width == 4144

    def test_image_width_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.image_width = 800
        mock_cam.set_roi.assert_called_with(width=800, height=2822, bins=1)

    def test_image_height_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.image_height == 2822

    def test_image_height_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.image_height = 600
        mock_cam.set_roi.assert_called_with(width=4144, height=600, bins=1)

    def test_target_temp_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.target_temp == -15.0

    def test_target_temp_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.target_temp = -20.0
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_TARGET_TEMP, -20)

    def test_target_temp_uncooled_raises(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        mock_cam.get_camera_property.return_value = {
            **mock_cam.get_camera_property.return_value,
            "IsCoolerCam": False,
        }
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError, match="no cooler"):
                cam.target_temp = -10.0

    def test_cooler_on_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.cooler_on is True

    def test_cooler_on_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.cooler_on = False
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_COOLER_ON, 0)

    def test_fan_on_get(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.fan_on is True

    def test_fan_on_set(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.fan_on = False
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_FAN_ON, 0)


# ---------------------------------------------------------------------------
# Bandwidth overload
# ---------------------------------------------------------------------------

class TestBandwidthOverload:
    def test_connect_sets_safe_default(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib"):
            pass
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_BANDWIDTHOVERLOAD, 70)

    def test_get_bandwidth_overload(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        mock_cam.get_control_value.side_effect = lambda ct: (70, False) if ct == module.asi.ASI_BANDWIDTHOVERLOAD else (0, False)
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            assert cam.bandwidth_overload == 70

    def test_set_bandwidth_overload(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.bandwidth_overload = 60
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_BANDWIDTHOVERLOAD, 60)

    def test_bandwidth_overload_requires_connection(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError):
            _ = cam.bandwidth_overload
        with pytest.raises(module.ConnectionError):
            cam.bandwidth_overload = 40


# ---------------------------------------------------------------------------
# Persistent meta
# ---------------------------------------------------------------------------

class TestPersistentMeta:
    def _grab_frame(self, cam, module):
        cam.start_exposure()
        return cam.read_frame()

    def test_default_values_are_zero(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            frame = self._grab_frame(cam, module)
        assert frame.meta.focal_length_mm == 0.0
        assert frame.meta.telescope_description == ""
        assert frame.meta.RA  == 0.0
        assert frame.meta.Dec == 0.0
        assert frame.meta.Lat == 0.0
        assert frame.meta.Lon == 0.0

    def test_set_focal_length_appears_in_frame(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.meta.focal_length_mm = 600.0
            cam.meta.telescope_description = "Vixen_600mm"
            frame = self._grab_frame(cam, module)
        assert frame.meta.focal_length_mm == 600.0
        assert frame.meta.telescope_description == "Vixen_600mm"

    def test_meta_persists_across_multiple_frames(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.meta.RA  = 83.82
            cam.meta.Dec = -5.39
            f1 = self._grab_frame(cam, module)
            f2 = self._grab_frame(cam, module)
        assert f1.meta.RA  == f2.meta.RA  == 83.82
        assert f1.meta.Dec == f2.meta.Dec == -5.39

    def test_meta_update_reflected_in_next_frame(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.meta.focal_length_mm = 600.0
            f1 = self._grab_frame(cam, module)
            cam.meta.focal_length_mm = 2000.0
            f2 = self._grab_frame(cam, module)
        assert f1.meta.focal_length_mm == 600.0
        assert f2.meta.focal_length_mm == 2000.0

    def test_filter_id_default_is_zero(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            frame = self._grab_frame(cam, module)
        assert frame.meta.filter_id == 0

    def test_set_filter_id_appears_in_frame(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.meta.filter_id = 3   # Ha
            frame = self._grab_frame(cam, module)
        assert frame.meta.filter_id == 3

    def test_disconnect_clears_meta(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        cam.connect()
        cam.meta.focal_length_mm = 600.0
        cam.meta.Lat = 37.77
        cam.disconnect()
        assert cam.meta.focal_length_mm == 0.0
        assert cam.meta.Lat == 0.0


# ---------------------------------------------------------------------------
# list_cameras
# ---------------------------------------------------------------------------

class TestListCameras:
    def test_returns_list_of_dicts(self, mock_sdk_and_cam):
        module, sdk, _ = mock_sdk_and_cam
        sdk.get_num_cameras.return_value = 1
        cameras = module.list_cameras(library_path="/fake/lib.dylib")
        assert len(cameras) == 1
        assert "camera_id" in cameras[0]
        assert "model" in cameras[0]
        assert "usb_index" in cameras[0]

    def test_sorted_by_camera_id(self, mock_sdk_and_cam):
        module, sdk, _ = mock_sdk_and_cam
        sdk.get_num_cameras.return_value = 2

        # usb_index 0 has camera_id=3, usb_index 1 has camera_id=1.
        # After sorting, camera_id=1 should appear first.
        with patch.object(module, "_read_camera_id", side_effect=[3, 1]):
            cameras = module.list_cameras(library_path="/fake/lib.dylib")

        assert cameras[0]["camera_id"] == 1
        assert cameras[1]["camera_id"] == 3


# ---------------------------------------------------------------------------
# ROI and flip controls
# ---------------------------------------------------------------------------

class TestRoiAndFlip:
    def test_get_roi_returns_x_y_width_height(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        mock_cam.get_roi.return_value = (100, 50, 800, 600)
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            x, y, w, h = cam.get_roi()
        assert (x, y, w, h) == (100, 50, 800, 600)

    def test_set_roi_calls_sdk(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.set_roi(x=10, y=20, width=800, height=600)
        mock_cam.set_roi.assert_called_with(width=800, height=600, bins=1)
        mock_cam.set_roi_start_position.assert_called_with(10, 20)

    def test_set_roi_defaults_to_full_sensor(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.set_roi()
        mock_cam.set_roi.assert_called_with(width=4144, height=2822, bins=1)
        mock_cam.set_roi_start_position.assert_called_with(0, 0)

    def test_set_roi_with_explicit_bin(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        # 2822 / 2 = 1411 (odd) — floored to 1410 to satisfy height % 2 == 0.
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.set_roi(bin=2)
        mock_cam.set_roi.assert_called_with(width=2072, height=1410, bins=2)

    def test_get_flip_returns_flipmode(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            mode = cam.flip
        assert mode == module.FlipMode.NONE

    def test_set_flip_sends_correct_value(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.flip = module.FlipMode.HORIZ
        mock_cam.set_control_value.assert_any_call(module.asi.ASI_FLIP, 1)

    def test_read_frame_meta_includes_roi_and_flip(self, mock_sdk_and_cam):
        module, _, mock_cam = mock_sdk_and_cam
        mock_cam.get_roi.return_value = (100, 50, 800, 600)
        mock_cam.get_roi_format.return_value = (800, 600, 1, 2)
        mock_cam.get_data_after_exposure.return_value = np.zeros((600, 800), dtype=np.uint16)
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            cam.start_exposure()
            frame = cam.read_frame()
        assert frame.meta.roi_x == 100
        assert frame.meta.roi_y == 50
        assert frame.meta.roi_width == 800
        assert frame.meta.roi_height == 600
        assert frame.meta.flip == "NONE"

    def test_set_roi_rejects_width_not_multiple_of_8(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError, match="multiple of 8"):
                cam.set_roi(width=801, height=600)

    def test_set_roi_rejects_height_not_multiple_of_2(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError, match="multiple of 2"):
                cam.set_roi(width=800, height=601)

    def test_set_roi_rejects_odd_start_x(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError, match="start x"):
                cam.set_roi(x=1, width=800, height=600)

    def test_set_roi_rejects_odd_start_y(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        with module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib") as cam:
            with pytest.raises(module.ConfigurationError, match="start y"):
                cam.set_roi(y=1, width=800, height=600)

    def test_roi_requires_connection(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError):
            cam.get_roi()
        with pytest.raises(module.ConnectionError):
            cam.set_roi(width=800, height=600)

    def test_flip_requires_connection(self, mock_sdk_and_cam):
        module, _, _ = mock_sdk_and_cam
        cam = module.ZwoAsiCamera(index=0, library_path="/fake/lib.dylib")
        with pytest.raises(module.ConnectionError):
            _ = cam.flip
        with pytest.raises(module.ConnectionError):
            cam.flip = module.FlipMode.VERT


# ---------------------------------------------------------------------------
# Hardware integration tests — require a real camera
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestHardware:
    """
    Run with: pytest -m hardware --lib /path/to/libASICamera2.dylib
    """

    @pytest.fixture(autouse=True)
    def _require_lib(self, lib_path):
        if lib_path is None:
            pytest.skip("Pass --lib <path> to run hardware tests")

    def test_list_cameras(self, lib_path):
        from astrocore.camera.zwo_asi import list_cameras
        cameras = list_cameras(library_path=lib_path)
        assert len(cameras) >= 1, "No cameras found — is one plugged in?"
        print(f"\nDetected: {cameras}")

    def test_connect_and_info(self, lib_path):
        from astrocore.camera.zwo_asi import ZwoAsiCamera
        with ZwoAsiCamera(index=0, library_path=lib_path) as cam:
            info = cam.info
            print(f"\n{info}")
            assert info.sensor_width_px > 0
            assert info.pixel_size_um > 0

    def test_short_exposure(self, lib_path):
        from astrocore.camera.zwo_asi import ZwoAsiCamera
        with ZwoAsiCamera(index=0, library_path=lib_path) as cam:
            cam.exposure_time = 0.1
            cam.gain = 100
            frame = cam.expose()
        assert frame.data is not None
        assert frame.meta.camera_id >= 0
        print(f"\nFrame shape: {frame.data.shape}, camera_id: {frame.meta.camera_id}")
