"""
Tests for digital_eyepiece/zoom/multi_cam_zoom.py.
Run with: pytest digital_eyepiece/tests/test_multi_cam_zoom.py -v
"""

import pytest
from unittest.mock import MagicMock

from digital_eyepiece.view_state import ViewState
from digital_eyepiece.zoom.multi_cam_zoom import MultiCamZoom


def _grabbers(n: int):
    return [MagicMock() for _ in range(n)]


def _mcz(n: int = 3) -> MultiCamZoom:
    grabbers = _grabbers(n)
    fovs = [0.5 * (i + 1) for i in range(n)]   # 0.5°, 1.0°, 1.5°, ...
    return MultiCamZoom(grabbers, fovs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            MultiCamZoom(_grabbers(2), [1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MultiCamZoom([], [])

    def test_initial_index_is_zero(self):
        assert _mcz().active_index == 0

    def test_active_grabber_is_first(self):
        grabbers = _grabbers(3)
        mcz = MultiCamZoom(grabbers, [1.0, 2.0, 3.0])
        assert mcz.active_grabber is grabbers[0]

    def test_camera_count(self):
        assert _mcz(4).camera_count == 4


# ---------------------------------------------------------------------------
# step_out (zoom out → wider camera)
# ---------------------------------------------------------------------------

class TestStepOut:
    def test_step_out_increments_index(self):
        mcz = _mcz(3)
        state = ViewState()
        mcz.step_out(state)
        assert mcz.active_index == 1

    def test_step_out_updates_view_state(self):
        mcz = _mcz(3)
        state = ViewState()
        mcz.step_out(state)
        assert state.active_camera_index == 1

    def test_step_out_resets_zoom_level(self):
        mcz = _mcz(3)
        state = ViewState()
        state.zoom_level = 3.5
        mcz.step_out(state)
        assert state.zoom_level == pytest.approx(1.0)

    def test_step_out_returns_true_on_success(self):
        mcz = _mcz(3)
        assert mcz.step_out(ViewState()) is True

    def test_step_out_returns_false_at_widest(self):
        mcz = _mcz(3)
        state = ViewState()
        mcz.step_out(state)
        mcz.step_out(state)
        assert mcz.step_out(state) is False

    def test_step_out_does_not_exceed_bounds(self):
        mcz = _mcz(2)
        state = ViewState()
        mcz.step_out(state)
        mcz.step_out(state)
        assert mcz.active_index == 1   # clamped


# ---------------------------------------------------------------------------
# step_in (zoom in → narrower camera)
# ---------------------------------------------------------------------------

class TestStepIn:
    def test_step_in_decrements_index(self):
        mcz = _mcz(3)
        state = ViewState()
        mcz.step_out(state)
        mcz.step_in(state)
        assert mcz.active_index == 0

    def test_step_in_returns_false_at_narrowest(self):
        mcz = _mcz(3)
        assert mcz.step_in(ViewState()) is False

    def test_step_in_returns_true_on_success(self):
        mcz = _mcz(3)
        state = ViewState()
        mcz.step_out(state)
        assert mcz.step_in(state) is True

    def test_step_in_resets_zoom_level(self):
        mcz = _mcz(3)
        state = ViewState()
        mcz.step_out(state)
        state.zoom_level = 3.5
        mcz.step_in(state)
        assert state.zoom_level == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FOV property
# ---------------------------------------------------------------------------

class TestFov:
    def test_active_fov_matches_initial_camera(self):
        mcz = MultiCamZoom(_grabbers(3), [0.5, 1.0, 2.0])
        assert mcz.active_fov == pytest.approx(0.5)

    def test_active_fov_updates_after_step_out(self):
        mcz = MultiCamZoom(_grabbers(3), [0.5, 1.0, 2.0])
        mcz.step_out(ViewState())
        assert mcz.active_fov == pytest.approx(1.0)
