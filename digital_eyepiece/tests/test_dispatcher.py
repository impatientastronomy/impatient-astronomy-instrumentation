"""
Tests for digital_eyepiece/input/dispatcher.py.
Run with: pytest digital_eyepiece/tests/test_dispatcher.py -v
"""

import pytest
from unittest.mock import MagicMock

from digital_eyepiece.input.dispatcher import InputDispatcher, ScrollContext, OVERLAY_DURATION
from digital_eyepiece.input.menu import Menu, MenuItem
from digital_eyepiece.view_state import ViewMode, ViewState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    return ViewState()


@pytest.fixture
def menu():
    m = Menu()
    m.add(MenuItem("Cancel"))
    m.add(MenuItem("Save", action=lambda: None))
    return m


@pytest.fixture
def dispatcher(state, menu):
    return InputDispatcher(state, menu, zoom_step=2.0, zoom_min=1.0, zoom_max=8.0)


# ---------------------------------------------------------------------------
# Context detection
# ---------------------------------------------------------------------------

class TestContext:
    def test_image_context_by_default(self, dispatcher, state):
        assert dispatcher._context() == ScrollContext.IMAGE

    def test_menu_context_when_open(self, dispatcher, state):
        state.menu_open = True
        assert dispatcher._context() == ScrollContext.MENU

    def test_overlay_context_when_active(self, dispatcher, state):
        state.overlay_active = True
        assert dispatcher._context() == ScrollContext.OVERLAY

    def test_menu_takes_priority_over_overlay(self, dispatcher, state):
        state.menu_open = True
        state.overlay_active = True
        assert dispatcher._context() == ScrollContext.MENU


# ---------------------------------------------------------------------------
# Scroll — image zoom
# ---------------------------------------------------------------------------

class TestScrollImageZoom:
    def test_scroll_in_increases_zoom(self, dispatcher, state):
        dispatcher.on_scroll(1)
        assert state.zoom_level == pytest.approx(2.0)

    def test_scroll_out_decreases_zoom(self, dispatcher, state):
        state.zoom_level = 4.0
        dispatcher.on_scroll(-1)
        assert state.zoom_level == pytest.approx(2.0)

    def test_zoom_clamped_at_max(self, dispatcher, state):
        state.zoom_level = 8.0
        dispatcher.on_scroll(1)
        assert state.zoom_level == pytest.approx(8.0)

    def test_zoom_clamped_at_min_without_multicam(self, dispatcher, state):
        state.zoom_level = 1.0
        dispatcher.on_scroll(-1)
        assert state.zoom_level == pytest.approx(1.0)

    def test_multi_step_scroll(self, dispatcher, state):
        dispatcher.on_scroll(2)     # zoom_step=2.0, delta=2 → 2.0^2 = 4×
        assert state.zoom_level == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Scroll — menu navigation
# ---------------------------------------------------------------------------

class TestScrollMenu:
    def test_scroll_navigates_menu(self, dispatcher, state, menu):
        state.menu_open = True
        dispatcher.on_scroll(1)
        assert menu.selection_index == 1

    def test_scroll_does_not_change_zoom_when_menu_open(self, dispatcher, state):
        state.menu_open = True
        dispatcher.on_scroll(1)
        assert state.zoom_level == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scroll — overlay context (zooms image, triggers all-sky at min)
# ---------------------------------------------------------------------------

class TestScrollOverlay:
    def test_scroll_in_increases_image_zoom(self, dispatcher, state):
        state.overlay_active = True
        dispatcher.on_scroll(1)
        assert state.zoom_level == pytest.approx(2.0)

    def test_scroll_out_decreases_image_zoom(self, dispatcher, state):
        state.overlay_active = True
        state.zoom_level = 4.0
        dispatcher.on_scroll(-1)
        assert state.zoom_level == pytest.approx(2.0)

    def test_scroll_out_at_min_triggers_all_sky_when_mount_connected(self, dispatcher, state):
        state.overlay_active = True
        state.mount_connected = True
        state.zoom_level = 1.0
        dispatcher.on_scroll(-1)
        assert state.all_sky_mode is True

    def test_scroll_out_at_min_clamps_without_mount(self, dispatcher, state):
        state.overlay_active = True
        state.mount_connected = False
        state.zoom_level = 1.0
        dispatcher.on_scroll(-1)
        assert state.all_sky_mode is False
        assert state.zoom_level == pytest.approx(1.0)

    def test_scroll_in_exits_all_sky(self, dispatcher, state):
        state.overlay_active = True
        state.all_sky_mode = True
        dispatcher.on_scroll(1)
        assert state.all_sky_mode is False
        assert state.overlay_active is False

    def test_overlay_scroll_does_not_affect_menu(self, dispatcher, state):
        state.overlay_active = True
        dispatcher.on_scroll(1)
        assert state.menu_open is False


# ---------------------------------------------------------------------------
# Left-click
# ---------------------------------------------------------------------------

class TestLeftClick:
    def test_left_click_toggles_to_accumulate(self, dispatcher, state):
        dispatcher.on_left_click()
        assert state.mode == ViewMode.ACCUMULATE

    def test_left_click_toggles_back_to_live(self, dispatcher, state):
        state.mode = ViewMode.ACCUMULATE
        dispatcher.on_left_click()
        assert state.mode == ViewMode.LIVE

    def test_left_click_does_not_open_menu(self, dispatcher, state):
        dispatcher.on_left_click()
        assert state.menu_open is False

    def test_left_click_selects_and_closes_on_leaf(self, dispatcher, state):
        state.menu_open = True
        dispatcher.on_left_click()   # selects "Cancel" (no-op, closes)
        assert state.menu_open is False

    def test_left_click_enters_submenu_stays_open(self, dispatcher, state, menu):
        sub = [MenuItem("1×", action=lambda: None)]
        menu.add(MenuItem("Brightness", submenu=sub))
        menu.scroll(2)
        state.menu_open = True
        dispatcher.on_left_click()
        assert state.menu_open is True


# ---------------------------------------------------------------------------
# Right-click
# ---------------------------------------------------------------------------

class TestRightClick:
    def test_right_click_opens_menu_when_closed(self, dispatcher, state):
        dispatcher.on_right_click()
        assert state.menu_open is True

    def test_right_click_resets_menu_to_root_on_open(self, dispatcher, state, menu):
        menu.scroll(1)              # move to "Save"
        dispatcher.on_right_click()
        assert menu.selection_index == 0

    def test_right_click_closes_menu_when_open(self, dispatcher, state):
        state.menu_open = True
        dispatcher.on_right_click()
        assert state.menu_open is False

    def test_right_click_exits_all_sky_mode(self, dispatcher, state):
        state.all_sky_mode = True
        state.overlay_active = True
        dispatcher.on_right_click()
        assert state.all_sky_mode is False
        assert state.overlay_active is False

    def test_right_click_all_sky_does_not_open_menu(self, dispatcher, state):
        state.all_sky_mode = True
        dispatcher.on_right_click()
        assert state.menu_open is False


# ---------------------------------------------------------------------------
# Middle-click (mount slew)
# ---------------------------------------------------------------------------

class TestMiddleClick:
    def test_middle_click_returns_false_when_mount_not_connected(self, dispatcher, state):
        state.overlay_active = True
        assert dispatcher.on_middle_click() is False

    def test_middle_click_returns_false_when_overlay_not_active(self, dispatcher, state):
        state.mount_connected = True
        assert dispatcher.on_middle_click() is False

    def test_middle_click_returns_true_when_conditions_met(self, dispatcher, state):
        state.mount_connected = True
        state.overlay_active = True
        assert dispatcher.on_middle_click() is True

    def test_middle_click_calls_slew_action(self, dispatcher, state):
        state.mount_connected = True
        state.overlay_active = True
        action = MagicMock()
        dispatcher.register_slew_action(action)
        dispatcher.on_middle_click()
        action.assert_called_once()


# ---------------------------------------------------------------------------
# Mouse move → overlay trigger
# ---------------------------------------------------------------------------

class TestMouseMove:
    def test_mouse_move_shows_overlay_when_mount_connected(self, dispatcher, state):
        state.mount_connected = True
        dispatcher.on_mouse_move(100, 200)
        assert state.overlay_active is True

    def test_mouse_move_resets_overlay_timer(self, dispatcher, state):
        state.mount_connected = True
        dispatcher.on_mouse_move(100, 200)
        assert dispatcher._overlay_timer == pytest.approx(OVERLAY_DURATION)

    def test_mouse_move_no_effect_without_mount(self, dispatcher, state):
        dispatcher.on_mouse_move(100, 200)
        assert state.overlay_active is False

    def test_mouse_move_no_effect_when_menu_open(self, dispatcher, state):
        state.mount_connected = True
        state.menu_open = True
        dispatcher.on_mouse_move(100, 200)
        assert state.overlay_active is False


# ---------------------------------------------------------------------------
# Overlay auto-hide via update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_overlay_hides_after_timer_expires(self, dispatcher, state):
        state.mount_connected = True
        state.overlay_active = True
        dispatcher._overlay_timer = 0.1
        dispatcher.update(0.2)
        assert state.overlay_active is False

    def test_overlay_stays_visible_before_timer_expires(self, dispatcher, state):
        state.overlay_active = True
        dispatcher._overlay_timer = 5.0
        dispatcher.update(0.1)
        assert state.overlay_active is True

    def test_update_no_effect_when_menu_open(self, dispatcher, state):
        state.overlay_active = True
        state.menu_open = True
        dispatcher._overlay_timer = 0.1
        dispatcher.update(1.0)
        assert state.overlay_active is True


# ---------------------------------------------------------------------------
# Back button
# ---------------------------------------------------------------------------

class TestBack:
    def test_back_closes_menu_at_root(self, dispatcher, state):
        state.menu_open = True
        dispatcher.on_back()
        assert state.menu_open is False

    def test_back_exits_submenu_keeps_menu_open(self, dispatcher, state, menu):
        sub = [MenuItem("A")]
        menu.add(MenuItem("Parent", submenu=sub))
        menu.scroll(2)
        state.menu_open = True
        dispatcher.on_left_click()   # enter submenu
        assert state.menu_open is True
        dispatcher.on_back()         # exit submenu
        assert state.menu_open is True


# ---------------------------------------------------------------------------
# Multi-cam integration
# ---------------------------------------------------------------------------

class TestMultiCamIntegration:
    def test_zoom_out_at_min_triggers_step_out(self, dispatcher, state):
        mock_cam = MagicMock()
        mock_cam.step_out.return_value = True
        dispatcher.register_multi_cam(mock_cam)
        state.zoom_level = 1.0
        dispatcher.on_scroll(-1)
        mock_cam.step_out.assert_called_once_with(state)

    def test_zoom_out_clamps_if_step_out_fails(self, dispatcher, state):
        mock_cam = MagicMock()
        mock_cam.step_out.return_value = False
        dispatcher.register_multi_cam(mock_cam)
        state.zoom_level = 1.0
        dispatcher.on_scroll(-1)
        assert state.zoom_level == pytest.approx(1.0)

    def test_zoom_in_does_not_trigger_step_out(self, dispatcher, state):
        mock_cam = MagicMock()
        dispatcher.register_multi_cam(mock_cam)
        dispatcher.on_scroll(1)
        mock_cam.step_out.assert_not_called()
