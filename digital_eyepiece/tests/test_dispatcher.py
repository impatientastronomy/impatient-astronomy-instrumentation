"""
Tests for digital_eyepiece/input/dispatcher.py.
Run with: pytest digital_eyepiece/tests/test_dispatcher.py -v
"""

import pytest
from unittest.mock import MagicMock

from digital_eyepiece.input.dispatcher import InputDispatcher, ScrollContext, OVERLAY_DURATION
from digital_eyepiece.input.menu import Menu, MenuItem
from digital_eyepiece.view_state import ViewState


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
        state.active_menu = "menu"
        assert dispatcher._context() == ScrollContext.MENU

    def test_overlay_context_when_active(self, dispatcher, state):
        state.overlay_active = True
        assert dispatcher._context() == ScrollContext.OVERLAY

    def test_menu_takes_priority_over_overlay(self, dispatcher, state):
        state.active_menu = "menu"
        state.overlay_active = True
        assert dispatcher._context() == ScrollContext.MENU

    def test_sky_map_context_when_all_sky_mode(self, dispatcher, state):
        state.all_sky_mode = True
        state.overlay_active = True
        assert dispatcher._context() == ScrollContext.SKY_MAP

    def test_sky_map_takes_priority_over_overlay(self, dispatcher, state):
        state.all_sky_mode = True
        state.overlay_active = True
        assert dispatcher._context() == ScrollContext.SKY_MAP


# ---------------------------------------------------------------------------
# Scroll — image zoom
# ---------------------------------------------------------------------------

class TestScrollImageZoom:
    def test_scroll_in_increases_zoom(self, dispatcher, state):
        dispatcher.on_scroll(-1)    # scroll down = zoom in
        assert state.zoom_level == pytest.approx(2.0)

    def test_scroll_out_decreases_zoom(self, dispatcher, state):
        state.zoom_level = 4.0
        dispatcher.on_scroll(1)     # scroll up = zoom out
        assert state.zoom_level == pytest.approx(2.0)

    def test_zoom_clamped_at_max(self, dispatcher, state):
        state.zoom_level = 8.0
        dispatcher.on_scroll(-1)    # scroll down = zoom in; already at max
        assert state.zoom_level == pytest.approx(8.0)

    def test_zoom_clamped_at_min_without_multicam(self, dispatcher, state):
        state.zoom_level = 1.0
        dispatcher.on_scroll(1)     # scroll up = zoom out; already at min
        assert state.zoom_level == pytest.approx(1.0)

    def test_multi_step_scroll(self, dispatcher, state):
        dispatcher.on_scroll(-2)    # scroll down 2 clicks = zoom_step^2 = 4×
        assert state.zoom_level == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Scroll — menu navigation
# ---------------------------------------------------------------------------

class TestScrollMenu:
    def test_scroll_navigates_menu(self, dispatcher, state, menu):
        state.active_menu = "menu"
        dispatcher.on_scroll(1)
        assert menu.selection_index == 1

    def test_scroll_does_not_change_zoom_when_menu_open(self, dispatcher, state):
        state.active_menu = "menu"
        dispatcher.on_scroll(1)
        assert state.zoom_level == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scroll — overlay context (zooms image, triggers all-sky at min)
# ---------------------------------------------------------------------------

class TestScrollOverlay:
    def test_scroll_in_increases_image_zoom(self, dispatcher, state):
        state.overlay_active = True
        dispatcher.on_scroll(-1)    # scroll down = zoom in
        assert state.zoom_level == pytest.approx(2.0)

    def test_scroll_out_decreases_image_zoom(self, dispatcher, state):
        state.overlay_active = True
        state.zoom_level = 4.0
        dispatcher.on_scroll(1)     # scroll up = zoom out
        assert state.zoom_level == pytest.approx(2.0)

    def test_scroll_out_at_min_clamps_and_recenters(self, dispatcher, state):
        state.overlay_active = True
        state.mount_connected = True
        state.zoom_level = 1.0
        state.zoom_center_x = 0.7
        state.zoom_center_y = 0.3
        dispatcher.on_scroll(1)     # scroll up = zoom out; already at min
        assert state.all_sky_mode is False
        assert state.zoom_level == pytest.approx(1.0)
        assert state.zoom_center_x == pytest.approx(0.5)
        assert state.zoom_center_y == pytest.approx(0.5)

    def test_scroll_in_does_not_exit_all_sky(self, dispatcher, state):
        # all_sky_mode is now exited only by right-click, not by scroll
        state.overlay_active = True
        state.all_sky_mode = True
        dispatcher.on_scroll(-1)
        assert state.all_sky_mode is True

    def test_overlay_scroll_does_not_affect_menu(self, dispatcher, state):
        state.overlay_active = True
        dispatcher.on_scroll(-1)
        assert state.active_menu is None


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
        state.active_menu = "menu"
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
        state.active_menu = "menu"
        dispatcher._overlay_timer = 0.1
        dispatcher.update(1.0)
        assert state.overlay_active is True

    def test_overlay_timer_does_not_expire_in_sky_map_mode(self, dispatcher, state):
        state.overlay_active = True
        state.all_sky_mode = True
        dispatcher._overlay_timer = 0.1
        dispatcher.update(1.0)
        assert state.overlay_active is True

    def test_overlay_does_not_hide_while_pinned(self, dispatcher, state):
        state.overlay_active = True
        state.overlay_pinned = True
        dispatcher._overlay_timer = 0.1
        dispatcher.update(1.0)
        assert state.overlay_active is True


# ---------------------------------------------------------------------------
# Overlay pin (show_overlay / mouse-move interaction)
# ---------------------------------------------------------------------------

class TestOverlayPin:
    def test_show_overlay_sets_active_and_arms_timer_regardless_of_pin(self, dispatcher, state):
        state.overlay_pinned = True
        dispatcher.show_overlay()
        assert state.overlay_active is True
        assert dispatcher._overlay_timer == pytest.approx(OVERLAY_DURATION)

    def test_mouse_move_does_not_affect_pinned_flag(self, dispatcher, state):
        state.mount_connected = True
        state.overlay_pinned = True
        dispatcher.on_mouse_move(10, 10)
        assert state.overlay_pinned is True

    def test_unpinning_then_expiring_hides_overlay(self, dispatcher, state):
        # Pin on, then simulate the button un-pinning it (as main.py's
        # _toggle_overlay_button does): flip the flag and re-show.
        state.overlay_pinned = True
        dispatcher.show_overlay()
        state.overlay_pinned = False
        dispatcher.show_overlay()
        dispatcher._overlay_timer = 0.1
        dispatcher.update(0.2)
        assert state.overlay_active is False


# ---------------------------------------------------------------------------
# Sky map scroll (FOV zoom)
# ---------------------------------------------------------------------------

class TestScrollSkyMap:
    @pytest.fixture
    def sky_dispatcher(self, state, menu):
        return InputDispatcher(state, menu,
                               zoom_step=2.0, zoom_min=1.0, zoom_max=8.0,
                               sky_map_fov_min=10.0, sky_map_fov_max=60.0)

    def test_scroll_down_decreases_fov(self, sky_dispatcher, state):
        state.all_sky_mode = True
        state.sky_map_fov = 20.0
        sky_dispatcher.on_scroll(-1)    # scroll down = zoom in = smaller FOV
        assert state.sky_map_fov == pytest.approx(10.0)

    def test_scroll_up_increases_fov(self, sky_dispatcher, state):
        state.all_sky_mode = True
        state.sky_map_fov = 20.0
        sky_dispatcher.on_scroll(1)     # scroll up = zoom out = larger FOV
        assert state.sky_map_fov == pytest.approx(40.0)

    def test_fov_clamped_at_min(self, sky_dispatcher, state):
        state.all_sky_mode = True
        state.sky_map_fov = 10.0
        sky_dispatcher.on_scroll(-1)
        assert state.sky_map_fov == pytest.approx(10.0)

    def test_fov_clamped_at_max(self, sky_dispatcher, state):
        state.all_sky_mode = True
        state.sky_map_fov = 60.0
        sky_dispatcher.on_scroll(1)
        assert state.sky_map_fov == pytest.approx(60.0)

    def test_sky_map_scroll_does_not_change_zoom_level(self, sky_dispatcher, state):
        state.all_sky_mode = True
        state.sky_map_fov = 20.0
        sky_dispatcher.on_scroll(-1)
        assert state.zoom_level == pytest.approx(1.0)

    def test_sky_map_scroll_does_not_affect_image_zoom(self, sky_dispatcher, state):
        state.all_sky_mode = False
        state.sky_map_fov = 20.0
        sky_dispatcher.on_scroll(-1)    # normal mode: zooms image, not sky map
        assert state.sky_map_fov == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Multi-cam integration
# ---------------------------------------------------------------------------

class TestMultiCamIntegration:
    def test_zoom_out_at_min_triggers_step_out(self, dispatcher, state):
        mock_cam = MagicMock()
        mock_cam.step_out.return_value = True
        dispatcher.register_multi_cam(mock_cam)
        state.zoom_level = 1.0
        dispatcher.on_scroll(1)     # scroll up = zoom out at min
        mock_cam.step_out.assert_called_once_with(state)

    def test_zoom_out_clamps_if_step_out_fails(self, dispatcher, state):
        mock_cam = MagicMock()
        mock_cam.step_out.return_value = False
        dispatcher.register_multi_cam(mock_cam)
        state.zoom_level = 1.0
        dispatcher.on_scroll(1)     # scroll up = zoom out; step_out fails → clamp
        assert state.zoom_level == pytest.approx(1.0)

    def test_zoom_in_does_not_trigger_step_out(self, dispatcher, state):
        mock_cam = MagicMock()
        dispatcher.register_multi_cam(mock_cam)
        dispatcher.on_scroll(-1)    # scroll down = zoom in
        mock_cam.step_out.assert_not_called()


# ---------------------------------------------------------------------------
# Zoom always about center (never the cursor)
# ---------------------------------------------------------------------------

class _FakeRect:
    """Minimal pygame.Rect substitute for tests."""
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height


class TestZoomAboutCenter:
    def _make_dispatcher_with_rect(self, state, menu, rect):
        d = InputDispatcher(state, menu, zoom_step=2.0, zoom_min=1.0, zoom_max=8.0)
        d.set_img_rect(rect)
        return d

    def test_on_scroll_takes_no_position_argument(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_scroll(-1)  # would raise TypeError if on_scroll still accepted pos
        assert state.zoom_level == pytest.approx(2.0)

    def test_zoom_in_does_not_move_center(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_center_x = 0.7
        state.zoom_center_y = 0.4
        d.on_scroll(-1)  # scroll down = zoom in
        assert state.zoom_center_x == pytest.approx(0.7)
        assert state.zoom_center_y == pytest.approx(0.4)

    def test_zoom_out_does_not_move_center(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_level = 4.0
        state.zoom_center_x = 0.3
        state.zoom_center_y = 0.6
        d.on_scroll(1)  # scroll up = zoom out, still above zoom_min
        assert state.zoom_center_x == pytest.approx(0.3)
        assert state.zoom_center_y == pytest.approx(0.6)

    def test_zoom_out_to_min_resets_center(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_level = 1.0
        state.zoom_center_x = 0.7
        state.zoom_center_y = 0.3
        d.on_scroll(1)  # scroll up = zoom out past min → clamp and recenter
        assert state.zoom_level == pytest.approx(1.0)
        assert state.zoom_center_x == pytest.approx(0.5)
        assert state.zoom_center_y == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Right-drag pan
# ---------------------------------------------------------------------------

class TestRightDragPan:
    def _make_dispatcher_with_rect(self, state, menu, rect):
        d = InputDispatcher(state, menu, zoom_step=2.0, zoom_min=1.0, zoom_max=8.0)
        d.set_img_rect(rect)
        return d

    def test_small_move_is_a_click(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_right_button_down(400, 300)
        d.on_mouse_move(402, 301, right_held=True)  # 3px — below threshold
        assert d.on_right_button_up() is True

    def test_large_move_is_a_drag(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_right_button_down(400, 300)
        d.on_mouse_move(420, 300, right_held=True)  # 20px — above threshold
        assert d.on_right_button_up() is False

    def test_pan_moves_zoom_center(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_level = 2.0
        state.zoom_center_x = 0.5
        state.zoom_center_y = 0.5
        d.on_right_button_down(400, 300)
        d.on_mouse_move(440, 300, right_held=True)  # drag 40px right
        # Image moves right → source window moves left → zoom_center_x decreases
        assert state.zoom_center_x < 0.5

    def test_pan_constrained_to_image_extents(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_level = 2.0
        d.on_right_button_down(400, 300)
        # Huge pan to the right
        d.on_mouse_move(400 + 10000, 300, right_held=True)
        assert state.zoom_center_x >= 0.5 / state.zoom_level

    def test_pan_no_effect_at_zoom_1(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_level = 1.0
        d.on_right_button_down(400, 300)
        d.on_mouse_move(450, 300, right_held=True)
        assert state.zoom_center_x == pytest.approx(0.5)
        assert state.zoom_center_y == pytest.approx(0.5)

    def test_pan_no_effect_when_menu_open(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_level = 2.0
        state.active_menu = "menu"
        d.on_right_button_down(400, 300)
        d.on_mouse_move(450, 300, right_held=True)
        assert state.zoom_center_x == pytest.approx(0.5)

    def test_drag_tracking_resets_after_button_up(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_right_button_down(400, 300)
        d.on_mouse_move(450, 300, right_held=True)
        d.on_right_button_up()
        assert d._right_drag_start is None
        assert d._right_drag_total == pytest.approx(0.0)

    def test_right_drag_does_not_pan_in_sky_map(self, state, menu):
        # SkyMap panning moved to left-drag; right-drag is inert there.
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = True
        state.zoom_center_x = 0.5
        state.zoom_center_y = 0.5
        d.on_right_button_down(400, 300)
        d.on_mouse_move(440, 300, right_held=True)
        assert state.zoom_center_x == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Left-drag pan (SkyMap only)
# ---------------------------------------------------------------------------

class TestLeftDragPan:
    def _make_dispatcher_with_rect(self, state, menu, rect):
        d = InputDispatcher(state, menu, zoom_step=2.0, zoom_min=1.0, zoom_max=8.0)
        d.set_img_rect(rect)
        return d

    def test_small_move_is_a_click(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = True
        d.on_left_button_down(400, 300)
        d.on_mouse_move(402, 301, left_held=True)  # 3px — below threshold
        assert d.on_left_button_up() is True

    def test_large_move_is_a_drag(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = True
        d.on_left_button_down(400, 300)
        d.on_mouse_move(420, 300, left_held=True)  # 20px — above threshold
        assert d.on_left_button_up() is False

    def test_pan_moves_zoom_center_in_sky_map(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = True
        state.zoom_center_x = 0.5
        state.zoom_center_y = 0.5
        d.on_left_button_down(400, 300)
        d.on_mouse_move(440, 300, left_held=True)  # drag 40px right
        assert state.zoom_center_x != pytest.approx(0.5)

    def test_pan_no_effect_when_menu_open(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = True
        state.active_menu = "menu"
        d.on_left_button_down(400, 300)
        d.on_mouse_move(450, 300, left_held=True)
        assert state.zoom_center_x == pytest.approx(0.5)

    def test_left_drag_does_not_pan_outside_sky_map(self, state, menu):
        # Normal zoomed view still uses right-drag; left-drag is inert there.
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = False
        state.zoom_level = 2.0
        state.zoom_center_x = 0.5
        state.zoom_center_y = 0.5
        d.on_left_button_down(400, 300)
        d.on_mouse_move(440, 300, left_held=True)
        assert state.zoom_center_x == pytest.approx(0.5)

    def test_drag_tracking_resets_after_button_up(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.all_sky_mode = True
        d.on_left_button_down(400, 300)
        d.on_mouse_move(450, 300, left_held=True)
        d.on_left_button_up()
        assert d._left_drag_start is None
        assert d._left_drag_total == pytest.approx(0.0)
