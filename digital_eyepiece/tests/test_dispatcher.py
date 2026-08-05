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

    def test_right_click_in_all_sky_opens_menu(self, dispatcher, state):
        # Right-click in sky map mode no longer exits immediately —
        # the context menu (handled by main.py) gives the user the choice.
        state.all_sky_mode = True
        state.overlay_active = True
        state.menu_open = False
        dispatcher.on_right_click()
        # Dispatcher leaves all_sky_mode untouched; caller opens context menu
        assert state.all_sky_mode is True
        assert state.menu_open is True



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

    def test_overlay_timer_does_not_expire_in_sky_map_mode(self, dispatcher, state):
        state.overlay_active = True
        state.all_sky_mode = True
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
# Cursor-centred zoom
# ---------------------------------------------------------------------------

class _FakeRect:
    """Minimal pygame.Rect substitute for tests."""
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height


class TestCursorCentredZoom:
    def _make_dispatcher_with_rect(self, state, menu, rect):
        d = InputDispatcher(state, menu, zoom_step=2.0, zoom_min=1.0, zoom_max=8.0)
        d.set_img_rect(rect)
        return d

    def test_zoom_at_centre_does_not_shift_center(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_scroll(-1, pos=(400, 300))  # scroll down = zoom in; cursor at centre
        assert state.zoom_center_x == pytest.approx(0.5, abs=1e-6)
        assert state.zoom_center_y == pytest.approx(0.5, abs=1e-6)

    def test_zoom_in_at_right_shifts_center_right(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_scroll(-1, pos=(800, 300))  # scroll down = zoom in; cursor at right edge
        assert state.zoom_center_x > 0.5

    def test_zoom_in_at_top_left_shifts_center_up_left(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_scroll(-1, pos=(0, 0))  # scroll down = zoom in; cursor at top-left
        assert state.zoom_center_x < 0.5
        assert state.zoom_center_y < 0.5

    def test_zoom_center_clamped_to_bounds(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        d.on_scroll(-1, pos=(800, 300))  # scroll down = zoom in; push toward right edge
        assert state.zoom_center_x <= 1.0 - 0.5 / state.zoom_level
        assert state.zoom_center_x >= 0.5 / state.zoom_level

    def test_scroll_without_pos_does_not_move_center(self, state, menu):
        rect = _FakeRect(0, 0, 800, 600)
        d = self._make_dispatcher_with_rect(state, menu, rect)
        state.zoom_center_x = 0.7
        state.zoom_center_y = 0.4
        d.on_scroll(-1)  # scroll down = zoom in; no pos
        assert state.zoom_center_x == pytest.approx(0.7)
        assert state.zoom_center_y == pytest.approx(0.4)

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
        state.menu_open = True
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
