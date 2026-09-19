"""
InputDispatcher — routes mouse events to the right handler based on
the current viewing context.

Click handling (left-click / right-click-to-open-menu / edge buttons) lives
directly in main.py's event loop. This dispatcher covers everything else:

Mouse mapping
-------------
Right-drag  : pan the zoomed image (normal live view)
Left-drag   : pan the sky map (all_sky_mode only)
Scroll up   : zoom in  (about the current view center) / menu up
Scroll down : zoom out / menu down

Overlay auto-hide
-----------------
When the mount is connected, moving the mouse shows the overlay for
OVERLAY_DURATION seconds. Call update(dt) every frame to drive the timer.
This auto-hide is suspended while view_state.overlay_pinned is True (the
user forced the overlay always-on via the edge button) -- see show_overlay().
"""

from __future__ import annotations

from enum import Enum, auto

from ..view_state import ViewState
from .menu import Menu

OVERLAY_DURATION = 5.0   # seconds the overlay stays visible after mouse move
_DRAG_THRESHOLD = 5  # pixels of travel before a button-hold becomes a pan


class ScrollContext(Enum):
    MENU    = auto()   # menu is open — scroll navigates items
    IMAGE   = auto()   # live/accumulate view — scroll zooms image
    OVERLAY = auto()   # overlay visible — scroll zooms image
    SKY_MAP = auto()   # all-sky mode — scroll changes sky map FOV


class InputDispatcher:
    """
    Routes scroll and click events to the correct handler.

    Parameters
    ----------
    view_state :
        Shared display state that this dispatcher reads and writes.
    menu :
        The Menu instance to navigate when the menu is open.
    zoom_step :
        Multiplicative factor per scroll click for image zoom.
    zoom_min / zoom_max :
        Clamp limits for image zoom_level.
    """

    def __init__(
        self,
        view_state: ViewState,
        menu: Menu,
        zoom_step: float = 1.1,
        zoom_min: float = 1.0,
        zoom_max: float = 8.0,
        sky_map_fov_min: float = 10.0,
        sky_map_fov_max: float = 60.0,
    ) -> None:
        self._state = view_state
        self._menu = menu
        self._zoom_step = zoom_step
        self._zoom_min = zoom_min
        self._zoom_max = zoom_max
        self._sky_map_fov_min = sky_map_fov_min
        self._sky_map_fov_max = sky_map_fov_max
        self._multi_cam = None
        self._overlay_timer: float = 0.0
        self._img_rect = None                           # pygame.Rect; set via set_img_rect()
        self._right_drag_start: tuple[int, int] | None = None
        self._right_drag_total: float = 0.0            # accumulated pixel travel since button-down
        self._left_drag_start: tuple[int, int] | None = None
        self._left_drag_total: float = 0.0             # accumulated pixel travel since button-down

    def register_multi_cam(self, multi_cam) -> None:
        """Register a MultiCamZoom. Zooming out past native FOV switches cameras."""
        self._multi_cam = multi_cam

    def set_img_rect(self, rect) -> None:
        """Provide the image display rect so pan/zoom can convert screen→source coords."""
        self._img_rect = rect

    # -- per-frame update ------------------------------------------------------

    def update(self, dt: float) -> None:
        """Call every frame with elapsed seconds to drive the overlay hide timer."""
        if (self._state.overlay_active
                and not self._state.overlay_pinned
                and not self._state.active_menu
                and not self._state.all_sky_mode):
            self._overlay_timer -= dt
            if self._overlay_timer <= 0.0:
                self._state.overlay_active = False
                self._overlay_timer = 0.0

    def show_overlay(self) -> None:
        """Show the overlay and (re)arm the auto-hide timer, as if the mouse had moved."""
        self._state.overlay_active = True
        self._overlay_timer = OVERLAY_DURATION

    # -- event handlers --------------------------------------------------------

    def on_scroll(self, delta: int) -> None:
        """
        Handle a scroll-wheel event.

        delta > 0 : scroll up   → zoom in  / menu up
        delta < 0 : scroll down → zoom out / menu down

        Zoom always keeps the current view center fixed -- it never re-centers
        on the cursor.
        """
        match self._context():
            case ScrollContext.MENU:
                self._menu.scroll(delta)
            case ScrollContext.SKY_MAP:
                self._zoom_sky_map(-delta)  # invert: scroll up = zoom in = smaller FOV
            case ScrollContext.IMAGE | ScrollContext.OVERLAY:
                self._zoom_image(delta)  # scroll up = zoom in

    def on_right_button_down(self, x: int, y: int) -> None:
        """Record the start of a right-button press for drag/click detection."""
        self._right_drag_start = (x, y)
        self._right_drag_total = 0.0

    def on_right_button_up(self) -> bool:
        """
        End right-button press.  Returns True if this was a click (not a drag)
        so the caller can open the context menu.
        """
        was_click = self._right_drag_total < _DRAG_THRESHOLD
        self._right_drag_start = None
        self._right_drag_total = 0.0
        return was_click

    def on_left_button_down(self, x: int, y: int) -> None:
        """
        Record the start of a left-button press for drag/click detection.

        Only meaningful in all_sky_mode (left-drag pans the sky map there);
        main.py only calls this when a left-click misses every other target
        (menu icon, edge buttons, menu items).
        """
        self._left_drag_start = (x, y)
        self._left_drag_total = 0.0

    def on_left_button_up(self) -> bool:
        """End left-button press. Returns True if this was a click (not a drag)."""
        was_click = self._left_drag_total < _DRAG_THRESHOLD
        self._left_drag_start = None
        self._left_drag_total = 0.0
        return was_click

    def on_mouse_move(
        self,
        x: int, y: int,
        right_held: bool = False,
        left_held: bool = False,
    ) -> None:
        """
        Handle mouse motion.

        right_held pans the normal (non-SkyMap) zoomed view; left_held pans
        the sky map in all_sky_mode. Only one is ever active per mode, so
        there's no conflict between them.
        """
        if self._state.mount_connected and not self._state.active_menu:
            self._state.overlay_active = True
            self._overlay_timer = OVERLAY_DURATION

        if (right_held and not self._state.all_sky_mode
                and self._right_drag_start is not None and self._img_rect is not None):
            dx = x - self._right_drag_start[0]
            dy = y - self._right_drag_start[1]
            dist = (dx * dx + dy * dy) ** 0.5
            self._right_drag_total += dist
            if not self._state.active_menu and self._right_drag_total >= _DRAG_THRESHOLD:
                self._pan(dx, dy)
            self._right_drag_start = (x, y)

        if (left_held and self._state.all_sky_mode
                and self._left_drag_start is not None and self._img_rect is not None):
            dx = x - self._left_drag_start[0]
            dy = y - self._left_drag_start[1]
            dist = (dx * dx + dy * dy) ** 0.5
            self._left_drag_total += dist
            if not self._state.active_menu and self._left_drag_total >= _DRAG_THRESHOLD:
                self._pan(dx, dy)
            self._left_drag_start = (x, y)

    # -- internal helpers ------------------------------------------------------

    def _context(self) -> ScrollContext:
        if self._state.active_menu:
            return ScrollContext.MENU
        if self._state.all_sky_mode:
            return ScrollContext.SKY_MAP
        if self._state.overlay_active:
            return ScrollContext.OVERLAY
        return ScrollContext.IMAGE

    def _zoom_image(self, delta: int) -> None:
        """Zoom about the current view center -- zoom_center_x/y never move here."""
        step = self._zoom_step ** abs(delta)
        old_zoom = self._state.zoom_level
        if delta > 0:
            if self._state.all_sky_mode:
                self._state.all_sky_mode = False
                self._state.overlay_active = self._state.overlay_pinned
            self._state.zoom_level = min(old_zoom * step, self._zoom_max)
        else:
            new_zoom = old_zoom / step
            if new_zoom < self._zoom_min:
                if self._multi_cam and self._multi_cam.step_out(self._state):
                    pass   # camera switched; MultiCamZoom reset zoom_level
                else:
                    self._state.zoom_level = self._zoom_min
                    self._state.zoom_center_x = 0.5
                    self._state.zoom_center_y = 0.5
            else:
                self._state.zoom_level = new_zoom

    def _zoom_sky_map(self, delta: int) -> None:
        """
        Zoom the sky map FOV.
        delta < 0 (scroll down) → zoom in → smaller (narrower) FOV
        delta > 0 (scroll up)   → zoom out → larger (wider) FOV
        """
        step = self._zoom_step ** abs(delta)
        if delta < 0:
            new_fov = self._state.sky_map_fov / step
        else:
            new_fov = self._state.sky_map_fov * step
        self._state.sky_map_fov = max(self._sky_map_fov_min,
                                       min(self._sky_map_fov_max, new_fov))

    def _pan(self, screen_dx: int, screen_dy: int) -> None:
        """
        Pan the view.  screen_dx/dy are pixel displacement since the last sample;
        positive values shift content right/down on screen.

        In sky_map mode: unconstrained pan that moves the sky centre.
        In normal zoom mode: constrained within image extents.
        """
        if self._state.all_sky_mode:
            # Free pan — zoom_center represents angular offset from mount pointing
            self._state.zoom_center_x -= screen_dx / self._img_rect.width
            self._state.zoom_center_y -= screen_dy / self._img_rect.height
            return
        zoom = self._state.zoom_level
        if zoom <= 1.0:
            return
        half_x = 0.5 / zoom
        half_y = 0.5 / zoom
        # Dragging right (screen_dx > 0) moves image right → source window moves left
        cx = self._state.zoom_center_x - screen_dx / (self._img_rect.width * zoom)
        cy = self._state.zoom_center_y - screen_dy / (self._img_rect.height * zoom)
        self._state.zoom_center_x = max(half_x, min(1.0 - half_x, cx))
        self._state.zoom_center_y = max(half_y, min(1.0 - half_y, cy))
