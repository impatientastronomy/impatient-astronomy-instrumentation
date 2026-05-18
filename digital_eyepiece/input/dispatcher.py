"""
InputDispatcher — routes mouse events to the right handler based on
the current viewing context.

Mouse mapping
-------------
Left-click  : toggle Stream/Stack (menu closed) or confirm selection (menu open)
Right-click : open context menu (no drag) or close menu; also exits all-sky mode
Right-drag  : pan the zoomed image
Middle-click: slew mount to cursor position (when mount connected + overlay visible)
Scroll up   : zoom in (cursor-centred) / menu up
Scroll down : zoom out / menu down

Overlay auto-hide
-----------------
When the mount is connected, moving the mouse shows the overlay for
OVERLAY_DURATION seconds. Call update(dt) every frame to drive the timer.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable

from ..view_state import ViewMode, ViewState
from .menu import Menu

OVERLAY_DURATION = 5.0   # seconds the overlay stays visible after mouse move
_RIGHT_DRAG_THRESHOLD = 5  # pixels of travel before right-hold becomes a pan


class ScrollContext(Enum):
    MENU    = auto()   # menu is open — scroll navigates items
    IMAGE   = auto()   # live/accumulate view — scroll zooms image
    OVERLAY = auto()   # overlay visible — scroll zooms image; extra out → all-sky


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
    ) -> None:
        self._state = view_state
        self._menu = menu
        self._zoom_step = zoom_step
        self._zoom_min = zoom_min
        self._zoom_max = zoom_max
        self._multi_cam = None
        self._slew_action: Callable[[], None] | None = None
        self._overlay_timer: float = 0.0
        self._img_rect = None                           # pygame.Rect; set via set_img_rect()
        self._right_drag_start: tuple[int, int] | None = None
        self._right_drag_total: float = 0.0            # accumulated pixel travel since button-down

    def register_multi_cam(self, multi_cam) -> None:
        """Register a MultiCamZoom. Zooming out past native FOV switches cameras."""
        self._multi_cam = multi_cam

    def register_slew_action(self, action: Callable[[], None]) -> None:
        """Register the mount-slew callback triggered by middle-click."""
        self._slew_action = action

    def set_img_rect(self, rect) -> None:
        """Provide the image display rect so pan/zoom can convert screen→source coords."""
        self._img_rect = rect

    # -- per-frame update ------------------------------------------------------

    def update(self, dt: float) -> None:
        """Call every frame with elapsed seconds to drive the overlay hide timer."""
        if self._state.overlay_active and not self._state.menu_open:
            self._overlay_timer -= dt
            if self._overlay_timer <= 0.0:
                self._state.overlay_active = False
                self._overlay_timer = 0.0

    # -- event handlers --------------------------------------------------------

    def on_scroll(self, delta: int, pos: tuple[int, int] | None = None) -> None:
        """
        Handle a scroll-wheel event.

        delta > 0 : scroll up   → zoom in  / menu up
        delta < 0 : scroll down → zoom out / menu down
        pos       : screen cursor position for cursor-centred zoom (optional)
        """
        match self._context():
            case ScrollContext.MENU:
                self._menu.scroll(delta)
            case ScrollContext.IMAGE | ScrollContext.OVERLAY:
                self._zoom_image(delta, pos)

    def on_left_click(self) -> None:
        """
        Menu closed → toggle Stream / Stack mode.
        Menu open   → confirm selection; close menu if a leaf was chosen.
        """
        if self._state.menu_open:
            should_close = self._menu.select()
            if should_close:
                self._state.menu_open = False
        else:
            if self._state.mode == ViewMode.LIVE:
                self._state.mode = ViewMode.ACCUMULATE
            else:
                self._state.mode = ViewMode.LIVE

    def on_right_click(self) -> None:
        """
        All-sky mode active → exit all-sky (clears overlay).
        Menu open           → close menu without selection.
        Menu closed         → open menu (reset to root).
        """
        if self._state.all_sky_mode:
            self._state.all_sky_mode = False
            self._state.overlay_active = False
        elif self._state.menu_open:
            self._state.menu_open = False
            self._menu.reset()
        else:
            self._menu.reset()
            self._state.menu_open = True

    def on_right_button_down(self, x: int, y: int) -> None:
        """Record the start of a right-button press for drag/click detection."""
        self._right_drag_start = (x, y)
        self._right_drag_total = 0.0

    def on_right_button_up(self) -> bool:
        """
        End right-button press.  Returns True if this was a click (not a drag)
        so the caller can open the context menu.
        """
        was_click = self._right_drag_total < _RIGHT_DRAG_THRESHOLD
        self._right_drag_start = None
        self._right_drag_total = 0.0
        return was_click

    def on_middle_click(self) -> bool:
        """
        Slew the mount to the cursor position when mount is connected and
        overlay is visible. Returns True if a slew was triggered (so the
        caller can display the "Caution: Mount is moving" alert).
        """
        if self._state.mount_connected and self._state.overlay_active:
            if self._slew_action is not None:
                self._slew_action()
            return True
        return False

    def on_mouse_move(self, x: int, y: int, right_held: bool = False) -> None:
        """
        Handle mouse motion.  right_held=True while the right button is pressed,
        enabling pan when the movement exceeds _RIGHT_DRAG_THRESHOLD pixels.
        """
        if self._state.mount_connected and not self._state.menu_open:
            self._state.overlay_active = True
            self._overlay_timer = OVERLAY_DURATION

        if right_held and self._right_drag_start is not None and self._img_rect is not None:
            dx = x - self._right_drag_start[0]
            dy = y - self._right_drag_start[1]
            dist = (dx * dx + dy * dy) ** 0.5
            self._right_drag_total += dist
            menu_active = getattr(self._state, "active_menu", None) or self._state.menu_open
            if not menu_active and self._right_drag_total >= _RIGHT_DRAG_THRESHOLD:
                self._pan(dx, dy)
            self._right_drag_start = (x, y)

    def on_back(self) -> None:
        """
        Exit the current submenu level. Closes the menu if already at root.
        Useful for a dedicated hardware back button.
        """
        if self._state.menu_open:
            at_root = self._menu.back()
            if at_root:
                self._state.menu_open = False

    # -- internal helpers ------------------------------------------------------

    def _context(self) -> ScrollContext:
        # active_menu is the authoritative flag; fall back to menu_open for tests
        menu_active = getattr(self._state, "active_menu", None) or self._state.menu_open
        if menu_active:
            return ScrollContext.MENU
        if self._state.overlay_active:
            return ScrollContext.OVERLAY
        return ScrollContext.IMAGE

    def _zoom_image(self, delta: int, pos: tuple[int, int] | None = None) -> None:
        step = self._zoom_step ** abs(delta)
        old_zoom = self._state.zoom_level
        if delta > 0:
            if self._state.all_sky_mode:
                self._state.all_sky_mode = False
                self._state.overlay_active = False
            new_zoom = min(old_zoom * step, self._zoom_max)
            self._state.zoom_level = new_zoom
            self._update_zoom_center(pos, old_zoom, new_zoom)
        else:
            new_zoom = old_zoom / step
            if new_zoom < self._zoom_min:
                if (self._state.mount_connected
                        and self._state.overlay_active
                        and not self._state.all_sky_mode):
                    self._state.all_sky_mode = True
                elif self._multi_cam and self._multi_cam.step_out(self._state):
                    pass   # camera switched; MultiCamZoom reset zoom_level
                else:
                    self._state.zoom_level = self._zoom_min
                    self._state.zoom_center_x = 0.5
                    self._state.zoom_center_y = 0.5
            else:
                self._state.zoom_level = new_zoom
                self._update_zoom_center(pos, old_zoom, new_zoom)

    def _update_zoom_center(
        self,
        pos: tuple[int, int] | None,
        old_zoom: float,
        new_zoom: float,
    ) -> None:
        """
        Shift zoom_center so the source pixel under pos stays at the same screen
        position after the zoom level changes.  No-op if pos or img_rect unknown.
        """
        if pos is None or self._img_rect is None or old_zoom == new_zoom:
            return
        nx = (pos[0] - self._img_rect.x) / self._img_rect.width
        ny = (pos[1] - self._img_rect.y) / self._img_rect.height
        # Source pixel currently under cursor
        src_x = self._state.zoom_center_x + (nx - 0.5) / old_zoom
        src_y = self._state.zoom_center_y + (ny - 0.5) / old_zoom
        # New center that maps src_x back to the same screen position
        half_x = 0.5 / new_zoom
        half_y = 0.5 / new_zoom
        self._state.zoom_center_x = max(half_x, min(1.0 - half_x,
            src_x - (nx - 0.5) / new_zoom))
        self._state.zoom_center_y = max(half_y, min(1.0 - half_y,
            src_y - (ny - 0.5) / new_zoom))

    def _pan(self, screen_dx: int, screen_dy: int) -> None:
        """
        Pan the zoomed view.  screen_dx/dy are the pixel displacement of the
        mouse since the last sample; positive values shift the image right/down.
        """
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
