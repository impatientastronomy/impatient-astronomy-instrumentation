"""
InputDispatcher — routes mouse events to the right handler based on
the current viewing context.

Mouse mapping
-------------
Left-click  : toggle Stream/Stack (menu closed) or confirm selection (menu open)
Right-click : open menu (menu closed) or close menu (menu open);
              also exits all-sky mode if active
Middle-click: slew mount to cursor position (when mount connected + overlay visible)
Scroll up   : zoom in / menu up
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
        self._multi_cam = None        # set via register_multi_cam()
        self._slew_action: Callable[[], None] | None = None
        self._overlay_timer: float = 0.0

    def register_multi_cam(self, multi_cam) -> None:
        """Register a MultiCamZoom. Zooming out past native FOV switches cameras."""
        self._multi_cam = multi_cam

    def register_slew_action(self, action: Callable[[], None]) -> None:
        """Register the mount-slew callback triggered by middle-click."""
        self._slew_action = action

    # -- per-frame update ------------------------------------------------------

    def update(self, dt: float) -> None:
        """Call every frame with elapsed seconds to drive the overlay hide timer."""
        if self._state.overlay_active and not self._state.menu_open:
            self._overlay_timer -= dt
            if self._overlay_timer <= 0.0:
                self._state.overlay_active = False
                self._overlay_timer = 0.0

    # -- event handlers --------------------------------------------------------

    def on_scroll(self, delta: int) -> None:
        """
        Handle a scroll-wheel event.

        delta > 0 : scroll up   → zoom in  / menu up
        delta < 0 : scroll down → zoom out / menu down
        """
        match self._context():
            case ScrollContext.MENU:
                self._menu.scroll(delta)
            case ScrollContext.IMAGE | ScrollContext.OVERLAY:
                self._zoom_image(delta)

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

    def on_mouse_move(self, x: int, y: int) -> None:
        """Show the overlay and reset its hide timer when the mount is connected."""
        if self._state.mount_connected and not self._state.menu_open:
            self._state.overlay_active = True
            self._overlay_timer = OVERLAY_DURATION

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
        if self._state.menu_open:
            return ScrollContext.MENU
        if self._state.overlay_active:
            return ScrollContext.OVERLAY
        return ScrollContext.IMAGE

    def _zoom_image(self, delta: int) -> None:
        step = self._zoom_step ** abs(delta)
        if delta > 0:
            # Zooming in — exit all-sky mode if active
            if self._state.all_sky_mode:
                self._state.all_sky_mode = False
                self._state.overlay_active = False
            self._state.zoom_level = min(
                self._state.zoom_level * step, self._zoom_max
            )
        else:
            new_zoom = self._state.zoom_level / step
            if new_zoom < self._zoom_min:
                if (self._state.mount_connected
                        and self._state.overlay_active
                        and not self._state.all_sky_mode):
                    # Enter all-sky mode
                    self._state.all_sky_mode = True
                elif self._multi_cam and self._multi_cam.step_out(self._state):
                    pass   # camera switched; MultiCamZoom reset zoom_level
                else:
                    self._state.zoom_level = self._zoom_min
            else:
                self._state.zoom_level = new_zoom
