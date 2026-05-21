"""
ViewState — the single source of truth for what the main loop renders.

Input handlers write to ViewState; the display pipeline reads from it.
This keeps input handling and rendering fully decoupled: neither side
needs to know how the other is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class ViewMode(Enum):
    LIVE       = auto()   # display the most recent frame
    ACCUMULATE = auto()   # display the running stack


class FocusState(Enum):
    OFF     = auto()   # normal operation
    WAITING = auto()   # 'f' pressed — showing crosshairs, awaiting click
    ACTIVE  = auto()   # ROI defined — displaying zoomed focus patch


@dataclass
class ViewState:
    """
    Mutable display state shared across the main loop, input dispatcher,
    and rendering pipeline.

    Zoom
    ----
    zoom_level applies to the image rendered from the active camera.
    1.0 is native (full sensor visible). Values > 1.0 are digital crop-zoom.
    overlay_zoom is a separate zoom for the all-sky star-chart overlay and
    can go below 1.0 (zooming out beyond the camera FOV).

    Multi-camera
    ------------
    active_camera_index is the index into the list of connected FrameGrabbers.
    The InputDispatcher (via MultiCamZoom) updates this when the user zooms
    out past the current camera's native FOV.

    Menus
    -----
    active_menu names which panel is open: 'action', 'controls', 'utilities',
    'context', or None. menu_open is kept for dispatcher/test compatibility.
    """
    mode: ViewMode = ViewMode.LIVE
    zoom_level: float = 1.0
    zoom_center_x: float = 0.5   # normalized [0, 1]
    zoom_center_y: float = 0.5   # normalized [0, 1]
    brightness: float = 1.0
    sky_subtraction: float = 1.0        # multiplier on the auto-computed skyco
    stream_exposure: float | None = None  # None = auto-exposure algorithm
    stack_exposure: float | None = None   # None = STACKING_SEQUENCE ramp
    active_camera_index: int = 0
    active_menu: str | None = None      # 'action' | 'controls' | 'utilities' | 'context' | None
    menu_open: bool = False             # kept for dispatcher / test compatibility
    overlay_active: bool = False
    sky_map_fov: float = 20.0       # active FOV (degrees) while in all_sky_mode
    moon_mode: bool = False         # True = moon feature overlay instead of sky catalog
    mount_connected: bool = False
    mount_tracking:  bool = False   # True only after connect confirms tracking or sync
    all_sky_mode: bool = False
    paused: bool = False
    recording: bool = False
    focus_state: FocusState = FocusState.OFF
    focus_center_x: float = 0.5   # normalized [0, 1] in window coords at time of click
    focus_center_y: float = 0.5
    context_menu_pos: tuple[int, int] = (0, 0)  # screen position of context menu
    sky_map_cam_fov_h: float | None = None   # camera horizontal FOV (deg) at SkyMap entry
    sky_map_cam_fov_v: float | None = None   # camera vertical FOV (deg) at SkyMap entry
