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
    """
    mode: ViewMode = ViewMode.LIVE
    zoom_level: float = 1.0
    zoom_center_x: float = 0.5   # normalized [0, 1]
    zoom_center_y: float = 0.5   # normalized [0, 1]
    brightness: float = 1.0
    active_camera_index: int = 0
    menu_open: bool = False
    overlay_active: bool = False
    overlay_zoom: float = 1.0
    mount_connected: bool = False
    all_sky_mode: bool = False
    recording: bool = False
    focus_state: FocusState = FocusState.OFF
    focus_center_x: float = 0.5   # normalized [0, 1] in window coords at time of click
    focus_center_y: float = 0.5
