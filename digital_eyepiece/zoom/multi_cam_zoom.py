"""
MultiCamZoom — seamless camera switching when the user zooms past the
native FOV of the current camera.

Cameras are supplied in FOV order, narrowest first:
  index 0  →  longest focal length, smallest FOV  (most zoomed in)
  index N  →  shortest focal length, largest FOV  (widest view)

When the user scrolls out past zoom = 1.0 on any camera, MultiCamZoom
steps to the next wider camera and resets zoom_level to 1.0, so the
user experiences a continuous zoom-out across the full camera range.
Zooming back in reverses the process.

Registration::

    multi_cam = MultiCamZoom(grabbers=[g0, g1, g2], fovs=[0.5, 1.0, 2.0])
    dispatcher.register_multi_cam(multi_cam)
"""

from __future__ import annotations

from astrocore.camera.frame_grabber import FrameGrabber
from ..view_state import ViewState


class MultiCamZoom:
    """
    Manages camera selection for a multi-focal-length rig.

    Parameters
    ----------
    grabbers :
        FrameGrabber instances, ordered from narrowest to widest FOV.
    fovs :
        Field-of-view in degrees for each grabber, same order.
    """

    def __init__(self, grabbers: list[FrameGrabber], fovs: list[float]) -> None:
        if len(grabbers) != len(fovs):
            raise ValueError("grabbers and fovs must have the same length.")
        if len(grabbers) == 0:
            raise ValueError("At least one camera is required.")
        self._grabbers = grabbers
        self._fovs = fovs
        self._active_index: int = 0

    # -- properties ---------------------------------------------------------

    @property
    def active_index(self) -> int:
        """Index of the currently active camera."""
        return self._active_index

    @property
    def active_grabber(self) -> FrameGrabber:
        """The FrameGrabber for the active camera."""
        return self._grabbers[self._active_index]

    @property
    def active_fov(self) -> float:
        """FOV of the active camera in degrees."""
        return self._fovs[self._active_index]

    @property
    def camera_count(self) -> int:
        return len(self._grabbers)

    # -- switching ----------------------------------------------------------

    def step_out(self, view_state: ViewState) -> bool:
        """
        Switch to the next wider camera (higher FOV, lower magnification).

        Updates view_state.active_camera_index and resets zoom_level to 1.0.
        Returns True if a switch occurred, False if already at the widest camera.
        """
        if self._active_index >= len(self._grabbers) - 1:
            return False
        self._active_index += 1
        view_state.active_camera_index = self._active_index
        view_state.zoom_level = 1.0
        return True

    def step_in(self, view_state: ViewState) -> bool:
        """
        Switch to the next narrower camera (lower FOV, higher magnification).

        Updates view_state.active_camera_index and resets zoom_level to 1.0.
        Returns True if a switch occurred, False if already at the narrowest camera.
        """
        if self._active_index <= 0:
            return False
        self._active_index -= 1
        view_state.active_camera_index = self._active_index
        view_state.zoom_level = 1.0
        return True
