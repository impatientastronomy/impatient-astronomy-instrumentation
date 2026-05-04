"""
Single-frame processing and auto-exposure for streaming (live preview) mode.

Classes
-------
StreamExposure  -- steps exposure up/down based on pixel saturation
"""

import numpy as np

from .stretch import auto_brightness, gamma_correct, saturation_fraction

# Default exposure ladder (seconds). Each step is ~10x the previous, matching
# the ratio of the two saturation thresholds used in update().
_DEFAULT_SEQUENCE = [0.0001, 0.001, 0.01, 0.1]

_SAT_HIGH       = 0.010   # step down if > 1.0% of pixels exceed sat_threshold_high
_SAT_LOW        = 0.008   # step up   if < 0.8% of pixels exceed sat_threshold_low
_THRESHOLD_HIGH = 65000   # ADU — near sensor saturation
_THRESHOLD_LOW  =  6500   # ADU — ~10% of full scale; ratio matches the 10x exposure steps


class StreamExposure:
    """
    Auto-exposure controller for streaming mode.

    Uses two separate ADU thresholds to avoid toggling between adjacent steps:
      - If > 1% of pixels exceed 65000 ADU  → step down (image too bright)
      - If < 0.8% of pixels exceed 6500 ADU → step up   (image too dim)

    The 10:1 ratio between thresholds matches the 10:1 ratio between adjacent
    exposure times, so a correctly-exposed image satisfies neither condition.

    Typical usage::

        ae = StreamExposure()
        while streaming:
            cam.exposure_time = ae.current
            frame = grab()
            ae.update(frame)

    Call reset() when switching away from streaming so the next session starts
    from the default exposure again.
    """

    def __init__(
        self,
        sequence: list[float] | None = None,
        start: float = 0.1,
        sat_threshold_high: int = _THRESHOLD_HIGH,
        sat_threshold_low: int = _THRESHOLD_LOW,
        sat_high: float = _SAT_HIGH,
        sat_low: float = _SAT_LOW,
    ) -> None:
        self._sequence = list(sequence or _DEFAULT_SEQUENCE)
        self._sat_threshold_high = sat_threshold_high
        self._sat_threshold_low = sat_threshold_low
        self._sat_high = sat_high
        self._sat_low = sat_low
        self._start_index = min(
            range(len(self._sequence)),
            key=lambda i: abs(self._sequence[i] - start),
        )
        self._index = self._start_index

    @property
    def current(self) -> float:
        """Current exposure time in seconds."""
        return self._sequence[self._index]

    def update(self, image: np.ndarray) -> bool:
        """
        Examine image saturation and step the exposure up or down if needed.

        image   : raw uint16 or float32 frame from grab_frame() (pre-display)
        Returns : True if the exposure changed, False if it stayed the same.
        """
        if saturation_fraction(image, self._sat_threshold_high) > self._sat_high:
            new_index = max(0, self._index - 1)
        elif saturation_fraction(image, self._sat_threshold_low) < self._sat_low:
            new_index = min(len(self._sequence) - 1, self._index + 1)
        else:
            return False

        if new_index == self._index:
            return False
        self._index = new_index
        return True

    def reset(self) -> None:
        """Return to the starting exposure."""
        self._index = self._start_index


def process_stream_frame(frame: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """
    Process a single calibrated frame for streaming display.

    Steps: gamma correction → auto-brightness → uint8 output.

    frame  : uint16 or float32 from grab_frame(), shape (H, W) or (H, W, C)
    Returns: processed image, same shape, dtype uint8
    """
    img = frame.astype(np.float32)
    gn = auto_brightness(img, gamma, skyco=0.0)
    img = np.clip(gamma_correct(img, gamma) * gn, 0.0, 65535.0)
    return (img / 256.0).astype(np.uint8)
