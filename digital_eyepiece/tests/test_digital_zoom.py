"""
Tests for digital_eyepiece/zoom/digital_zoom.py.
Run with: pytest digital_eyepiece/tests/test_digital_zoom.py -v
"""

import numpy as np
import pytest

from digital_eyepiece.zoom.digital_zoom import apply_digital_zoom


def _solid(h: int, w: int, value: int = 128) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint16)


class TestApplyDigitalZoom:
    def test_zoom_one_returns_unchanged(self):
        img = _solid(100, 100)
        result = apply_digital_zoom(img, zoom=1.0)
        assert result is img

    def test_zoom_below_one_returns_unchanged(self):
        img = _solid(100, 100)
        result = apply_digital_zoom(img, zoom=0.5)
        assert result is img

    def test_output_shape_matches_input(self):
        img = _solid(200, 300)
        result = apply_digital_zoom(img, zoom=2.0)
        assert result.shape == (200, 300)

    def test_output_shape_matches_input_color(self):
        img = np.zeros((100, 100, 3), dtype=np.uint16)
        result = apply_digital_zoom(img, zoom=2.0)
        assert result.shape == (100, 100, 3)

    def test_center_zoom_uses_center_region(self):
        # Image split left=0, right=255; center zoom should land in right half
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        result = apply_digital_zoom(img, zoom=2.0, center_x=0.75, center_y=0.5)
        assert result.mean() > 200   # majority of crop is from the bright right half

    def test_zoom_clamps_at_image_edge(self):
        img = _solid(100, 100)
        # center_x=0.0 forces crop to the far left — should not raise
        result = apply_digital_zoom(img, zoom=4.0, center_x=0.0, center_y=0.0)
        assert result.shape == (100, 100)

    def test_high_zoom_does_not_raise(self):
        img = _solid(100, 100)
        result = apply_digital_zoom(img, zoom=50.0)
        assert result.shape == (100, 100)
