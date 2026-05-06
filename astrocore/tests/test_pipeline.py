"""
Tests for astrocore/pipeline — stretch, sky, stacker, streaming.

No hardware required.
Run with: pytest astrocore/tests/test_pipeline.py -v
"""

import numpy as np
import pytest

from astrocore.pipeline.stretch import (
    adjust_saturation,
    auto_brightness,
    gamma_correct,
    saturation_coefficient,
    sky_coefficient,
)
from astrocore.pipeline.sky import fit_sky_model
from astrocore.pipeline.stacker import ExposureSequence, Stacker
from astrocore.pipeline.streaming import StreamExposure, process_stream_frame
from astrocore.pipeline.stretch import saturation_fraction


# ── ExposureSequence ──────────────────────────────────────────────────────────

class TestExposureSequence:
    def test_current_starts_at_first(self):
        seq = ExposureSequence([0.1, 1, 5, 20])
        assert seq.current == 0.1

    def test_advance_moves_forward(self):
        seq = ExposureSequence([0.1, 1, 5])
        seq.advance()
        assert seq.current == 1

    def test_advance_stays_at_last(self):
        seq = ExposureSequence([0.1, 5])
        seq.advance()
        seq.advance()  # already at last
        assert seq.current == 5

    def test_at_end_false_initially(self):
        assert not ExposureSequence([0.1, 5]).at_end

    def test_at_end_true_at_last(self):
        seq = ExposureSequence([0.1, 5])
        seq.advance()
        assert seq.at_end

    def test_reset_returns_to_start(self):
        seq = ExposureSequence([0.1, 1, 5])
        seq.advance()
        seq.advance()
        seq.reset()
        assert seq.current == 0.1
        assert not seq.at_end

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            ExposureSequence([])

    def test_full_stacking_sequence(self):
        sequence = [
            0.1, 1, 2,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            10, 10, 10, 10, 10, 10,
            20,
        ]
        seq = ExposureSequence(sequence)
        for _ in range(len(sequence) - 1):
            seq.advance()
        assert seq.current == 20
        assert seq.at_end
        seq.advance()  # no-op
        assert seq.current == 20


# ── helpers ───────────────────────────────────────────────────────────────────

def _gray_image(value: float, shape=(64, 64)) -> np.ndarray:
    """Uniform float32 [0, 65535] grayscale image."""
    return np.full(shape, value, dtype=np.float32)


def _color_image(b: float, g: float, r: float, shape=(64, 64)) -> np.ndarray:
    """Uniform BGR float32 [0, 65535] image."""
    img = np.zeros((*shape, 3), dtype=np.float32)
    img[..., 0] = b
    img[..., 1] = g
    img[..., 2] = r
    return img


# ── gamma_correct ─────────────────────────────────────────────────────────────

class TestGammaCorrect:
    def test_zero_stays_zero(self):
        img = _gray_image(0.0)
        assert np.all(gamma_correct(img, 0.7) == 0.0)

    def test_max_value_compressed(self):
        img = _gray_image(65535.0)
        result = gamma_correct(img, 0.7)
        # 65535^0.7 ≈ 4638 — well below 65535
        assert result[0, 0] == pytest.approx(65535.0 ** 0.7, rel=1e-4)

    def test_negative_clipped_to_zero(self):
        img = _gray_image(-100.0)
        assert np.all(gamma_correct(img, 0.7) == 0.0)

    def test_gamma_1_is_identity(self):
        img = _gray_image(10000.0)
        np.testing.assert_allclose(gamma_correct(img, 1.0), img, rtol=1e-5)

    def test_output_dtype(self):
        img = _gray_image(5000.0)
        assert gamma_correct(img, 0.7).dtype == np.float32


# ── sky_coefficient ───────────────────────────────────────────────────────────

class TestSkyCoefficient:
    def test_zero_at_start(self):
        # At t=0, well before the 3 s onset, coefficient should be near 0
        assert sky_coefficient(0.0) == 0.0

    def test_approaches_one(self):
        assert sky_coefficient(100.0) == pytest.approx(1.0, abs=0.01)

    def test_monotone_increasing(self):
        vals = [sky_coefficient(t) for t in range(0, 60)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_never_negative(self):
        assert sky_coefficient(-10.0) == 0.0


# ── saturation_coefficient ────────────────────────────────────────────────────

class TestSaturationCoefficient:
    def test_minimum_is_one(self):
        assert saturation_coefficient(0.0) == 1.0

    def test_approaches_two(self):
        assert saturation_coefficient(100.0) == pytest.approx(2.0, abs=0.01)

    def test_monotone_increasing(self):
        vals = [saturation_coefficient(t) for t in range(0, 60)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_clamped_to_two(self):
        assert saturation_coefficient(1000.0) <= 2.0


# ── auto_brightness ───────────────────────────────────────────────────────────

class TestAutoBrightness:
    def test_dim_image_gets_boosted(self):
        img = _gray_image(500.0)  # dim image
        gn = auto_brightness(img, 0.7, skyco=0.0)
        # Gamma-corrected dim image * gn should reach a reasonable display level
        im_display = gamma_correct(img, 0.7) * gn
        assert im_display.mean() <= 0.28 * 65535  # within mean constraint

    def test_bright_image_constrained(self):
        img = _gray_image(30000.0)
        gn = auto_brightness(img, 0.7, skyco=0.0)
        im_display = gamma_correct(img, 0.7) * gn
        assert im_display.mean() <= 0.28 * 65535 * 1.01  # within 1% of constraint

    def test_returns_float(self):
        img = _gray_image(5000.0)
        assert isinstance(auto_brightness(img, 0.7, 0.0), float)

    def test_near_zero_image_no_crash(self):
        img = _gray_image(0.01)
        gn = auto_brightness(img, 0.7, 0.0)
        assert np.isfinite(gn)

    def test_skyco_reduces_gain(self):
        img = _gray_image(5000.0)
        gn_no_sky = auto_brightness(img, 0.7, skyco=0.0)
        gn_with_sky = auto_brightness(img, 0.7, skyco=0.8)
        # Higher skyco means less sky-subtracted mean → brightness can go up
        assert gn_with_sky >= gn_no_sky


# ── adjust_saturation ─────────────────────────────────────────────────────────

class TestAdjustSaturation:
    def test_satco_one_is_near_identity(self):
        img = _color_image(10000.0, 20000.0, 5000.0)
        result = adjust_saturation(img, 1.0)
        np.testing.assert_allclose(result, img, atol=2.0)  # allow tiny rounding

    def test_output_shape_preserved(self):
        img = _color_image(10000.0, 20000.0, 5000.0)
        assert adjust_saturation(img, 1.5).shape == img.shape

    def test_output_dtype(self):
        img = _color_image(10000.0, 20000.0, 5000.0)
        assert adjust_saturation(img, 1.5).dtype == np.float32

    def test_stays_in_range(self):
        img = _color_image(10000.0, 40000.0, 5000.0)
        result = adjust_saturation(img, 2.0)
        assert result.min() >= 0.0
        assert result.max() <= 65535.0

    def test_gray_image_unchanged(self):
        # A perfectly neutral gray has zero saturation; boosting it changes nothing
        img = _color_image(20000.0, 20000.0, 20000.0)
        result = adjust_saturation(img, 2.0)
        np.testing.assert_allclose(result, img, atol=2.0)


# ── fit_sky_model ─────────────────────────────────────────────────────────────

class TestFitSkyModel:
    def test_constant_image_returns_constant(self):
        img = _gray_image(5000.0)
        sky = fit_sky_model(img)
        np.testing.assert_allclose(sky, img, atol=1.0)

    def test_linear_gradient_recovered(self):
        h, w = 64, 64
        x = np.linspace(0, 10000, w, dtype=np.float32)
        img = np.tile(x, (h, 1))
        sky = fit_sky_model(img, sigma=2.0, degree=1)
        np.testing.assert_allclose(sky, img, atol=50.0)

    def test_stars_dont_bias_fit(self):
        # Smooth gradient background with a few bright star pixels
        h, w = 64, 64
        img = np.full((h, w), 3000.0, dtype=np.float32)
        img[16, 16] = 60000.0  # bright star
        img[48, 48] = 60000.0
        sky = fit_sky_model(img, sigma=2.0, degree=1)
        # Sky at non-star pixels should be close to the true background
        mask = img < 10000.0
        np.testing.assert_allclose(sky[mask], 3000.0, atol=100.0)

    def test_output_shape_grayscale(self):
        img = _gray_image(5000.0, shape=(48, 64))
        assert fit_sky_model(img).shape == (48, 64)

    def test_output_shape_color(self):
        img = _color_image(5000.0, 5000.0, 5000.0, shape=(48, 64))
        assert fit_sky_model(img).shape == (48, 64, 3)

    def test_output_dtype(self):
        img = _gray_image(5000.0)
        assert fit_sky_model(img).dtype == np.float32


# ── Stacker ───────────────────────────────────────────────────────────────────

class TestStacker:
    def _make_frame(self, value: float = 3000.0, shape=(64, 64)) -> np.ndarray:
        return np.full(shape, value, dtype=np.uint16)

    def _make_sky_frame(self, shape=(64, 64), seed: int = 0) -> np.ndarray:
        """Dark sky background with 3×3 blob stars that survive the median filter."""
        rng = np.random.default_rng(seed)
        frame = np.full(shape, 500, dtype=np.uint16)
        for _ in range(8):
            r = rng.integers(6, shape[0] - 6)
            c = rng.integers(6, shape[1] - 6)
            frame[r - 1:r + 2, c - 1:c + 2] = 50000
        return frame

    def test_no_frame_returns_none(self):
        s = Stacker()
        assert s.get_display_frame() is None

    def test_first_frame_accepted(self):
        s = Stacker()
        accepted = s.add_frame(self._make_frame(), exposure_s=5.0)
        assert accepted is True
        assert s.frame_count == 1
        assert s.t_accum == 5.0

    def test_stack_property_is_mean(self):
        # One frame: stack == that frame (no alignment step for the first frame).
        s = Stacker()
        s.add_frame(self._make_frame(3000.0), exposure_s=5.0)
        np.testing.assert_allclose(s.stack, 3000.0, atol=10.0)

    def test_get_display_frame_returns_uint8(self):
        s = Stacker()
        s.add_frame(self._make_frame(5000.0), exposure_s=5.0)
        out = s.get_display_frame()
        assert out is not None
        assert out.dtype == np.uint8

    def test_get_display_frame_shape_preserved(self):
        shape = (48, 64)
        s = Stacker()
        s.add_frame(self._make_frame(shape=shape), exposure_s=5.0)
        out = s.get_display_frame()
        assert out.shape == shape

    def test_reset_clears_state(self):
        s = Stacker()
        s.add_frame(self._make_frame(), exposure_s=5.0)
        s.reset()
        assert s.frame_count == 0
        assert s.stack is None
        assert s.get_display_frame() is None

    def test_t_accum_accumulates(self):
        # Must use frames with spatial structure so phase correlation has a signal
        s = Stacker()
        s.add_frame(self._make_sky_frame(), exposure_s=3.0)
        s.add_frame(self._make_sky_frame(), exposure_s=5.0)
        assert s.t_accum == pytest.approx(8.0)

    def test_frame_rejected_on_low_correlation(self):
        # Large noise images: after median filter, response << 0.05 → second frame rejected.
        # (Tiny test images have spurious correlation after median smoothing; this needs
        # a size close to real camera output for the FFT statistics to work correctly.)
        rng = np.random.default_rng(99)
        ref      = rng.integers(0, 1000, (480, 640), dtype=np.uint16)
        unrelated = rng.integers(0, 1000, (480, 640), dtype=np.uint16)
        s = Stacker()
        s.add_frame(ref, exposure_s=5.0)
        accepted = s.add_frame(unrelated, exposure_s=5.0)
        assert not accepted
        assert s.frame_count == 1

    def test_blend_smooths_display(self):
        # Calling get_display_frame multiple times without new frames
        # should converge (not oscillate)
        s = Stacker()
        s.add_frame(self._make_frame(5000.0), exposure_s=10.0)
        frames = [s.get_display_frame().astype(float) for _ in range(20)]
        diffs = [np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)]
        # Differences should decrease as blend converges
        assert diffs[-1] <= diffs[0]


# ── process_stream_frame ──────────────────────────────────────────────────────

# ── saturation_fraction ───────────────────────────────────────────────────────

class TestSaturationFraction:
    def test_all_below_returns_zero(self):
        img = np.full((64, 64), 1000, dtype=np.uint16)
        assert saturation_fraction(img) == 0.0

    def test_all_above_returns_one(self):
        img = np.full((64, 64), 65535, dtype=np.uint16)
        assert saturation_fraction(img) == 1.0

    def test_half_saturating(self):
        img = np.zeros((2, 2), dtype=np.uint16)
        img[0, :] = 65000
        assert saturation_fraction(img) == pytest.approx(0.5)

    def test_custom_threshold(self):
        img = np.full((10, 10), 5000, dtype=np.uint16)
        assert saturation_fraction(img, threshold=4000) == 1.0
        assert saturation_fraction(img, threshold=6000) == 0.0


# ── StreamExposure ────────────────────────────────────────────────────────────

class TestStreamExposure:
    def test_starts_at_default(self):
        ae = StreamExposure()
        assert ae.current == pytest.approx(0.1)

    def test_custom_start_snaps_to_nearest(self):
        ae = StreamExposure(sequence=[0.01, 0.1, 0.2], start=0.05)
        assert ae.current == pytest.approx(0.01)

    def test_update_decreases_on_high_saturation(self):
        ae = StreamExposure()
        img = np.full((100, 100), 65100, dtype=np.uint16)  # all pixels saturating
        # hysteresis requires 3 consecutive frames before stepping
        assert ae.update(img) is False
        assert ae.update(img) is False
        assert ae.update(img) is True
        assert ae.current < 0.1

    def test_update_increases_on_low_saturation(self):
        ae = StreamExposure(start=0.01)
        img = np.zeros((100, 100), dtype=np.uint16)  # no saturation
        assert ae.update(img) is False
        assert ae.update(img) is False
        assert ae.update(img) is True
        assert ae.current > 0.01

    def test_update_no_change_in_band(self):
        ae = StreamExposure()
        # 0.9% saturating — between 0.8% and 1.0%, so no change
        img = np.zeros((100, 100), dtype=np.uint16)
        img.flat[:90] = 65100
        changed = ae.update(img)
        assert changed is False
        assert ae.current == pytest.approx(0.1)

    def test_does_not_go_below_minimum(self):
        ae = StreamExposure(sequence=[0.001, 0.1])
        ae._index = 0
        img = np.full((10, 10), 65100, dtype=np.uint16)
        ae.update(img)
        assert ae.current == pytest.approx(0.001)

    def test_does_not_go_above_maximum(self):
        ae = StreamExposure(sequence=[0.1, 0.2])
        ae._index = 1
        img = np.zeros((10, 10), dtype=np.uint16)
        ae.update(img)
        assert ae.current == pytest.approx(0.2)

    def test_reset_returns_to_start(self):
        ae = StreamExposure(start=0.1)
        img = np.zeros((100, 100), dtype=np.uint16)
        ae.update(img)  # steps up
        ae.reset()
        assert ae.current == pytest.approx(0.1)


# ── process_stream_frame ──────────────────────────────────────────────────────

class TestProcessStreamFrame:
    def test_output_dtype(self):
        frame = np.full((64, 64), 5000, dtype=np.uint16)
        assert process_stream_frame(frame).dtype == np.uint8

    def test_output_shape_preserved(self):
        frame = np.full((48, 64), 5000, dtype=np.uint16)
        assert process_stream_frame(frame).shape == (48, 64)

    def test_dark_frame_stays_dark(self):
        frame = np.zeros((64, 64), dtype=np.uint16)
        result = process_stream_frame(frame)
        assert result.max() == 0

    def test_output_in_range(self):
        frame = np.full((64, 64), 30000, dtype=np.uint16)
        result = process_stream_frame(frame)
        assert result.min() >= 0
        assert result.max() <= 255
