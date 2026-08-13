"""The murk yardstick must separate noise from smooth paint, and dark from bright."""

import numpy as np

from escher.metrics_murk import MURK_BAND, murk_metrics


def valid_square(res=64, pad=12):
    v = np.zeros((res, res), dtype=bool)
    v[pad:-pad, pad:-pad] = True
    return v


def test_pure_noise_scores_near_one(seed=0):
    res = 96
    rng = np.random.default_rng(123)
    m = murk_metrics(rng.random((res, res)), valid_square(res), seed=seed)
    # A random texture is, by construction, at the noise null.
    assert 0.85 < m["noise_residual_all"] < 1.15


def test_smooth_paint_scores_near_zero():
    res = 96
    ramp = np.linspace(0.2, 0.35, res)[None, :].repeat(res, 0)  # smooth, in-band
    m = murk_metrics(ramp, valid_square(res), seed=0)
    assert m["noise_residual_all"] < 0.05
    assert m["murk_fraction"] > 0.9, "the ramp lies inside the murk band"


def test_murk_fraction_counts_only_the_band_on_valid_texels():
    res = 64
    lum = np.full((res, res), 0.9)  # bright everywhere
    lum[:, : res // 2] = 0.25  # left half in the murk band
    valid = valid_square(res)
    m = murk_metrics(lum, valid, seed=0)
    assert abs(m["murk_fraction"] - 0.5) < 0.05
    assert m["dark_fraction"] == 0.0


def test_ring_luminance_reads_the_gutter_not_the_tile():
    res = 96
    lum = np.full((res, res), 1.0)
    valid = valid_square(res, pad=20)
    from scipy.ndimage import binary_dilation

    ring = binary_dilation(valid, iterations=16) & ~valid
    lum[ring] = 0.1  # a dark ring exactly where the metric should look
    m = murk_metrics(lum, valid, seed=0)
    assert m["ring_luminance"] < 0.15
    assert m["mean_luminance"] == 1.0


def test_three_channel_texture_uses_channel_zero():
    res = 64
    tex = np.zeros((res, res, 3))
    tex[:, :, 0] = 0.8
    tex[:, :, 1:] = 0.1  # dead channels under BW must not drag the mean down
    m = murk_metrics(tex, valid_square(res), seed=0)
    assert abs(m["mean_luminance"] - 0.8) < 1e-9
    assert MURK_BAND == (0.15, 0.40)
