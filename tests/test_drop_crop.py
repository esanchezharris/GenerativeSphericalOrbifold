"""GEM's texture drop and tight crop, ported to the sphere path."""

from __future__ import annotations

import numpy as np
import torch

from escher.main_sphere import drop_textures
from escher.rendering.crop_rendering import crop_composited


def test_drop_keeps_element_zero_and_flattens_the_rest():
    np.random.seed(0)
    tex = torch.rand(16, 16, 3)
    out = drop_textures(tex, batch=6, prob_percent=100.0)
    assert out.shape == (6, 16, 16, 3)
    assert torch.equal(out[0], tex), "element 0 must never be dropped"
    for i in range(1, 6):
        flat = out[i]
        assert flat.std() < 1e-6, "dropped elements are a single flat color"


def test_dropped_elements_send_zero_texture_gradient():
    np.random.seed(0)
    tex = torch.rand(8, 8, 3, requires_grad=True)
    out = drop_textures(tex, batch=4, prob_percent=100.0)
    # A loss touching ONLY dropped elements must not move the texture.
    out[1:].sum().backward()
    assert tex.grad is not None
    assert float(tex.grad.abs().sum()) == 0.0


def test_drop_prob_zero_is_identity_view():
    tex = torch.rand(8, 8, 3)
    out = drop_textures(tex, batch=3, prob_percent=0.0)
    assert torch.equal(out[1], tex) and torch.equal(out[2], tex)


def _frame_with_disk(b, h, w, cy, cx, r):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    inside = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).float()
    alpha = inside.reshape(1, h, w, 1).repeat(b, 1, 1, 1)
    composited = torch.rand(b, h, w, 3) * alpha + 0.5 * (1 - alpha)
    return composited, alpha


def test_crop_makes_the_figure_fill_the_frame():
    composited, alpha = _frame_with_disk(2, 128, 128, 40, 90, 12)
    out = crop_composited(composited, alpha)
    assert out.shape == composited.shape
    # In the cropped frame the (resampled) figure diameter should be close to
    # frame/1.1; measure by variance concentration: the un-cropped frame is
    # mostly flat 0.5 background, the cropped one mostly textured figure.
    var_before = composited[0].var()
    var_after = out[0].var()
    assert var_after > 2 * var_before


def test_crop_empty_alpha_passes_through():
    composited = torch.rand(2, 64, 64, 3)
    alpha = torch.zeros(2, 64, 64, 1)
    out = crop_composited(composited, alpha)
    assert torch.equal(out, composited)


def test_crop_full_alpha_is_near_identity():
    composited = torch.rand(1, 64, 64, 3)
    alpha = torch.ones(1, 64, 64, 1)
    out = crop_composited(composited, alpha)
    assert torch.allclose(out, composited, atol=1e-5)


def test_crop_is_differentiable_and_out_of_place():
    composited, alpha = _frame_with_disk(1, 64, 64, 32, 32, 10)
    composited = composited.clone().requires_grad_(True)
    before = composited.detach().clone()
    out = crop_composited(composited, alpha)
    out.mean().backward()
    assert composited.grad is not None and composited.grad.abs().sum() > 0
    assert torch.equal(composited.detach(), before), "input must not be mutated"
