"""The white-fill penalty must charge for blank texels and spare real paint.

The rosettes are unpainted pockets at the rotation centres. The pre-existing
fill loss is a CHROMA floor, which is useless here: under BW chroma is
identically zero everywhere, so it has nothing to push on. This penalty keys on
luminance instead, and can be focused on the corners so it does not flatten
legitimate white elsewhere.
"""

from __future__ import annotations

import torch

from escher.main_sphere import corner_weight_map, texture_white_loss


def test_painted_tile_costs_nothing():
    tex = torch.full((16, 16), 0.4)
    valid = torch.ones(16, 16, dtype=torch.bool)
    assert float(texture_white_loss(tex, valid, 0.85)) == 0.0


def test_light_paint_below_the_level_is_untouched():
    """Only genuinely blank texels are charged -- 'merely light' must be free."""
    tex = torch.full((16, 16), 0.84)
    valid = torch.ones(16, 16, dtype=torch.bool)
    assert float(texture_white_loss(tex, valid, 0.85)) == 0.0


def test_blank_texels_are_charged_and_the_gradient_darkens_them():
    tex = torch.full((16, 16), 1.0, requires_grad=True)
    valid = torch.ones(16, 16, dtype=torch.bool)
    loss = texture_white_loss(tex, valid, 0.85)
    assert float(loss) > 0.0
    loss.backward()
    # gradient must push luminance DOWN (positive grad on a minimised loss)
    assert float(tex.grad.min()) > 0.0


def test_only_valid_texels_count():
    """The gutter is not sampled by any triangle -- charging it would be noise."""
    tex = torch.ones(8, 8)
    valid = torch.zeros(8, 8, dtype=torch.bool)
    valid[0, 0] = True
    tex = tex.clone()
    tex[0, 0] = 0.2
    assert float(texture_white_loss(tex, valid, 0.85)) == 0.0


def test_bw_channel_zero_is_what_is_measured():
    tex = torch.zeros(8, 8, 3)
    tex[..., 0] = 1.0  # channel 0 is what BW renders
    valid = torch.ones(8, 8, dtype=torch.bool)
    assert float(texture_white_loss(tex, valid, 0.85)) > 0.0


def test_corner_weighting_focuses_the_penalty():
    """A pocket at a corner must cost more than the same pocket mid-tile."""
    res = 64
    uv = torch.tensor([[0.5, 0.05], [0.5, 0.5]])  # a corner, and the middle
    w = corner_weight_map(uv, [0], res, radius=0.12)

    valid = torch.ones(res, res, dtype=torch.bool)
    at_corner = torch.full((res, res), 0.4)
    at_corner[int(0.05 * res) - 1 : int(0.05 * res) + 2, res // 2 - 1 : res // 2 + 2] = 1.0
    mid = torch.full((res, res), 0.4)
    mid[res // 2 - 1 : res // 2 + 2, res // 2 - 1 : res // 2 + 2] = 1.0

    c = float(texture_white_loss(at_corner, valid, 0.85, w))
    m = float(texture_white_loss(mid, valid, 0.85, w))
    assert c > 10 * m, f"corner pocket {c:.3e} should dominate interior {m:.3e}"


def test_uniform_weighting_treats_the_tile_equally():
    res = 32
    valid = torch.ones(res, res, dtype=torch.bool)
    a = torch.full((res, res), 0.4)
    a[0:3, 0:3] = 1.0
    b = torch.full((res, res), 0.4)
    b[res // 2 : res // 2 + 3, res // 2 : res // 2 + 3] = 1.0
    assert float(texture_white_loss(a, valid, 0.85)) == float(
        texture_white_loss(b, valid, 0.85)
    )
