"""BW (upstream greyscale parity) and the random texture init.

BW constrains the paint at the parameter level -- channel 0 tiled to RGB
(upstream main.py:624-625 semantics) -- so SDS sees 3-channel images but can
only deposit luminance. Off must stay identical to every prior run.
"""

from __future__ import annotations

import torch

from escher.main_shape import build_shape_run
from escher.main_sphere import drop_textures

from .test_main_shape import tiny_args


def bw_escher(tmp_path, **over):
    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    for k, v in over.items():
        setattr(args, k, v)
    return build_shape_run(args), args


def test_bw_off_returns_the_parameter_itself(tmp_path):
    escher, _ = bw_escher(tmp_path)
    assert escher.effective_texture() is escher.texture


def test_bw_tiles_channel_zero(tmp_path):
    escher, _ = bw_escher(tmp_path, BW=True)
    with torch.no_grad():
        escher.texture[:] = torch.rand_like(escher.texture)
    eff = escher.effective_texture()
    assert eff.shape == escher.texture.shape
    assert torch.equal(eff[..., 0], escher.texture[..., 0])
    assert torch.equal(eff[..., 1], escher.texture[..., 0])
    assert torch.equal(eff[..., 2], escher.texture[..., 0])


def test_bw_gradient_reaches_only_channel_zero(tmp_path):
    escher, _ = bw_escher(tmp_path, BW=True)
    eff = escher.effective_texture()
    eff.sum().backward()
    g = escher.texture.grad
    assert g is not None
    assert g[..., 0].abs().sum() > 0
    assert g[..., 1].abs().sum() == 0
    assert g[..., 2].abs().sum() == 0


def test_bw_composes_with_drop(tmp_path):
    escher, _ = bw_escher(tmp_path, BW=True)
    with torch.no_grad():
        escher.texture[:] = torch.rand_like(escher.texture)
    dropped = drop_textures(escher.effective_texture(), 4, 100.0)
    # Every batch element is greyscale: kept = the tiled parameter, dropped =
    # flat random gray (achromatic by construction).
    assert torch.equal(dropped[..., 0], dropped[..., 1])
    assert torch.equal(dropped[..., 1], dropped[..., 2])


def test_random_texture_init(tmp_path):
    escher, _ = bw_escher(tmp_path, TEXTURE_INIT_RANDOM=True)
    t = escher.texture.detach()
    # uniform(0,1) has std ~0.29; every flat init has std exactly 0.
    assert t.std() > 0.1
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_init_path_takes_precedence_over_random(tmp_path):
    import numpy as np

    escher0, args0 = bw_escher(tmp_path)
    res = int(args0.TEXTURE_RESOLUTION)
    arr = np.full((res, res, 3), 0.25, dtype=np.float32)
    p = tmp_path / "init.npy"
    np.save(p, arr)
    escher, _ = bw_escher(
        tmp_path, TEXTURE_INIT_PATH=str(p), TEXTURE_INIT_RANDOM=True
    )
    assert torch.allclose(escher.texture.detach().cpu(), torch.full((res, res, 3), 0.25))
