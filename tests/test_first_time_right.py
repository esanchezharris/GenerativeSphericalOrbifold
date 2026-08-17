"""Round-4 levers: image-anchored init, texture EMA, companion warp, noise knob."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.main_shape import build_shape_run
from escher.shape_target import align_mask_to

from .test_main_shape import tiny_args


def _disk(h, w, cy, cx, r):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def test_companion_rides_the_winning_transform(tmp_path):
    ref = _disk(96, 96, 48, 48, 30)
    tgt = _disk(96, 96, 30, 60, 15)
    color = np.ones((96, 96, 3), dtype=np.float32)
    color[tgt > 0.5] = (0.9, 0.2, 0.2)  # red figure on white

    aligned, params, _ = align_mask_to(ref, tgt, match_area=True, companion=color)
    comp = params["companion"]
    assert comp.shape == (96, 96, 3)
    # Wherever the aligned mask is figure, the companion must be figure-red.
    inside = aligned > 0.5
    assert inside.sum() > 100
    assert comp[inside, 0].mean() > 0.7
    assert comp[inside, 1].mean() < 0.4
    # And white where the mask is background (away from the figure edge).
    from scipy import ndimage

    far_outside = ~ndimage.binary_dilation(inside, iterations=4)
    assert comp[far_outside].mean() > 0.9


def test_texel_surface_points_covers_valid_texels(tmp_path):
    from escher.rendering.texture_mask import texel_surface_points, uv_valid_mask

    args = tiny_args(tmp_path)
    escher = build_shape_run(args)
    pts = escher.mesh.points
    surface, mask = texel_surface_points(pts, escher.mesh.uv, escher.mesh.faces, 32)
    valid = uv_valid_mask(escher.mesh.uv, escher.mesh.faces, 32)
    # The surface mask is the un-dilated valid set; every covered texel's blend
    # must lie within the mesh's bounding sphere.
    assert mask.sum() > 0
    assert (mask & ~valid).sum() == 0
    norms = np.linalg.norm(surface[mask], axis=-1)
    assert norms.max() <= np.linalg.norm(pts, axis=1).max() + 1e-6


def test_bake_texture_init_end_to_end(tmp_path):
    import imageio.v2 as imageio

    from escher.texture_init import bake_texture_init

    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    donor = build_shape_run(args)
    ckpt = donor.save_checkpoint(3)

    color = np.full((64, 64, 3), 255, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    fig = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 20**2
    color[fig] = (40, 160, 60)  # green figure on white
    img_path = tmp_path / "color.png"
    imageio.imwrite(img_path, color)

    out = bake_texture_init(ckpt, img_path, tmp_path / "init.npy")
    baked = np.load(out)
    res = int(args.TEXTURE_RESOLUTION)
    assert baked.shape == (res, res, 3)
    assert baked.min() >= 0.0 and baked.max() <= 1.0
    # The bake fills every texel from FIGURE colors: dominantly green, nowhere white.
    assert baked[..., 1].mean() > baked[..., 0].mean()
    assert (baked.mean(-1) > 0.95).mean() < 0.05, "no white background texels"
    assert (tmp_path / "texture_init_evidence.png").exists()


def test_texture_init_path_knob(tmp_path):
    args = tiny_args(tmp_path)
    res = int(args.TEXTURE_RESOLUTION)
    arr = np.random.default_rng(0).random((res, res, 3)).astype(np.float32)
    p = tmp_path / "init.npy"
    np.save(p, arr)
    args.TEXTURE_INIT_PATH = str(p)
    escher = build_shape_run(args)
    assert torch.allclose(escher.texture.detach().cpu(), torch.as_tensor(arr))

    args.TEXTURE_INIT_PATH = str(p)
    args.TEXTURE_RESOLUTION = res * 2  # mismatched shape must be loud
    with pytest.raises(ValueError, match="TEXTURE_INIT_PATH"):
        build_shape_run(args)


def test_ema_saved_and_loadable(tmp_path):
    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    donor = build_shape_run(args)
    ema = torch.rand_like(donor.texture.detach())
    donor._texture_ema = ema
    ckpt = donor.save_checkpoint(9)

    loader = build_shape_run(args)
    loader.load_checkpoint(ckpt)  # default: the raw texture
    assert torch.allclose(loader.texture.detach().cpu(), donor.texture.detach().cpu())
    loader.load_checkpoint(ckpt, ema=True)
    assert torch.allclose(loader.texture.detach().cpu(), ema.cpu())


def test_noise_samples_config_field():
    pytest.importorskip("diffusers")
    import escher.guidance.sd as sd

    assert sd.Config().noise_samples == 1
    assert sd.Config(noise_samples=2).noise_samples == 2
