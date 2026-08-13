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


def test_w_init_randn(tmp_path):
    escher, _ = bw_escher(tmp_path, W_INIT_RANDN=1.0)
    w = escher.W.detach()
    assert w.abs().sum() > 0, "randn init must not be zeros"
    assert w.dtype == torch.float64
    # deterministic under the run seed
    escher2, _ = bw_escher(tmp_path, W_INIT_RANDN=1.0)
    assert torch.equal(escher2.W.detach(), w)
    # default stays zeros (every prior run)
    escher0, _ = bw_escher(tmp_path)
    assert torch.equal(escher0.W.detach(), torch.zeros_like(escher0.W.detach()))


def test_stationarity_column_in_metrics(tmp_path):
    escher, _ = bw_escher(tmp_path)
    info = {
        "loss": 1.0,
        "silhouette": 0.0,
        "area_reg": 0.0,
        "energy": 1.0,
        "boundary_ratio": 1.0,
        "area_spread": 1.0,
        "flips": 0,
        "reverts": 0,
        "solver_iters": 5,
        "stationarity": 1.25e-8,
    }
    escher.output_dir.mkdir(parents=True, exist_ok=True)
    escher.log_metrics(0, info, fresh=True)
    lines = (escher.output_dir / "metrics.csv").read_text().splitlines()
    assert lines[0].split(",")[-1] == "stationarity"
    assert lines[1].split(",")[-1] == "1.250e-08"


def test_finalize_explicit_colorize_beats_config_tint(tmp_path, capsys):
    """The bug this pins: texture checkpoints embed TILE_TINT true, which
    silently preempted an explicit colorize request with a hue rotation --
    a no-op on BW textures, so the 'colorized' render came out gray."""
    from escher.render_final import finalize

    escher, _ = bw_escher(tmp_path, TILE_TINT=True)
    escher.output_dir.mkdir(parents=True, exist_ok=True)
    escher.save_checkpoint(0)
    finalize(
        escher.output_dir / "checkpoint.pt",
        colorize=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        turntable=False,
        gutter=False,
    )
    out = capsys.readouterr().out
    assert "colorizing" in out
    assert "tinting" not in out


def test_colorize_matrices_are_diagonal_palette_scalers(tmp_path):
    from escher.rendering.palette import assign_palette_indices, colorize_matrices

    escher, _ = bw_escher(tmp_path)
    palette = [[1.0, 0.5, 0.25], [0.2, 0.9, 0.4], [0.3, 0.3, 1.0]]
    mats = colorize_matrices(escher.tiler, escher.mesh, palette)
    idx = assign_palette_indices(escher.tiler, escher.mesh, len(palette))
    assert mats.shape == (escher.tiler.order, 3, 3)
    for g in range(escher.tiler.order):
        expect = torch.diag(torch.tensor(palette[idx[g]], dtype=torch.float32))
        assert torch.allclose(mats[g], expect)
    # off-diagonals are zero: pure channel scaling, blacks stay black
    off = mats - torch.diag_embed(torch.diagonal(mats, dim1=1, dim2=2))
    assert off.abs().max() == 0
