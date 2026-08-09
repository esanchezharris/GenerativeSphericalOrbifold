"""Safety kit for the joint shape+texture (GEM-parity) window.

Spherical Tutte has no injectivity guarantee, so running the shape under SDS
needs the fold-rejection machinery weights mode never had: backtracking revert,
the freeze-latch certified-state guard, and honest checkpoint-mismatch warnings.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

import escher.main_sphere as ms
from escher.main_shape import build_shape_run

from .test_main_shape import tiny_args


def weights_escher(tmp_path, **over):
    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    for k, v in over.items():
        setattr(args, k, v)
    return build_shape_run(args), args


def patch_flips_relative_to_anchor(monkeypatch, escher):
    """count_flipped_faces reports folds unless W equals the certified anchor."""
    anchor = escher._W_good.clone()

    def fake(points, faces):
        return 0 if torch.allclose(escher.W.detach(), anchor) else 3

    monkeypatch.setattr(ms, "count_flipped_faces", fake)
    return anchor


def test_w_revert_backtracks_to_certified_anchor(tmp_path, monkeypatch):
    escher, _ = weights_escher(tmp_path, W_REVERT_ON_FOLD=True)
    anchor = patch_flips_relative_to_anchor(monkeypatch, escher)

    with torch.no_grad():
        escher.W.add_(1.0)  # a "step" that folds
    points, flips, reverted = escher.ensure_valid_shape()
    assert reverted
    assert flips == 0
    assert torch.allclose(escher.W.detach(), anchor), "must land on the anchor"


def test_w_revert_off_is_tripwire_only(tmp_path, monkeypatch):
    escher, _ = weights_escher(tmp_path)  # knob off (default)
    patch_flips_relative_to_anchor(monkeypatch, escher)
    with torch.no_grad():
        escher.W.add_(1.0)
    _, flips, reverted = escher.ensure_valid_shape()
    assert flips == 3 and not reverted, "default behavior: report, never mutate"


def test_freeze_latch_guard_restores_certified_state(tmp_path, monkeypatch, capsys):
    escher, _ = weights_escher(tmp_path, W_REVERT_ON_FOLD=True)
    anchor = patch_flips_relative_to_anchor(monkeypatch, escher)

    with torch.no_grad():
        escher.W.add_(1.0)  # folded state at the moment the freeze latches
    cache = escher._frozen_solve()
    assert cache["flips"] == 0
    assert torch.allclose(escher.W.detach(), anchor)
    assert "restoring the last certified W" in capsys.readouterr().out


def test_load_checkpoint_warns_on_geometry_mismatch(tmp_path, capsys):
    donor, args = weights_escher(tmp_path, COTANGENT_RELATIVE_WEIGHTS=False)
    path = donor.save_checkpoint(3)

    loader, _ = weights_escher(tmp_path, COTANGENT_RELATIVE_WEIGHTS=True)
    loader.args.OUTPUT_DIR = str(tmp_path / "loader")
    loader.load_checkpoint(path)
    out = capsys.readouterr().out
    assert "COTANGENT_RELATIVE_WEIGHTS mismatch" in out


def test_load_checkpoint_refreshes_revert_anchor(tmp_path):
    donor, _ = weights_escher(tmp_path)
    with torch.no_grad():
        donor.W.add_(0.25)
    path = donor.save_checkpoint(5)

    loader, _ = weights_escher(tmp_path)
    loader.load_checkpoint(path)
    assert torch.allclose(loader._W_good, donor.W.detach().cpu()), (
        "the revert anchor must track the LOADED state, not the init"
    )


def test_isolated_fraction_switches_at_the_freeze():
    from escher.main_sphere import isolated_fraction

    args = OmegaConf.create(
        {"ISOLATED_TILE_FRACTION": 0.5, "ISOLATED_TILE_FRACTION_FROZEN": 0.25}
    )
    assert isolated_fraction(args, frozen=False) == 0.5
    assert isolated_fraction(args, frozen=True) == 0.25
    # null (or absent) = historical behavior, both phases identical.
    args2 = OmegaConf.create(
        {"ISOLATED_TILE_FRACTION": 0.5, "ISOLATED_TILE_FRACTION_FROZEN": None}
    )
    assert isolated_fraction(args2, frozen=True) == 0.5
    assert isolated_fraction(OmegaConf.create({"ISOLATED_TILE_FRACTION": 0.5}), True) == 0.5


def test_boundary_loop_property_is_a_closed_walk(tmp_path):
    escher, _ = weights_escher(tmp_path)
    loop = escher.boundary_loop_t
    assert loop.ndim == 1 and loop.numel() >= 4
    assert loop.unique().numel() == loop.numel(), "no repeated vertices"
