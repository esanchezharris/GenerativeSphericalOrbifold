"""Measure the inter-figure "murk" in a trained texture.

The round-12 diagnostic established what the murk IS, on a finished checkpoint:

* **Residual init noise.** With a per-texel random init and no gutter refill,
  mid-dark texels still carried 57-65% of pure-noise amplitude after 7000 steps,
  and the ~59% of the texture no triangle samples was *literally still the init
  draw* (high-pass correlation 0.72-0.79 to the reproduced draw). The mip chain
  averages that into every minified view.
* **A dark ring just outside the tile.** SDS smoothly painted the 8-16 texels
  beyond the UV boundary near-black; bilinear and low-mip taps drag it into
  every tile edge.
* **Corner pools.** Cone-adjacent faces are *compressed*, so they minify hard
  and sample high mips -- which is where the ring and the noise live.

So three numbers decide whether a run is clean, and this module computes them:

``murk_fraction``
    Share of rasterized texels in the mid-dark band -- the murk's own luminance
    range. Lower is cleaner.
``noise_residual``
    Mean ``|lum - mean of its 8 neighbours|`` over rasterized texels, as a
    fraction of the same statistic on a uniform-random field. 1.0 = the texture
    is still noise at texel scale; ~0 = smoothly painted. Reported for the
    murk band specifically, which is where leftover init noise hides.
``ring_luminance``
    Mean luminance of the gutter ring within ``RING_TEXELS`` of the tile
    boundary -- the material every edge tap averages in. Higher is cleaner
    (dark ring = dark margins between figures).

Usage::

    python escher/metrics_murk.py runs/r12_ablation/abl_white/tex_s0/checkpoint.pt

Prints a one-line summary and writes ``murk_metrics.json`` beside the checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

__all__ = ["measure", "murk_metrics"]

MURK_BAND = (0.15, 0.40)
RING_TEXELS = 16


def _neighbour_mean(a: np.ndarray) -> np.ndarray:
    """Mean of the 8 neighbours, edge-replicated. Excludes the texel itself, so
    comparing it against the texel is an unbiased roughness probe."""
    p = np.pad(a, 1, mode="edge")
    total = np.zeros_like(a, dtype=np.float64)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr == 1 and dc == 1:
                continue
            total += p[dr : dr + a.shape[0], dc : dc + a.shape[1]]
    return total / 8.0


def _ring_mask(valid: np.ndarray, width: int) -> np.ndarray:
    """Texels OUTSIDE ``valid`` within ``width`` of it -- the gutter ring that
    bilinear and low-mip taps pull into the tile's edge."""
    from scipy.ndimage import binary_dilation

    grown = binary_dilation(valid, iterations=int(width))
    return grown & ~valid


def murk_metrics(texture: np.ndarray, valid: np.ndarray, seed: int = 0) -> dict:
    """The three numbers, from a texture array and its UV validity mask."""
    lum = np.asarray(texture, dtype=np.float64)
    if lum.ndim == 3:
        lum = lum[:, :, 0]  # BW consumes channel 0; see main_sphere.effective_texture
    valid = np.asarray(valid, dtype=bool)

    resid = np.abs(lum - _neighbour_mean(lum))
    rng = np.random.default_rng(seed)
    ref = rng.random(lum.shape)
    null = float(np.abs(ref - _neighbour_mean(ref)).mean())

    band = (lum >= MURK_BAND[0]) & (lum < MURK_BAND[1])
    in_band = valid & band
    ring = _ring_mask(valid, RING_TEXELS)

    return {
        "murk_fraction": float(in_band.sum() / max(valid.sum(), 1)),
        "noise_residual_murk": float(resid[in_band].mean() / null) if in_band.any() else 0.0,
        "noise_residual_all": float(resid[valid].mean() / null),
        "noise_residual_gutter": float(resid[~valid].mean() / null),
        "ring_luminance": float(lum[ring].mean()) if ring.any() else float("nan"),
        "mean_luminance": float(lum[valid].mean()),
        "dark_fraction": float((valid & (lum < MURK_BAND[0])).sum() / max(valid.sum(), 1)),
        "valid_texels": int(valid.sum()),
    }


def measure(checkpoint: str | Path) -> dict:
    """Load a run's checkpoint and measure its texture. Writes murk_metrics.json."""
    from omegaconf import OmegaConf

    from escher.main_shape import build_shape_run
    from escher.rendering.texture_mask import uv_valid_mask

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(dict(state["config"]))
    cfg.DEVICE = "cpu"
    cfg.OUTPUT_DIR = str(checkpoint.parent)
    escher = build_shape_run(cfg)

    texture = state["texture"].detach().cpu().numpy()
    valid = uv_valid_mask(escher.mesh.uv, escher.mesh.faces, texture.shape[0])

    out = murk_metrics(texture, valid, seed=int(cfg.get("SEED", 0)))
    out["checkpoint"] = str(checkpoint)
    out["iteration"] = int(state.get("iteration", -1))
    (checkpoint.parent / "murk_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoint.pt> [...]")
    for ck in sys.argv[1:]:
        m = measure(ck)
        print(
            f"{Path(ck).parent}: murk {m['murk_fraction']:.3f} | "
            f"noise(murk) {m['noise_residual_murk']:.2f} | "
            f"noise(gutter) {m['noise_residual_gutter']:.2f} | "
            f"ring {m['ring_luminance']:.3f} | mean {m['mean_luminance']:.3f}"
        )


if __name__ == "__main__":
    main()
