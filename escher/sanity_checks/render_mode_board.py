"""Palette x colour-mode boards and the cross-era retrospective, stills only.

Round-18 context: the user judged the R16b/R17 fill-penalty spheres worse and
asked (a) for less obvious tile edges, (b) whether a fourth colour -- "maybe
white?" -- helps, and (c) for a fresh review across ALL eras because "the fish
era seemed the strongest in retrospect". Everything here is render-time: the
same checkpoints, re-presented.

Two products, both matplotlib grids written under ``--out``:

board
    One sheet per checkpoint: rows = palettes, cols = colour modes
    (:func:`escher.rendering.palette.apply_tile_color`). The same two fixed
    orbit views in every cell, so cells compare.

retro (``--retro``)
    One row per checkpoint, three columns: the historical ``final.png`` crop
    exactly as shipped that round | a fresh flat re-render (all eras rendered
    identically -- the apples-to-apples column) | the figure-mode
    XMAS_WHITE render (every era under the round-18 treatment).

Deliberately NOT ``finalize`` per cell: that writes an OBJ and a 120-frame
video each call. Stills via render_tiled_sphere are ~1-2 s per cell.

Usage::

    python escher/sanity_checks/render_mode_board.py --out runs/r18_review/board \
        --checkpoints runs/r17_morning/stocking/tex_s0/checkpoint.pt [...] \
        --palettes xmas,xmas_white,xmas4 --modes flat,figure,ink
    python escher/sanity_checks/render_mode_board.py --out runs/r18_review/retro \
        --retro --checkpoints ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from escher.rendering.palette import (
    PASTEL_PALETTE,
    XMAS4_PALETTE,
    XMAS_PALETTE,
    XMAS_WHITE_PALETTE,
    colorize_matrices,
)

PALETTES = {
    "xmas": XMAS_PALETTE,
    "xmas4": XMAS4_PALETTE,
    "xmas_white": XMAS_WHITE_PALETTE,
    "pastel": PASTEL_PALETTE,
}


def _load(checkpoint: Path):
    """Checkpoint -> (escher, tiled sphere, label). GPU-light: one CPU solve."""
    from escher.render_final import apply_gutter, load_run
    from escher.rendering.render_sphere_nvdiffrast import build_tiled_sphere

    escher, iteration = load_run(checkpoint)
    apply_gutter(escher)
    with torch.no_grad():
        points = escher.solve_points()
    sphere = build_tiled_sphere(
        points.to(escher.device).float(), escher.mesh.faces, escher.mesh.uv, escher.tiler
    )
    return escher, sphere, iteration


def _stills(escher, sphere, palette, mode, gate, n_views) -> np.ndarray:
    """Fixed orbit stills composited over white; (V, H, W, 3) in [0, 1]."""
    from escher.rendering.camera import orbit_views
    from escher.rendering.render_sphere_nvdiffrast import render_tiled_sphere

    mtx = None
    if palette is not None:
        mtx = colorize_matrices(escher.tiler, escher.mesh, [list(c) for c in palette])
    views = orbit_views(
        n_views, distance=float(escher.args.get("PREVIEW_DISTANCE", 3.0)), elevation_deg=15.0
    )
    shade = float(escher.args.get("SHADE_AMBIENT", 0.55))
    with torch.no_grad():
        images, alpha = render_tiled_sphere(
            sphere,
            escher.effective_texture(),
            mv=views,
            image_size=int(escher.args.RENDER_SIZE),
            tile_color_matrices=mtx,
            shade_ambient=shade,
            color_mode=mode,
            color_gate=gate,
        )
        comp = (images * alpha + 1.0 * (1 - alpha)).clamp(0, 1).cpu().numpy()
    return comp


def _strip(cells: np.ndarray) -> np.ndarray:
    """(V, H, W, 3) -> one horizontal strip."""
    return np.concatenate(list(cells), axis=1)


def _label_of(escher, checkpoint: Path) -> str:
    run = checkpoint.parent.parent  # <batch>/<arm>/tex_s*/checkpoint.pt
    prompt = str(escher.args.get("PROMPT", ""))
    subject = prompt.split(" of ")[-1].split(",")[0] if " of " in prompt else prompt[:40]
    return f"{run.parent.name}/{run.name}\n{subject}"


def board(checkpoints, palettes, modes, gate, n_views, out: Path) -> list[Path]:
    written = []
    for ck in checkpoints:
        ck = Path(ck)
        try:
            escher, sphere, _ = _load(ck)
        except Exception as e:  # legacy checkpoints must not kill the batch
            print(f"SKIP {ck}: {type(e).__name__}: {e}")
            continue
        fig, axes = plt.subplots(
            len(palettes), len(modes), figsize=(5.4 * n_views * len(modes) / 2, 2.9 * len(palettes))
        )
        axes = np.atleast_2d(axes)
        for r, pname in enumerate(palettes):
            for c, mode in enumerate(modes):
                cells = _stills(escher, sphere, PALETTES[pname], mode, gate, n_views)
                axes[r, c].imshow(_strip(cells))
                axes[r, c].set_axis_off()
                axes[r, c].set_title(f"{pname} | {mode}", fontsize=9)
        fig.suptitle(_label_of(escher, ck), fontsize=11)
        fig.tight_layout()
        path = out / f"board_{ck.parent.parent.name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        written.append(path)
    return written


def retro(checkpoints, gate, n_views, out: Path, sheet_name: str) -> Path | None:
    rows = []
    for ck in checkpoints:
        ck = Path(ck)
        try:
            escher, sphere, _ = _load(ck)
        except Exception as e:
            print(f"SKIP {ck}: {type(e).__name__}: {e}")
            continue
        historical = None
        hist_path = ck.parent / "final.png"
        if hist_path.exists():
            historical = plt.imread(hist_path)
        flat = _strip(_stills(escher, sphere, None, "flat", gate, n_views))
        treated = _strip(
            _stills(escher, sphere, XMAS_WHITE_PALETTE, "figure", gate, n_views)
        )
        rows.append((_label_of(escher, ck), historical, flat, treated))

    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (label, historical, flat, treated) in enumerate(rows):
        for c, (img, title) in enumerate(
            [
                (historical, "as shipped that round"),
                (flat, "fresh render, no colour"),
                (treated, "figure mode, xmas+white"),
            ]
        ):
            ax = axes[r, c]
            ax.set_axis_off()
            if img is not None:
                ax.imshow(img)
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r, 0].text(
            -0.02, 0.5, label, transform=axes[r, 0].transAxes, fontsize=8,
            ha="right", va="center",
        )
    fig.tight_layout()
    path = out / sheet_name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--palettes", default="xmas,xmas_white,xmas4")
    ap.add_argument("--modes", default="flat,figure,ink")
    ap.add_argument("--gate", default="0.65,0.85")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--retro", action="store_true")
    ap.add_argument("--sheet-name", default="retrospective.png")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gate = tuple(float(v) for v in args.gate.split(","))

    if args.retro:
        retro(args.checkpoints, gate, args.views, out, args.sheet_name)
    else:
        board(
            args.checkpoints,
            [p.strip() for p in args.palettes.split(",")],
            [m.strip() for m in args.modes.split(",")],
            gate,
            args.views,
            out,
        )


if __name__ == "__main__":
    main()
