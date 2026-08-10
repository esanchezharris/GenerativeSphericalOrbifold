"""Morning eval for a specificity batch: per-arm metrics + one comparison sheet.

For every job in a batch that produced ``tex_s0/checkpoint.pt``: run the
background gate, the outline metrics against the arm's own carve (its
``carve_s0`` if the job ran one, else the shared ``CARVE``), and render two
panels -- the tinted sphere and the OUTLINE ALONE (flat-filled isolated tile,
the panel round 2 proved decisive). Writes ``comparison.png`` + prints a table.

Usage (inside the WSL venv, GPU)::

    python escher/sanity_checks/eval_specificity_batch.py \
        ROOT=runs/gem_round3 CARVE=runs/acceptance_fish/fish_run1/carve_s0/checkpoint.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from escher.main_shape import build_shape_run
from escher.metrics_background import measure as measure_bg
from escher.metrics_outline import measure as measure_outline
from escher.rendering.camera import orbit_views, tile_centric_views

DEFAULTS = OmegaConf.create(
    {
        "ROOT": "runs/gem_specificity",
        "CARVE": "runs/acceptance_fish/fish_run1/carve_s0/checkpoint.pt",
        "N_VIEWS": 30,
    }
)


def eval_batch(root: Path, default_carve: Path, n_views: int = 30) -> None:
    arms = sorted(
        p.parent.parent for p in root.glob("*/tex_s0/checkpoint.pt")
    )
    if not arms:
        raise SystemExit(f"no */tex_s0/checkpoint.pt under {root}")

    rows, panels, failed = [], [], []
    for arm_dir in arms:
        arm = arm_dir.name
        ckpt = arm_dir / "tex_s0" / "checkpoint.pt"
        own_carve = arm_dir / "carve_s0" / "checkpoint.pt"
        carve = own_carve if own_carve.exists() else default_carve

        # One broken arm must not kill the whole morning report.
        try:
            bg = measure_bg(ckpt, n_views=n_views)
            outline = measure_outline(ckpt, carve)
        except Exception as e:  # noqa: BLE001 -- report and move on
            print(f"{arm}: metrics failed ({type(e).__name__}: {e}), skipping arm")
            failed.append(arm)
            continue

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        args = OmegaConf.create(state["config"])
        args.DEVICE = "cuda"
        args.OUTPUT_DIR = str(arm_dir / "eval_tmp")
        escher = build_shape_run(args)
        escher.load_checkpoint(ckpt)
        from escher.rendering.palette import tile_color_matrices

        tint = tile_color_matrices(
            escher.tiler,
            escher.mesh,
            list(args.get("PALETTE_HUES_DEG", [0.0, 120.0, -120.0])),
        )
        with torch.no_grad():
            mv = orbit_views(
                1, distance=float(args.get("PREVIEW_DISTANCE", 3.0)), elevation_deg=15.0
            )
            img, alpha, _ = escher.render(1, mv=mv, tint=tint, shade_ambient=0.55)
            sphere_view = (img * alpha + (1 - alpha)).clamp(0, 1)[0].cpu().numpy()

            centers = torch.as_tensor(
                escher.solo_tiler.tile_centers(escher.mesh.points), dtype=torch.float32
            )
            mv_iso = tile_centric_views(centers, 1, distance=2.4, angular_jitter_deg=0.0)
            flat = torch.full((8, 8, 3), 0.25, device=escher.device)
            img2, a2, _ = escher.render(1, mv=mv_iso, isolated=True, texture=flat)
            outline_view = (img2 * a2 + (1 - a2)).clamp(0, 1)[0].cpu().numpy()

        # EMA panel when the checkpoint carries a trajectory average.
        ema_view = None
        if "texture_ema" in state:
            escher.load_checkpoint(ckpt, ema=True)
            with torch.no_grad():
                img3, a3, _ = escher.render(1, mv=mv, tint=tint, shade_ambient=0.55)
                ema_view = (img3 * a3 + (1 - a3)).clamp(0, 1)[0].cpu().numpy()

        rows.append((arm, bg, outline))
        panels.append((arm, sphere_view, outline_view, ema_view, bg, outline))

    print(f"\n{'arm':24s} {'bg%':>7s} {'vs_tgt':>7s} {'drift':>7s} {'perim':>7s} {'flips':>5s}")
    for arm, bg, o in rows:
        print(
            f"{arm:24s} {100 * bg['bg_frac']:7.3f} {o['iou_vs_target']:7.4f} "
            f"{o['iou_vs_carve']:7.4f} {o['final_perim']:7.4f} {o['final_flips']:5d}"
        )

    n_rows = 3 if any(p[3] is not None for p in panels) else 2
    fig, axes = plt.subplots(
        n_rows, len(panels), figsize=(3.6 * len(panels), 4.0 * n_rows)
    )
    axes = np.atleast_2d(axes)
    if axes.shape[0] != n_rows:
        axes = axes.T
    for i, (arm, sphere_view, outline_view, ema_view, bg, o) in enumerate(panels):
        col = axes[:, i]
        col[0].imshow(sphere_view)
        col[0].set_title(
            f"{arm}\nbg {100 * bg['bg_frac']:.2f}% | vs-tgt {o['iou_vs_target']:.3f}",
            fontsize=9,
        )
        row = 1
        if n_rows == 3:
            if ema_view is not None:
                col[1].imshow(ema_view)
            col[1].set_title("EMA texture", fontsize=8)
            row = 2
        col[row].imshow(outline_view)
        col[row].set_title(
            f"outline | drift {o['iou_vs_carve']:.3f} | perim {o['final_perim']:.3f}",
            fontsize=8,
        )
        for ax in col:
            ax.set_axis_off()
    fig.suptitle(
        f"{root.name}: tinted sphere / EMA / outline alone", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=110, bbox_inches="tight")
    print(f"\nwrote {root / 'comparison.png'}")
    if failed:
        print(f"FAILED arms (see messages above): {', '.join(failed)}")


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    eval_batch(Path(args.ROOT), Path(args.CARVE), n_views=int(args.N_VIEWS))


if __name__ == "__main__":
    main()
