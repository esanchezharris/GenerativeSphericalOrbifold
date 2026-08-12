r"""Correspondence-warped flat bake: dress an ARTICULATED outline correctly.

The rigid anchor alignment that served every convex-ish carve breaks on the
escherized outline -- a bent fish cannot wear a straight fish's markings. But
the escherize solve already computed the exact registration needed: 160 pairs
(target contour pixel w_i  <->  realized boundary pixel u_i). This module
fits a thin-plate-spline INVERSE warp (carve frame -> target frame) on those
pairs, pulls the rigidly-target-aligned anchor through it, and hands the
warped color+mask to the flat bake. The marks follow the limbs.

    python escher/warped_bake.py CARVE=<esch carve ckpt> \
        CORRESPONDENCE=<escherize>/correspondence.npz \
        TARGET=<target_aligned.png used by escherize> \
        COLOR_IMAGE=<anchor color.png> OUT=<out>/texture_init.npy
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy import ndimage
from scipy.interpolate import RBFInterpolator

from escher.shape_target import align_mask_to, figure_mask_from_flat_ground
from escher.texture_init import bake_texture_init

DEFAULTS = OmegaConf.create(
    {"CARVE": "", "CORRESPONDENCE": "", "TARGET": "", "COLOR_IMAGE": "", "OUT": ""}
)


def warped_anchor(
    correspondence: str | Path,
    target_png: str | Path,
    color_image: str | Path,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-align the anchor to the TARGET mask, then TPS-warp into the carve
    frame via the boundary correspondence. Returns (color, mask), carve frame."""
    pairs = np.load(correspondence)
    u_px, w_px = pairs["u_px"], pairs["w_px"]

    target = np.asarray(imageio.imread(target_png), dtype=np.float64)
    target = target / max(target.max(), 1e-9)

    color = np.asarray(imageio.imread(color_image), dtype=np.float64)[..., :3] / 255.0
    mask, _ = figure_mask_from_flat_ground(color)
    # Rigid stage: anchor and target are both the CANONICAL (unbent) figure.
    aligned_mask, params, iou = align_mask_to(
        target, mask, match_area=True, translation_px=12.0, companion=color
    )
    aligned_color = params["companion"]
    print(f"anchor->target rigid alignment IoU {iou:.3f}")

    # Inverse warp: for each carve-frame pixel, where in the target frame?
    # Fit on (u -> w); thin-plate smoothing keeps it tame between constraints.
    rbf = RBFInterpolator(u_px, w_px, kernel="thin_plate_spline", smoothing=1.0)
    yy, xx = np.mgrid[0:size, 0:size]
    grid = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
    src = rbf(grid)  # (N, 2) target-frame (col, row)
    src_cols = src[:, 0].reshape(size, size)
    src_rows = src[:, 1].reshape(size, size)

    warp_color = np.stack(
        [
            ndimage.map_coordinates(
                aligned_color[..., ch], [src_rows, src_cols], order=1, mode="nearest"
            )
            for ch in range(3)
        ],
        axis=-1,
    )
    warp_mask = ndimage.map_coordinates(
        aligned_mask.astype(np.float64), [src_rows, src_cols], order=1, mode="constant"
    )
    return warp_color, warp_mask


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if not all([args.CARVE, args.CORRESPONDENCE, args.TARGET, args.COLOR_IMAGE, args.OUT]):
        raise SystemExit(
            "usage: CARVE=<ckpt> CORRESPONDENCE=<npz> TARGET=<png> COLOR_IMAGE=<png> OUT=<npy>"
        )
    import torch

    state = torch.load(args.CARVE, map_location="cpu", weights_only=False)
    size = int(state["config"].get("SHAPE_RENDER_SIZE") or state["config"]["RENDER_SIZE"])

    color, mask = warped_anchor(args.CORRESPONDENCE, args.TARGET, args.COLOR_IMAGE, size)

    out = Path(args.OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    axes[0].imshow(np.clip(color, 0, 1))
    axes[0].set_title("warped anchor (carve frame)", fontsize=10)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("warped figure mask", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out.parent / "warped_anchor_evidence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    bake_texture_init(
        args.CARVE, args.COLOR_IMAGE, out, fill="flat", aligned_override=(color, mask)
    )


if __name__ == "__main__":
    main()
