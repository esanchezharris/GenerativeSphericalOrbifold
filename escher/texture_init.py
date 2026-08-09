"""Image-anchored texture init: bake an aligned color figure into the texture.

The dominant source of run-to-run texture variance is BASIN SELECTION -- which
composition and palette early SDS noise happens to fall into (measured across
rounds 2-3: the same recipe produced a rich nested-fish texture at one seed and
a pale single-fish at another). This bake removes that dice roll: a color
figure image (generated once by the same SD model, ``make_target.py COLOR=true``)
is aligned over the CARVED tile with the exact machinery the carve target used,
warped into texture space through the mesh's own UV mapping, and written as the
texture init. SDS then refines a fish-colored, fish-composed texture instead of
choosing a basin by chance.

Usage (inside the WSL venv)::

    python escher/texture_init.py CARVE=<carve>/checkpoint.pt \
        COLOR_IMAGE=assets/targets/fish64/color.png OUT=<run>/texture_init.npy

Writes the init npy + ``texture_init_evidence.png`` (baked texture, aligned
color, mask overlay) beside it. Eyeball the evidence BEFORE spending GPU hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage

from escher.main_shape import build_shape_run, make_context, soft_alpha
from escher.rendering.texture_mask import gutter_fill, texel_surface_points, uv_valid_mask
from escher.shape_target import align_mask_to, figure_mask_from_flat_ground
from escher.soft_silhouette import project_to_pixels

DEFAULTS = OmegaConf.create(
    {"CARVE": "", "COLOR_IMAGE": "", "OUT": "", "DEVICE": "cpu"}
)


def bake_texture_init(
    carve_ckpt: str | Path, color_image: str | Path, out_path: str | Path
) -> Path:
    carve_ckpt, out_path = Path(carve_ckpt), Path(out_path)
    state = torch.load(carve_ckpt, map_location="cpu", weights_only=False)
    args = OmegaConf.create(state["config"])
    args.DEVICE = "cpu"
    args.OUTPUT_DIR = str(out_path.parent / "bake_tmp")
    escher = build_shape_run(args)
    escher.load_checkpoint(carve_ckpt, reset_texture=True)
    ctx = make_context(escher, args)

    with torch.no_grad():
        points = escher.solve_points()
        alpha = soft_alpha(escher, points, ctx)[0, ..., 0].cpu().numpy()

    color = np.asarray(imageio.imread(color_image), dtype=np.float64)[..., :3] / 255.0
    mask, ground_std = figure_mask_from_flat_ground(color)
    print(f"color figure area {mask.mean():.3f}, ground spread {ground_std:.3f}")

    # The same alignment the carve target got (including the carve's corner /
    # translation knobs), with the color image riding the winning transform.
    aligned_mask, params, iou = align_mask_to(
        alpha,
        mask,
        match_area=bool(args.get("MATCH_TILE_AREA", True)),
        corner_px=None,
        corner_weight=0.0,
        translation_px=float(args.get("ALIGN_TRANSLATION_SEARCH_PX", 0.0)),
        companion=color,
    )
    aligned_color = params["companion"]
    print(
        f"color aligned: scale {params['scale']:.3f}, angle "
        f"{params['angle_deg']:+.1f} deg, IoU vs carved tile {iou:.3f}"
    )

    # Texel -> carved 3D surface point -> shape-camera pixel -> color sample.
    res = int(args.TEXTURE_RESOLUTION)
    surface, covered = texel_surface_points(
        points.detach().cpu().numpy(), escher.mesh.uv, escher.mesh.faces, res
    )
    texture = np.ones((res, res, 3), dtype=np.float32)
    figure_texel = np.zeros((res, res), dtype=bool)

    pts = torch.as_tensor(surface[covered], dtype=torch.float64)
    with torch.no_grad():
        px = project_to_pixels(pts, ctx.mv, ctx.proj, ctx.size, ctx.size).numpy()
    cols, rows = px[:, 0], px[:, 1]

    for c in range(3):
        texture[covered, c] = ndimage.map_coordinates(
            aligned_color[..., c], [rows, cols], order=1, mode="nearest"
        )
    on_figure = (
        ndimage.map_coordinates(
            aligned_mask.astype(np.float64), [rows, cols], order=1, mode="constant"
        )
        > 0.5
    )
    figure_texel[covered] = on_figure

    # Texels off the figure (the carve residual) and unsampled gutter texels
    # both take their nearest FIGURE color -- the whole tile starts fish-colored.
    if figure_texel.any():
        texture = gutter_fill(texture, figure_texel)
    valid = uv_valid_mask(escher.mesh.uv, escher.mesh.faces, res)
    texture = gutter_fill(texture, valid | figure_texel)
    texture = np.clip(texture, 0.0, 1.0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, texture)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(texture)
    axes[0].set_title("baked texture init", fontsize=10)
    axes[1].imshow(aligned_color)
    axes[1].set_title(f"aligned color (IoU {iou:.3f})", fontsize=10)
    overlay = np.stack([aligned_mask, alpha, np.zeros_like(alpha)], axis=-1)
    axes[2].imshow(overlay)
    axes[2].set_title("mask (R) vs carved tile (G)", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    evidence = out_path.with_name("texture_init_evidence.png")
    fig.savefig(evidence, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} + {evidence}")
    return out_path


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if not (args.CARVE and args.COLOR_IMAGE):
        raise SystemExit("usage: CARVE=<ckpt> COLOR_IMAGE=<png> [OUT=<npy>]")
    out = args.OUT or str(Path(args.CARVE).parent / "texture_init.npy")
    bake_texture_init(args.CARVE, args.COLOR_IMAGE, out)


if __name__ == "__main__":
    main()
