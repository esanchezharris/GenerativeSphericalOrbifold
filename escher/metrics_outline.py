"""Outline metrics for a post-joint texture checkpoint against its source carve.

The joint (GEM-parity) window lets SDS move the outline, so the final shape MAY
trade hard IoU against the carve's target for subject specificity -- that trade
is pre-registered, and this script measures it instead of letting a lower
target-IoU be misread as a regression:

- ``iou_vs_target``: final silhouette vs the carve's aligned target mask
  (guardrail: should not fall more than ~0.05 below the carve's own score).
- ``iou_vs_carve``: final silhouette vs the CARVE's silhouette -- how far the
  joint window actually moved the outline (1.0 = frozen recipe, unchanged).
- perimeter ratio + fold count of the final shape.

Usage (inside the WSL venv)::

    python escher/metrics_outline.py <texture_ckpt> <carve_ckpt>

Writes ``outline_metrics.json`` beside the texture checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from escher.geometry.spherical_sanity_checks import count_flipped_faces
from escher.main_shape import build_shape_run, make_context, soft_alpha
from escher.shape_target import hard_iou


def _silhouette(escher, ctx) -> tuple[np.ndarray, int, float]:
    with torch.no_grad():
        points = escher.solve_points()
        alpha = soft_alpha(escher, points, ctx)[0, ..., 0].cpu().numpy()
    flips = count_flipped_faces(points.detach().cpu().numpy(), escher.mesh.faces)
    return alpha, flips, escher.boundary_ratio(points)


def measure(texture_ckpt: str | Path, carve_ckpt: str | Path) -> dict:
    texture_ckpt, carve_ckpt = Path(texture_ckpt), Path(carve_ckpt)

    # Evaluate BOTH shapes in the carve's own frame so the numbers are
    # comparable: build from the carve's embedded config on CPU.
    state = torch.load(carve_ckpt, map_location="cpu", weights_only=False)
    args = OmegaConf.create(state["config"])
    args.DEVICE = "cpu"
    args.OUTPUT_DIR = str(texture_ckpt.parent / "outline_tmp")
    escher = build_shape_run(args)
    ctx = make_context(escher, args)

    escher.load_checkpoint(carve_ckpt, reset_texture=True)
    carve_alpha, carve_flips, carve_perim = _silhouette(escher, ctx)

    escher.load_checkpoint(texture_ckpt, reset_texture=True)
    final_alpha, final_flips, final_perim = _silhouette(escher, ctx)

    target_png = carve_ckpt.parent / "target_aligned.png"
    target = imageio.imread(target_png).astype(np.float32)
    target = (target / max(target.max(), 1e-9) > 0.5).astype(np.float32)

    result = {
        "texture_checkpoint": str(texture_ckpt),
        "carve_checkpoint": str(carve_ckpt),
        "iou_vs_target": hard_iou(final_alpha, target),
        "carve_iou_vs_target": hard_iou(carve_alpha, target),
        "iou_vs_carve": hard_iou(final_alpha, carve_alpha),
        "final_perim": final_perim,
        "carve_perim": carve_perim,
        "final_flips": int(final_flips),
        "carve_flips": int(carve_flips),
    }
    out = texture_ckpt.parent / "outline_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"outline: vs-target {result['iou_vs_target']:.4f} "
        f"(carve was {result['carve_iou_vs_target']:.4f}) | "
        f"drift vs carve {result['iou_vs_carve']:.4f} | "
        f"perim {final_perim:.4f} (carve {carve_perim:.4f}) | "
        f"flips {final_flips} -> {out}"
    )
    return result


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <texture_ckpt> <carve_ckpt>")
    result = measure(sys.argv[1], sys.argv[2])
    raise SystemExit(0 if result["final_flips"] == 0 else 3)


if __name__ == "__main__":
    main()
