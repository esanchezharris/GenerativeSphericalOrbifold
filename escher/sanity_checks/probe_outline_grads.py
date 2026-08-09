"""Do outline (vertex/W) gradients reach the shape through the isolated render?

The 2026-08-06 probe that froze the sphere's shape during SDS ("the rasterized
alpha carries NO vertex grads") was run at ISOLATED_DISTANCE 1.7, where the lone
tile OVERFILLS the frame: with no background pixels there is no silhouette
discontinuity, and ``dr.antialias`` -- the ONLY source of vertex-position
gradients in this renderer (alpha is a clamped triangle-ID) -- has nothing to
differentiate. This probe re-measures across distances, with the background
fraction and frame margin printed beside each gradient norm so framing and
gradient can never be conflated again.

Usage (inside the WSL venv, GPU)::

    python escher/sanity_checks/probe_outline_grads.py \
        CHECKPOINT=output/sphere_shape_fish2/checkpoint.pt

Exit code 0 iff, at the LARGEST probed distance, the frame shows real background
(> 1%) AND the W-gradient through render + implicit solve is nonzero -- the gate
for the joint shape+texture (GEM-parity) work. The smallest distance is reported
but not asserted: ~0 there is the expected historical artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from escher.main_shape import build_shape_run
from escher.rendering.camera import perspective, tile_centric_views
from escher.soft_silhouette import boundary_loop, project_to_pixels

DEFAULTS = OmegaConf.create(
    {
        "CHECKPOINT": "output/sphere_shape_fish2/checkpoint.pt",
        "DISTANCES": [1.7, 2.0, 2.4],
        "DEVICE": "cuda",
    }
)


def probe(checkpoint: str | Path, distances, device: str = "cuda") -> bool:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = OmegaConf.create(state["config"])
    args.DEVICE = device
    args.OUTPUT_DIR = str(Path(checkpoint).parent / "probe_tmp")

    escher = build_shape_run(args)
    escher.load_checkpoint(checkpoint, reset_texture=True)
    if args.PARAM_MODE != "weights":
        raise SystemExit("probe expects a weights-mode carve checkpoint")

    loop = torch.as_tensor(boundary_loop(escher.mesh), dtype=torch.long)
    centers = torch.as_tensor(
        escher.solo_tiler.tile_centers(escher.mesh.points), dtype=torch.float32
    )
    fov = float(args.CAMERA_FOV)
    size = int(args.RENDER_SIZE)
    flat = torch.full((8, 8, 3), 0.9, device=escher.device)

    print(f"probing {checkpoint} | fov {fov} | frame {size}px")
    results = []
    for d in distances:
        mv = tile_centric_views(centers, 1, distance=float(d), angular_jitter_deg=0.0)

        points = escher.solve_points()
        images, alpha, _ = escher.render(
            1, mv=mv, isolated=True, texture=flat, points=points
        )
        composite = images * alpha  # black background: maximal contrast
        loss = composite.mean()
        g_points, g_w = torch.autograd.grad(loss, [points, escher.W])

        bg_frac = float((alpha < 0.5).float().mean())
        with torch.no_grad():
            px = project_to_pixels(
                points.detach()[loop], mv, perspective(fovy_deg=fov), size, size
            )
            margin = float(min(px.min(), size - 1 - px.max()))

        results.append(
            {
                "d": float(d),
                "g_points": float(g_points.norm()),
                "g_w": float(g_w.norm()),
                "bg_frac": bg_frac,
                "margin_px": margin,
            }
        )
        print(
            f"  d={d:4.2f} | dL/dpoints {results[-1]['g_points']:.3e} | "
            f"dL/dW {results[-1]['g_w']:.3e} | background {100 * bg_frac:5.1f}% | "
            f"margin {margin:+7.1f} px"
        )

    far = results[-1]
    ok = far["bg_frac"] > 0.01 and far["g_w"] > 1e-12
    if ok:
        print(
            f"\nPASS: at d={far['d']} the outline gradient is real "
            f"(dL/dW {far['g_w']:.3e} with {100 * far['bg_frac']:.1f}% background) "
            "-- joint shape+texture SDS is unlocked."
        )
    else:
        print(
            f"\nFAIL: at d={far['d']} bg {100 * far['bg_frac']:.1f}%, "
            f"dL/dW {far['g_w']:.3e} -- outline gradients do NOT reach W; "
            "the joint window premise does not hold at this framing."
        )
    return ok


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    ok = probe(args.CHECKPOINT, list(args.DISTANCES), device=str(args.DEVICE))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
