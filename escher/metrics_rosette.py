"""Measure the unpainted pockets at the orbifold's rotation centres.

At each cone point, k copies of the fundamental domain pinwheel around a single
vertex. Whatever the shared atlas holds at that UV corner is therefore replicated
k times, and the render's per-tile colourisation is a DIAGONAL multiply -- so a
near-white texel comes out as the tile's palette colour at full brightness while
a dark texel stays dark. The visible consequence is a hard-edged multicoloured
polygon at every rotation centre: a 4-wedge square at each of the 6 four-fold
axes, a 6-wedge rosette at each of the 8 three-fold axes.

The forensic finding (2026-08-14) is that these pockets are NOT a geometry gap
-- the carved tile's solid angle near the corners is within +-25% of the
undeformed kite and the tiling certificate is exactly 4pi -- but flat, unpainted
background filler. Measured on the gingerbread run, a 32-texel disc around the
order-3 corner had mean luminance 0.98 (interior null 0.353 +- 0.205); on the
stocking, mean 1.0000 with standard deviation exactly 0.0000. SDS never put a
figure there, because every training view is one isolated tile composited on
white, where stopping short of the corner is a perfect score.

The two-fold centres are exempt and that is a useful control: ``cone1``'s kite
angle is 180 degrees, so it is not a corner at all, and the figure is painted
straight through it.

So the number that matters is how far the flat pocket extends:

``rosette_radius_deg``
    Per corner, the largest geodesic radius (degrees) at which the
    solid-angle-weighted mean atlas luminance stays at or above ``WHITE_LEVEL``.
    0 means the figure reaches the rotation centre -- the goal.
``pocket_solid_angle``
    Solid angle of the tile within that radius, in steradians.
``white_fraction``
    Solid-angle-weighted share of the tile whose texture is above
    ``WHITE_LEVEL`` -- how much of the tile is unpainted overall.

Validated against the runs the round-16 forensics characterised: gingerbread
4.25 deg (4-fold) / 20.25 deg (3-fold), stocking 2.25 / 10.75, and the r13
reindeer -- whose atlas is 0.3% white -- at 0 degrees on every corner.

Usage::

    python escher/metrics_rosette.py runs/r14_accel/ctrl/tex_s0/checkpoint.pt

Prints a one-line summary and writes ``rosette_metrics.json`` beside the checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

__all__ = ["measure", "rosette_metrics", "WHITE_LEVEL"]

#: Luminance at or above which a texel counts as unpainted background. The atlas
#: pockets measure 0.94-1.00 against an interior mean of 0.35-0.59, so anything
#: in 0.8-0.9 separates them cleanly; 0.85 is the midpoint of that gap.
WHITE_LEVEL = 0.85

#: Radii swept when growing the disc around each corner.
RADII_DEG = np.arange(0.25, 45.0, 0.25)


def _face_luminance(texture: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Atlas luminance at each face's UV centroid.

    Nearest-texel rather than bilinear on purpose: the pockets are flat fields,
    so interpolation cannot change the verdict, and nearest keeps the measure
    independent of the filtering the renderer happens to use.
    """
    lum = np.asarray(texture, dtype=np.float64)
    if lum.ndim == 3:
        lum = lum[:, :, 0]  # BW consumes channel 0 (main_sphere.effective_texture)
    res = lum.shape[0]
    c = np.asarray(uv)[faces].mean(axis=1)  # (n_faces, 2)
    col = np.clip((c[:, 0] * res).astype(int), 0, res - 1)
    row = np.clip((c[:, 1] * res).astype(int), 0, res - 1)
    return lum[row, col]


def rosette_metrics(
    points: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture: np.ndarray,
    corners: dict[str, int],
) -> dict:
    """Pocket size at each cone corner, from a carved tile and its atlas."""
    from escher.geometry.spherical_sanity_checks import signed_solid_angles

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces)
    areas = np.abs(signed_solid_angles(points, faces))
    lum = _face_luminance(texture, uv, faces)

    # Face centroids, projected back to the sphere so distances are geodesic.
    cent = points[faces].mean(axis=1)
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-30)

    total = float(areas.sum())
    out: dict = {
        "white_fraction": float(areas[lum >= WHITE_LEVEL].sum() / max(total, 1e-30)),
        "tile_solid_angle": total,
        "mean_luminance": float((areas * lum).sum() / max(total, 1e-30)),
    }

    for name, idx in corners.items():
        p = points[int(idx)]
        p = p / max(float(np.linalg.norm(p)), 1e-30)
        # Geodesic angle from this corner to every face centroid.
        ang = np.degrees(np.arccos(np.clip(cent @ p, -1.0, 1.0)))

        radius = 0.0
        pocket = 0.0
        for r in RADII_DEG:
            m = ang <= r
            a = areas[m].sum()
            if a <= 0:
                continue
            if float((areas[m] * lum[m]).sum() / a) >= WHITE_LEVEL:
                radius, pocket = float(r), float(a)
            else:
                break
        out[f"rosette_radius_deg_{name}"] = radius
        out[f"pocket_solid_angle_{name}"] = pocket

    keys = [k for k in out if k.startswith("rosette_radius_deg_")]
    out["rosette_radius_deg_max"] = max(out[k] for k in keys)
    out["rosette_solid_angle_total"] = sum(
        out[k.replace("rosette_radius_deg_", "pocket_solid_angle_")] for k in keys
    )
    return out


def measure(checkpoint: str | Path) -> dict:
    """Load a run, solve its carved shape, and measure its rosettes."""
    from omegaconf import OmegaConf

    from escher.main_shape import build_shape_run

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(dict(state["config"]))
    cfg.DEVICE = "cpu"
    cfg.OUTPUT_DIR = str(checkpoint.parent)
    escher = build_shape_run(cfg)
    escher.load_checkpoint(checkpoint)

    with torch.no_grad():
        points = escher.solve_points().detach().cpu().numpy()

    mesh = escher.mesh
    corners = {
        n: getattr(mesh, n)
        for n in ("cone1", "cone2a", "cone3", "cone2b")
        if hasattr(mesh, n)
    }
    if not corners:
        raise SystemExit(
            "this mesh exposes no cone corners -- rosettes are a kite-domain "
            "artifact, so there is nothing to measure on a lune"
        )

    out = rosette_metrics(
        points,
        np.asarray(mesh.faces),
        np.asarray(mesh.uv),
        state["texture"].detach().cpu().numpy(),
        corners,
    )
    out["checkpoint"] = str(checkpoint)
    out["iteration"] = int(state.get("iteration", -1))
    (checkpoint.parent / "rosette_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoint.pt> [...]")
    for ck in sys.argv[1:]:
        m = measure(ck)
        parts = " ".join(
            f"{n} {m[f'rosette_radius_deg_{n}']:5.2f}"
            for n in ("cone1", "cone2a", "cone3", "cone2b")
            if f"rosette_radius_deg_{n}" in m
        )
        print(
            f"{Path(ck).parent}: rosette(deg) {parts} | "
            f"white {m['white_fraction']:.3f} | mean {m['mean_luminance']:.3f}"
        )


if __name__ == "__main__":
    main()
