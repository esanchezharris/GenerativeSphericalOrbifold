r"""Closed-form Escherization: the globally optimal REACHABLE tile outline.

Twenty years of Escherization literature (Kaplan 2000 -> Nagata-Imahori 2020)
says: score local differential structure, not area, and search with a solver
that cannot plateau. Our pinned cones remove the scale/rotation ambiguity that
forced that literature's Procrustes eigenproblems, so the whole fit collapses
to ONE linear solve:

    minimize_xi  (u - w)^T (S x I3) (u - w)      u = M xi + c

- ``u``: the full 160-vertex boundary loop, affine in the free coordinates
  ``xi`` (interiors of the two left cut chains; right chains are the R1/R2
  rotation images -- LINEAR in R^3; the four cones are pinned constants).
- ``w``: the target silhouette's contour, resampled to the loop's vertex count
  and unprojected onto the unit sphere through the carve camera.
- ``S``: an AD/GAD-style metric graph -- consecutive differences plus
  multi-scale chords, node-weighted by limb thinness (thin structures pay
  more; the exact weighting the area-IoU carve lacks).

The global optimum over the reachable family is a Cholesky solve; the
correspondence's discrete structure (cyclic offset x orientation) is searched
exhaustively with one factorization and cheap RHS re-solves. The result is
rasterized to ``escherized_target.npy`` -- a carve target that is REACHABLE BY
CONSTRUCTION -- plus the optimal boundary itself and evidence imagery.

Usage (inside the WSL venv)::

    python escher/escherize.py CARVE=<carve>/checkpoint.pt \
        TARGET=<carve>/target_aligned.png OUT_DIR=runs/r8/escherize_fish
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from omegaconf import OmegaConf

from escher.main_shape import build_shape_run, make_context
from escher.pixel_solid_angle import _as_matrix, _sphere_hits
from escher.shape_target import hard_iou, limb_thinness_weights
from escher.soft_silhouette import boundary_loop, project_to_pixels

DEFAULTS = OmegaConf.create(
    {
        "CARVE": "",
        "TARGET": "",
        "OUT_DIR": "",
        "CHORDS": [2, 3, 5, 8],  # multi-scale GAD chords, in loop steps
        "LIMB_WEIGHT_MAX": 5.0,
        "POSITION_EPS": 0.05,    # small absolute-position term for conditioning
    }
)


# --------------------------------------------------------------------- structure
def boundary_affine_map(mesh, R1: np.ndarray, R2: np.ndarray):
    """``u = M xi + c`` over the ordered boundary loop.

    Returns ``loop`` (ordered vertex ids), dense ``M (3L, 3F)``, ``c (3L,)``,
    and ``free_ids`` (the loop's free-parameter vertex ids, xi order).
    """
    loop = boundary_loop(mesh)
    L = len(loop)

    free_l1 = list(mesh.left1[1:-1])
    free_l2 = list(mesh.left2[1:-1])
    free_ids = free_l1 + free_l2
    F = len(free_ids)
    slot = {v: j for j, v in enumerate(free_ids)}

    gen = {}
    for i, v in enumerate(mesh.right1[1:-1]):
        gen[int(v)] = (slot[free_l1[i]], R1)
    for i, v in enumerate(mesh.right2[1:-1]):
        gen[int(v)] = (slot[free_l2[i]], R2)
    pinned = {int(getattr(mesh, k)) for k in ("cone1", "cone2a", "cone2b", "cone3")}

    M = np.zeros((3 * L, 3 * F))
    c = np.zeros(3 * L)
    cone_loop_idx = []
    for i, v in enumerate(loop):
        v = int(v)
        r = slice(3 * i, 3 * i + 3)
        if v in pinned:
            c[r] = mesh.points[v]
            cone_loop_idx.append(i)
        elif v in slot:
            M[r, 3 * slot[v] : 3 * slot[v] + 3] = np.eye(3)
        elif v in gen:
            j, R = gen[v]
            M[r, 3 * j : 3 * j + 3] = R
        else:
            raise ValueError(f"boundary vertex {v} has no role (loop index {i})")
    return loop, M, c, free_ids, cone_loop_idx


# ----------------------------------------------------------------------- target
def target_contour(mask: np.ndarray, n: int) -> np.ndarray:
    """Longest closed 0.5-contour of ``mask``, resampled to ``n`` points by arc
    length. Returns ``(n, 2)`` as (col, row) pixel coordinates."""
    # Pad with a zero border first: production targets run to the frame edge,
    # which breaks the level-set into open segments (measured: the sampler
    # caught one fin arc and the fit followed the garbage). Padding closes it.
    pad = 2
    padded = np.pad(mask, pad, mode="constant")
    fig = plt.figure()
    cs = plt.contour(padded, levels=[0.5])
    try:
        segs = [s for level in cs.allsegs for s in level]
    finally:
        plt.close(fig)
    if not segs:
        raise ValueError("no 0.5-contour found in the target mask")
    seg = max(segs, key=lambda s: len(s)) - pad  # (m, 2) in (x=col, y=row)

    d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    if t[-1] <= 0:
        raise ValueError("degenerate target contour")
    ti = np.linspace(0.0, t[-1], n, endpoint=False)
    out = np.stack(
        [np.interp(ti, t, seg[:, 0]), np.interp(ti, t, seg[:, 1])], axis=1
    )
    return out


def unproject(px: np.ndarray, mv, proj, size: int) -> np.ndarray:
    """Pixel (col, row) -> unit-sphere points through the carve camera."""
    ndc = np.stack(
        [2.0 * px[:, 0] / size - 1.0, -(2.0 * px[:, 1] / size - 1.0)], axis=1
    )
    pm_inv = np.linalg.inv(_as_matrix(proj) @ _as_matrix(mv))
    pts, hit = _sphere_hits(pm_inv, ndc)
    if not hit.all():
        # Points marginally off the disc: pull their ray to the horizon circle.
        miss = ~hit
        pts[miss] /= np.linalg.norm(pts[miss], axis=1, keepdims=True).clip(1e-9)
    return pts


# ------------------------------------------------------------------------ metric
def metric_graph(n: int, node_w: np.ndarray, chords: list[int], eps: float):
    """``S (n, n)``: weighted graph Laplacian over consecutive + chord edges,
    plus ``eps * diag(node_w)`` absolute-position term."""
    S = np.zeros((n, n))
    steps = [1] + [int(j) for j in chords]
    for j in steps:
        for i in range(n):
            a, b = i, (i + j) % n
            k = 0.5 * (node_w[a] + node_w[b]) / j  # longer chords count less
            S[a, a] += k
            S[b, b] += k
            S[a, b] -= k
            S[b, a] -= k
    S += eps * np.diag(node_w)
    return S


def expand3(S: np.ndarray) -> np.ndarray:
    return np.kron(S, np.eye(3))


# ------------------------------------------------------------------------- solve
def escherize(args) -> dict:
    out_dir = Path(args.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.CARVE, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    cfg.DEVICE = "cpu"
    cfg.OUTPUT_DIR = str(out_dir / "build_tmp")
    escher = build_shape_run(cfg)
    ctx = make_context(escher, cfg)
    mesh = escher.mesh

    R1, R2 = escher.orbifold.R1, escher.orbifold.R2
    loop, M, c, free_ids, cone_loop_idx = boundary_affine_map(mesh, R1, R2)
    L = len(loop)

    import imageio.v2 as imageio

    mask = np.asarray(imageio.imread(args.TARGET), dtype=np.float64)
    mask = mask / max(mask.max(), 1e-9)

    contour_px = target_contour(mask, L)
    w_sphere = unproject(contour_px, ctx.mv, ctx.proj, ctx.size)

    limb_map = limb_thinness_weights(mask, float(args.LIMB_WEIGHT_MAX))
    rows = np.clip(contour_px[:, 1].round().astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(contour_px[:, 0].round().astype(int), 0, mask.shape[1] - 1)
    node_w = limb_map[rows, cols]

    S3 = expand3(
        metric_graph(L, node_w, list(args.CHORDS), float(args.POSITION_EPS))
    )
    A = M.T @ S3 @ M
    A += 1e-9 * np.trace(A) / A.shape[0] * np.eye(A.shape[0])
    chol = np.linalg.cholesky(A)
    MtS = M.T @ S3

    def solve_for(w_stack: np.ndarray) -> tuple[np.ndarray, float]:
        rhs = MtS @ (w_stack - c)
        xi = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        u = M @ xi + c
        r = u - w_stack
        return u, float(r @ (S3 @ r))

    # CONE-ANCHORED PIECEWISE CORRESPONDENCE (the literature's decisive discrete
    # structure, Nagata-Imahori 2020): the four pinned cones cannot move, so one
    # global cyclic offset can never satisfy all four anchors. Instead: anchor
    # each cone's LOOP position to the contour point nearest its pin, and
    # resample each chain uniformly BETWEEN its anchors, both orientations.
    pins3 = c.reshape(L, 3)[cone_loop_idx]

    def piecewise_w(w_pts: np.ndarray) -> np.ndarray | None:
        n = len(w_pts)
        anchors = [int(np.argmin(np.linalg.norm(w_pts - p, axis=1))) for p in pins3]
        # cyclic monotonicity: walking the cones in LOOP order must wrap the
        # contour exactly once; distinct anchors required.
        k_ = len(anchors)
        diffs = [(anchors[(j + 1) % k_] - anchors[j]) % n for j in range(k_)]
        if 0 in diffs or sum(diffs) != n:
            return None
        w_out = np.zeros((L, 3))
        src = np.zeros(L, dtype=int)
        k = len(cone_loop_idx)
        for a in range(k):
            i0, i1 = cone_loop_idx[a], cone_loop_idx[(a + 1) % k]
            seg_len = (i1 - i0) % L or L
            t0, t1 = anchors[a], anchors[(a + 1) % k]
            span = (t1 - t0) % n or n
            for s in range(seg_len):
                j = (t0 + round(s / seg_len * span)) % n
                w_out[(i0 + s) % L] = w_pts[j]
                src[(i0 + s) % L] = j
        return w_out, src

    best = None
    for orient in (1, -1):
        pw = piecewise_w(w_sphere[::orient])
        if pw is None:
            continue
        w_o, src_o = pw
        u, obj = solve_for(w_o.reshape(-1))
        if best is None or obj < best[0]:
            best = (obj, orient, w_o, src_o)
    if best is None:
        raise ValueError("no orientation gives cyclically consistent cone anchors")
    obj, orient, w_best, src_best = best
    off = -1  # correspondence is anchor-derived, not a global offset
    u_best, _ = solve_for(w_best.reshape(-1))
    # Pixel-space correspondence: target-frame contour pixel per loop node.
    w_px_best = contour_px[::orient][src_best]

    u_pts = u_best.reshape(L, 3)
    u_pts /= np.linalg.norm(u_pts, axis=1, keepdims=True).clip(1e-9)

    # Rasterize the optimal outline in the carve frame -> the reachable target.
    px = (
        project_to_pixels(
            torch.as_tensor(u_pts, dtype=torch.float64), ctx.mv, ctx.proj, ctx.size, ctx.size
        )
        .cpu()
        .numpy()
    )
    yy, xx = np.mgrid[0 : ctx.size, 0 : ctx.size]
    inside = MplPath(px).contains_points(
        np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
    )
    esch_mask = inside.reshape(ctx.size, ctx.size).astype(np.float32)

    ceiling = hard_iou(esch_mask, (mask > 0.5).astype(np.float32))
    np.save(out_dir / "escherized_target.npy", esch_mask)
    # The boundary <-> contour correspondence IS the non-rigid registration the
    # flat bake needs to dress an articulated outline (phase 2: warped bake).
    np.savez(
        out_dir / "correspondence.npz", u_px=px, w_px=w_px_best, loop=loop
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    axes[0].imshow(mask, cmap="gray")
    axes[0].plot(contour_px[:, 0], contour_px[:, 1], "r.", ms=2)
    axes[0].set_title("target + sampled contour", fontsize=10)
    axes[1].imshow(mask, cmap="gray")
    axes[1].plot(
        np.append(px[:, 0], px[0, 0]), np.append(px[:, 1], px[0, 1]), "c-", lw=1.5
    )
    axes[1].set_title(f"optimal reachable outline (offset {off}, orient {orient})", fontsize=10)
    overlay = np.stack([(mask > 0.5).astype(float), esch_mask, np.zeros_like(esch_mask)], axis=-1)
    axes[2].imshow(overlay)
    axes[2].set_title(f"escherized mask vs target | hard IoU {ceiling:.4f}", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_dir / "escherize_evidence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    result = {
        "objective": obj,
        "orientation": int(orient),
        "offset": int(off),
        "ceiling_hard_iou_vs_target": ceiling,
        "n_free": len(free_ids),
        "escherized_target": str(out_dir / "escherized_target.npy"),
    }
    (out_dir / "escherize_result.json").write_text(json.dumps(result, indent=2))
    print(
        f"escherized: ceiling hard IoU {ceiling:.4f} vs target "
        f"(orient {orient}, offset {off}) -> {out_dir}"
    )
    return result


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if not (args.CARVE and args.TARGET and args.OUT_DIR):
        raise SystemExit("usage: CARVE=<ckpt> TARGET=<aligned png> OUT_DIR=<dir>")
    escherize(args)


if __name__ == "__main__":
    main()
