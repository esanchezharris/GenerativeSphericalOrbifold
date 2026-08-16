"""Per-tile hue rotation: the alternating-color Escher look, at render time only.

In the target style every figure is the SAME figure with its hue rotated (blue / pink /
orange), and neighboring figures never share a color -- that alternation is what makes
the interlocking legible. All tiles here sample one shared texture, so the recolor is a
per-tile 3x3 matrix applied to the sampled color: a rotation about the RGB gray axis,
which preserves luminance and turns one gingerbread into the palette's variants.

Training never uses this (each tile view would push a different hue into the one shared
texture and the gradients would fight to gray); it is applied in previews and final
renders via ``render_tiled_sphere(tile_color_matrices=...)``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "hue_rotation_matrix",
    "tile_adjacency",
    "cone_pinwheels",
    "assign_palette_indices",
    "tile_color_matrices",
    "colorize_matrices",
    "apply_tile_color",
    "COLOR_MODES",
    "PASTEL_PALETTE",
    "XMAS_PALETTE",
    "XMAS4_PALETTE",
    "XMAS_WHITE_PALETTE",
]

# Colorization palettes (RGB scalers). Pastel = the fake-sphere family;
# XMAS = red / green / gold for the ornament theme; XMAS4 adds frost blue so the
# four tiles at every 4-fold rotation centre can all differ.
PASTEL_PALETTE = [[1.00, 0.62, 0.66], [1.00, 0.80, 0.52], [0.58, 0.76, 1.00]]
XMAS_PALETTE = [[0.86, 0.30, 0.30], [0.38, 0.64, 0.40], [0.93, 0.78, 0.42]]
XMAS4_PALETTE = [
    [0.86, 0.30, 0.30],
    [0.38, 0.64, 0.40],
    [0.93, 0.78, 0.42],
    [0.58, 0.73, 0.88],
]
# XMAS_WHITE: red / green / gold plus identity white. Under COLORIZE_MODE
# "figure" a white tile keeps its greyscale figure on white ground (reads as
# iced/snow figures joining the alternation). CAVEAT (pinned by test): in "ink"
# mode the white entry maps the whole tile to white (I - I = 0) and the figure
# vanishes -- pair white with "figure" mode.
XMAS_WHITE_PALETTE = [
    [0.86, 0.30, 0.30],
    [0.38, 0.64, 0.40],
    [0.93, 0.78, 0.42],
    [1.00, 1.00, 1.00],
]


def hue_rotation_matrix(degrees: float) -> torch.Tensor:
    """(3, 3) rotation of RGB space about the gray axis (1,1,1)/sqrt(3).

    Rodrigues' formula; 0 deg is the identity and +120 deg cyclically permutes
    R -> G -> B -> R, so a three-hue palette at 0/+120/-120 costs no saturation.
    """
    theta = np.deg2rad(degrees)
    k = np.ones(3) / np.sqrt(3.0)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) * np.cos(theta) + np.sin(theta) * K + (1 - np.cos(theta)) * np.outer(k, k)
    return torch.as_tensor(R, dtype=torch.float32)


def _boundary_position_tiles(tiler, mesh) -> dict[tuple, set[int]]:
    """For each rounded boundary-point position, the set of tiles whose image lands there."""
    boundary = np.concatenate(list(mesh.boundary_chains))
    pts = mesh.points[boundary]

    seen: dict[tuple, set[int]] = {}
    for g, rot in enumerate(tiler.rotations):
        for p in pts @ rot.T:
            seen.setdefault(tuple(np.round(p, 6)), set()).add(g)
    return seen


def tile_adjacency(tiler, mesh) -> list[tuple[int, int]]:
    """Tile pairs that share a whole boundary chain on the UNDEFORMED tiling.

    Two tiles are adjacent iff their vertex images coincide at >= 2 positions: chains
    (cut mates, the shared bottom arc) give whole shared curves, while cone points --
    the pole is one vertex on every rotation copy -- give a single position and do not
    count. Robust for any k because it reads the group itself, not an assumed layout.
    """
    counts: dict[tuple[int, int], int] = {}
    for tiles in _boundary_position_tiles(tiler, mesh).values():
        ordered = sorted(tiles)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    return sorted(pair for pair, n in counts.items() if n >= 2)


def cone_pinwheels(tiler, mesh) -> list[list[int]]:
    """Tile sets that pinwheel around each rotation centre (>= 3 wedges).

    A rotation centre is a single boundary position shared by three or more tile
    images: on the (2,3,4) kite tiling the 6 four-fold centres carry 4 wedges and
    the 8 three-fold centres carry 6 (alternating cone2a/cone2b corners of six
    tiles); on a dihedral tiling the poles carry k. Two-fold centres carry only
    2 wedges and are excluded -- the figure paints straight through them. These
    are exactly the points where the tiling "meets", so a good palette wants
    maximal color diversity inside each set.
    """
    wheels = {
        tuple(sorted(tiles))
        for tiles in _boundary_position_tiles(tiler, mesh).values()
        if len(tiles) >= 3
    }
    return [list(w) for w in sorted(wheels)]


def assign_palette_indices(tiler, mesh, n_colors: int) -> np.ndarray:
    """A color index per group element such that adjacent tiles differ, deterministically.

    Properness alone is not enough: the (2,3,4) graph is bipartite, so a first-found
    proper 3-coloring ships class sizes like [9, 10, 5] -- and a 4th palette entry
    would go entirely UNUSED. So among proper colorings this prefers (a) balanced
    classes (each color used at most ceil(G/n) times, which forces every color into
    play) and (b) maximal color diversity around the rotation centres
    (:func:`cone_pinwheels`) via a deterministic hill-climb -- with four colors the
    four tiles at a 4-fold centre can all differ, turning the tiling's meeting
    points into a feature instead of a repeat. If ``n_colors`` cannot properly
    color the graph at all (odd-k antiprisms other than k=3 need 4), fall back to
    greedy least-conflict so rendering still works.
    """
    G = tiler.order
    adj = [[] for _ in range(G)]
    for a, b in tile_adjacency(tiler, mesh):
        adj[a].append(b)
        adj[b].append(a)

    cap = -(-G // n_colors)  # balanced ceiling; relaxed to G if infeasible

    def backtrack(limit: int) -> np.ndarray | None:
        colors = np.full(G, -1, dtype=np.int64)
        counts = [0] * n_colors

        def bt(i: int) -> bool:
            if i == G:
                return True
            for c in range(n_colors):
                if counts[c] < limit and all(colors[nb] != c for nb in adj[i]):
                    colors[i] = c
                    counts[c] += 1
                    if bt(i + 1):
                        return True
                    colors[i] = -1
                    counts[c] -= 1
            return False

        return colors if bt(0) else None

    colors = backtrack(cap)
    balanced = colors is not None
    if colors is None:
        colors = backtrack(G)
    if colors is None:
        colors = np.full(G, -1, dtype=np.int64)
        for i in range(G):  # fallback: minimize conflicts, never fail
            used = [colors[nb] for nb in adj[i] if colors[nb] >= 0]
            colors[i] = min(range(n_colors), key=lambda c: used.count(c))
        return colors

    wheels = cone_pinwheels(tiler, mesh)
    if not wheels:
        return colors

    def score(cs: np.ndarray) -> int:
        return sum(len({int(cs[t]) for t in w}) for w in wheels)

    # Hill-climb over proper (and, when feasible, balance-capped) colorings:
    # single recolors change class sizes within the cap, swaps preserve them.
    best = score(colors)
    improved = True
    while improved:
        improved = False
        counts = np.bincount(colors, minlength=n_colors)
        for i in range(G):
            for c in range(n_colors):
                if c == colors[i] or any(colors[nb] == c for nb in adj[i]):
                    continue
                if balanced and counts[c] + 1 > cap:
                    continue
                trial = colors.copy()
                trial[i] = c
                s = score(trial)
                if s > best:
                    colors, best, improved = trial, s, True
                    counts = np.bincount(colors, minlength=n_colors)
        for i in range(G):
            for j in range(i + 1, G):
                ci, cj = int(colors[i]), int(colors[j])
                if ci == cj:
                    continue
                if any(colors[nb] == cj for nb in adj[i] if nb != j):
                    continue
                if any(colors[nb] == ci for nb in adj[j] if nb != i):
                    continue
                trial = colors.copy()
                trial[i], trial[j] = cj, ci
                s = score(trial)
                if s > best:
                    colors, best, improved = trial, s, True
    return colors


def tile_color_matrices(tiler, mesh, hues_deg) -> torch.Tensor:
    """(G, 3, 3) per-tile color matrices from a palette of hue angles."""
    indices = assign_palette_indices(tiler, mesh, len(hues_deg))
    return torch.stack([hue_rotation_matrix(float(hues_deg[i])) for i in indices])


#: Per-pixel colour-transfer modes for the tiled renderer.
COLOR_MODES = ("flat", "figure", "ink")


def apply_tile_color(
    col: torch.Tensor,
    mtx: torch.Tensor,
    mode: str = "flat",
    gate: tuple[float, float] = (0.65, 0.85),
) -> torch.Tensor:
    """Apply per-tile colour matrices to sampled texture colours, per pixel.

    ``col`` is ``(..., 3)`` sampled colours; ``mtx`` is ``(..., 3, 3)`` the
    interpolated per-pixel tile matrices (constant across each tile's pixels).

    ``"flat"`` is the historical behaviour: the matrix applied everywhere. Its
    defect is that a DIAGONAL palette matrix maps white texels to the tile
    colour at full brightness, so unpainted ground renders as a saturated
    field whose seams are the tile boundaries, and the corner pockets become
    hard-edged colour polygons.

    ``"figure"`` gates the tint by sampled luminance: full tint at or below
    ``gate[0]``, none at or above ``gate[1]`` (smoothstep between). Near-white
    ground then stays white on EVERY tile -- tile boundaries through ground
    vanish and colour changes land only on figure paint, which is how Escher's
    own coloured tilings hide their seams. ``gate[1]`` defaults to the repo's
    WHITE_LEVEL 0.85 unpainted convention; valid because the tint runs BEFORE
    shading, so luminance here is texture-space.

    ``"ink"`` is the white-fixed-point affine ``1 - (I - M)(1 - col)``: white
    stays white, black paint takes the tile colour. Being AFFINE in ``col`` it
    commutes exactly with mip filtering and interpolation -- no edge halo by
    construction -- at the cost of re-inking the figure instead of preserving
    its luminance. Note ``M = I`` (a white palette entry) maps the whole tile
    to white in this mode.
    """
    flat = torch.einsum("...ij,...j->...i", mtx, col)
    if mode == "flat":
        return flat
    if mode == "figure":
        lo, hi = float(gate[0]), float(gate[1])
        lum = col.mean(dim=-1, keepdim=True)
        t = ((lum - lo) / max(hi - lo, 1e-8)).clamp(0.0, 1.0)
        w = 1.0 - t * t * (3.0 - 2.0 * t)
        return col + w * (flat - col)
    if mode == "ink":
        # 1 - (I - M)(1 - col) simplified: white (col=1) is a fixed point,
        # black maps to M @ 1 = the palette colour.
        return col + torch.einsum("...ij,...j->...i", mtx, 1.0 - col)
    raise ValueError(f"unknown color mode {mode!r}; expected one of {COLOR_MODES}")


def colorize_matrices(tiler, mesh, palette) -> torch.Tensor:
    """(G, 3, 3) per-tile DIAGONAL color scalers from an RGB palette.

    The hue rotation preserves luminance but has an achromatic fixed point:
    greyscale (BW) textures pass through it unchanged. Colorization multiplies
    luminance by a tile color instead -- the correct alternation for greyscale
    figures. White areas take the tile color at full brightness; blacks stay
    black, so line work survives.
    """
    indices = assign_palette_indices(tiler, mesh, len(palette))
    mats = [
        torch.diag(torch.as_tensor(np.asarray(palette[i], dtype=np.float32)))
        for i in indices
    ]
    return torch.stack(mats)
