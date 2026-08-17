"""The neighbour-context render must add surroundings without moving the figure.

The isolated pass grades one tile, but composited on flat background it also
teaches SDS that anything the figure does not cover is acceptable -- which is
how the unpainted rosettes at the rotation centres survive 7000 steps. Adding
the ring of group images puts that consequence in the picture. What must NOT
change is the framing: the camera and the crop still follow the centre tile.
"""

from __future__ import annotations

import numpy as np
import torch

from escher.main_shape import build_shape_run

from .test_main_shape import tiny_args


def escher_with(tmp_path, **over):
    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    for k, v in over.items():
        setattr(args, k, v)
    return build_shape_run(args)


def test_default_is_the_lone_tile(tmp_path):
    esc = escher_with(tmp_path)
    assert int(esc.args.get("ISOLATED_NEIGHBOURS", 0) or 0) == 0
    assert len(esc.solo_tiler.rotations) == 1


def test_identity_is_element_zero(tmp_path):
    """Callers rely on tile 0 being the fundamental domain."""
    esc = escher_with(tmp_path)
    t = esc.neighbour_tiler(4)
    assert np.allclose(t.rotations[0], np.eye(3))


def test_requested_count_is_honoured_and_capped(tmp_path):
    esc = escher_with(tmp_path)
    for k in (1, 3, 6):
        assert len(esc.neighbour_tiler(k).rotations) == k + 1
    # asking for more neighbours than the group has cannot duplicate the identity
    huge = esc.neighbour_tiler(999)
    assert len(huge.rotations) == len(esc.tiler.rotations)
    n_identity = sum(np.allclose(R, np.eye(3)) for R in huge.rotations)
    assert n_identity == 1


def test_neighbours_are_genuine_group_elements(tmp_path):
    """Every added tile must be an exact image under the orbifold group -- this
    is what makes the surroundings the REAL neighbours rather than decoration."""
    esc = escher_with(tmp_path)
    group = [np.asarray(R) for R in esc.tiler.rotations]
    for R in esc.neighbour_tiler(5).rotations:
        assert any(np.allclose(R, G, atol=1e-9) for G in group)


def test_neighbours_are_the_nearest_ones(tmp_path):
    """Proximity ordering is what picks up the tiles meeting at a cone point --
    the ones that form the rosettes."""
    esc = escher_with(tmp_path)
    centre = esc.tiler.tile_centers(esc.mesh.points)[0]
    picked = esc.neighbour_tiler(3).rotations[1:]
    chosen = sorted(float(np.asarray(R) @ centre @ centre) for R in picked)
    others = []
    for R in esc.tiler.rotations:
        R = np.asarray(R)
        if np.allclose(R, np.eye(3)) or any(np.allclose(R, P) for P in picked):
            continue
        others.append(float(R @ centre @ centre))
    if others:
        assert min(chosen) >= max(others) - 1e-9


def test_geometry_grows_but_the_camera_does_not_move(tmp_path):
    """The whole design: more tiles in the picture, same view of the centre one."""
    esc = escher_with(tmp_path)
    esc.args.TILE_CENTRIC_VIEWS = True
    esc.args.VIEW_JITTER_DEG = 0.0
    with torch.no_grad():
        pts = esc.solve_points()

    from escher.rendering.render_sphere_nvdiffrast import build_tiled_sphere

    lone = build_tiled_sphere(
        pts.float(), esc.mesh.faces, esc.mesh.uv, esc.solo_tiler
    )
    ring = build_tiled_sphere(
        pts.float(), esc.mesh.faces, esc.mesh.uv, esc.neighbour_tiler(4)
    )
    assert ring.vertices.shape[0] == 5 * lone.vertices.shape[0]
    assert ring.faces.shape[0] == 5 * lone.faces.shape[0]
    # the centre tile's geometry is bit-identical and comes first
    assert torch.equal(ring.vertices[: lone.vertices.shape[0]], lone.vertices)
    assert torch.equal(ring.faces[: lone.faces.shape[0]], lone.faces)
    # faces never index across tile copies (the tiler invariant)
    n = lone.vertices.shape[0]
    for g in range(5):
        blk = ring.faces[g * lone.faces.shape[0] : (g + 1) * lone.faces.shape[0]]
        assert int(blk.min()) >= g * n and int(blk.max()) < (g + 1) * n
