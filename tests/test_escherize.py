"""Closed-form Escherization: structure of the boundary affine map."""

from __future__ import annotations

import numpy as np
import pytest

from escher.escherize import boundary_affine_map, metric_graph
from escher.OTE.tilings_sphere import OctahedralOrbifold


@pytest.fixture(scope="module")
def orb():
    return OctahedralOrbifold.from_resolution((2, 3, 4), n=8)


def test_affine_map_covers_the_whole_loop(orb):
    loop, M, c, free_ids, cones = boundary_affine_map(orb.mesh, orb.R1, orb.R2)
    L = len(loop)
    assert M.shape == (3 * L, 3 * len(free_ids))
    assert len(cones) == 4
    # every loop vertex is pinned (c row set) or parameterized (M row nonzero)
    row_has = (np.abs(M).sum(axis=1) > 0) | (np.abs(c) > 0)
    assert row_has.reshape(L, 3).any(axis=1).all()


def test_affine_map_is_equivariant(orb):
    """Setting xi to the undeformed left chains must reproduce the undeformed loop."""
    mesh = orb.mesh
    loop, M, c, free_ids, _ = boundary_affine_map(mesh, orb.R1, orb.R2)
    xi = mesh.points[free_ids].reshape(-1)
    u = (M @ xi + c).reshape(len(loop), 3)
    assert np.allclose(u, mesh.points[loop], atol=1e-9)


def test_metric_graph_is_positive_definite():
    S = metric_graph(24, np.ones(24), chords=[2, 5], eps=0.05)
    w = np.linalg.eigvalsh(S)
    assert w.min() > 0
    assert np.allclose(S, S.T)


def test_curvature_penalty_is_psd_and_kills_wiggle():
    from escher.escherize import curvature_penalty

    Q = curvature_penalty(32)
    assert np.allclose(Q, Q.T)
    w = np.linalg.eigvalsh(Q)
    assert w.min() > -1e-10
    # a straight (linear) cyclic ramp has curvature only at the wrap; a
    # sawtooth has curvature everywhere -- the penalty must rank them.
    smooth = np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))
    jagged = np.tile([1.0, -1.0], 16)
    assert jagged @ Q @ jagged > 10 * (smooth @ Q @ smooth)
