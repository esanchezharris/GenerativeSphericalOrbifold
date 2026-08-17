"""The reference implementation's converged answer and ours are the same embedding.

Provenance of ``tests/golden/x_final.mat`` (2026-08-12): the reference MATLAB code
(noamaig/spherical_orbifolds) was rerun under GNU Octave 8.4 on the golden mesh
(``my_man_maxp.mat``, identical to golden V/F), ``orbifold_type 4`` -> cones
``[4 2 2]``, k=4 -- the exact configuration of the original dump. The rerun's
regenerated constraint system matched tests/golden to float noise (A: identical
sparsity, max entry diff 3.6e-16; b and wMat bit-identical), proving the cut and
the boundary equations reproduce. Scripts + Octave shims:
``tests/golden/reference_rerun/``.

The reference sums every edge in BOTH directions (its I, J come from
``find(adj)``), so its energy is exactly twice ours:
``E_ref = 7.873716855189 = 2 x 3.936858427594``. Measured agreement of the
converged embeddings at dump time: per-vertex max 8.3e-9, median 3.0e-9.
"""

import numpy as np
import pytest
import scipy.io
import torch

from tests.golden import GOLDEN_DIR, load_golden


@pytest.fixture(scope="module")
def reference():
    d = scipy.io.loadmat(str(GOLDEN_DIR / "x_final.mat"))
    return np.asarray(d["x_final"]).ravel(), float(np.asarray(d["E_final"]).ravel()[0])


@pytest.fixture(scope="module")
def solved():
    from escher.OTE.core.spherical.solver import solve_spherical_embedding

    g = load_golden()
    return solve_spherical_embedding(
        edges=g.edges, weights=g.edge_weights, laplacian=g.wmat, A=g.A, b=g.b, x0=g.x0
    )


def test_reference_solution_is_feasible_and_unit(reference):
    x_ref, _ = reference
    g = load_golden()
    assert np.abs(g.A @ x_ref - g.b).max() < 1e-9
    radii = np.linalg.norm(x_ref.reshape(-1, 3), axis=1)
    assert np.abs(radii - 1.0).max() < 1e-12


def test_energies_match_the_reference(reference, solved):
    _, e_ref = reference
    # Factor 2: the reference counts each edge in both directions.
    assert solved.stage2.energy == pytest.approx(e_ref / 2.0, rel=1e-9)


def test_our_energy_of_their_answer_equals_their_energy(reference):
    # Cross-checks the CONVENTIONS, not just the endpoints: our energy function,
    # evaluated at the reference's converged embedding, must reproduce its E/2.
    from escher.OTE.core.spherical.karcher import karcher_energy

    x_ref, e_ref = reference
    g = load_golden()
    e = karcher_energy(
        torch.as_tensor(x_ref.reshape(-1, 3)),
        torch.as_tensor(g.edges),
        torch.as_tensor(g.edge_weights),
    )
    assert float(e) == pytest.approx(e_ref / 2.0, rel=1e-11)


def test_converged_embeddings_coincide(reference, solved):
    x_ref, _ = reference
    dist = np.linalg.norm(x_ref.reshape(-1, 3) - solved.points, axis=1)
    # Observed 8.3e-9 max / 3.0e-9 median at dump time; 1e-6 leaves two orders of
    # margin for BLAS/platform variation while still pinning the same minimizer.
    assert dist.max() < 1e-6
