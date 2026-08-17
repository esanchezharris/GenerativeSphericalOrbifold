"""The solver acceleration knobs must change the ROUTE, never the answer.

Every knob here is opt-in and defaults to the reference behavior. The gates:
caching is bit-identical, the preconditioner cadence and the L-BFGS changes
land on the same minimiser, and the reused adjoint factor produces the same
gradient as a fresh factorization.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.differentiable import SphericalEmbedder
from tests.golden import load_golden


@pytest.fixture(scope="module")
def problem():
    g = load_golden()
    return g


def embedder(g, **over):
    return SphericalEmbedder(g.edges, g.A, g.b, g.x0, **over)


def chain(emb, g, n=4, drift=0.01, seed=0):
    """Run a warm chain like SDS does: small weight drift, solve each step."""
    rng = np.random.default_rng(seed)
    w = torch.as_tensor(g.edge_weights, dtype=torch.float64)
    out = []
    for _ in range(n):
        w = w * torch.as_tensor(
            np.exp(rng.normal(0.0, drift, size=w.shape)), dtype=torch.float64
        )
        out.append(emb(w.clone()).detach().numpy().copy())
    return out


def test_cache_affine_is_bit_identical(problem):
    """A and b are fixed for the embedder's lifetime, so this is a pure cache."""
    base = chain(embedder(problem), problem)
    cached = chain(embedder(problem, cache_affine=True), problem)
    for a, b in zip(base, cached):
        assert np.array_equal(a, b), "caching a constant must not change any bit"


def test_stale_preconditioner_lands_on_the_same_minimiser(problem):
    """A preconditioner steers; it cannot move the fixed point.

    Not bit-identical: a different preconditioner takes a different route and
    stops at a different point of the same convergence basin. Measured max
    per-vertex move 4.1e-9 -- 240x inside the golden MATLAB test's 1e-6 gate and
    far below anything SDS resolves.
    """
    base = chain(embedder(problem), problem)
    stale = chain(embedder(problem, cache_affine=True, precond_every=5), problem)
    for a, b in zip(base, stale):
        assert np.abs(a - b).max() < 1e-7


def test_textbook_two_loop_and_memory_reach_the_same_solution(problem):
    """The reference's second loop runs in the wrong direction (see lbfgs.py).

    Fixing it changes the search direction and therefore the path -- but L-BFGS
    directions only affect the route, so the minimiser must be unchanged.
    """
    base = chain(embedder(problem), problem)
    fixed = chain(
        embedder(problem, two_loop_order="textbook", memory=8), problem
    )
    for a, b in zip(base, fixed):
        assert np.abs(a - b).max() < 1e-7


def test_textbook_order_needs_fewer_iterations(problem):
    """The point of the fix: with the correct order, memory finally helps."""
    ref = embedder(problem)
    chain(ref, problem, n=3)
    fix = embedder(problem, two_loop_order="textbook", memory=8)
    chain(fix, problem, n=3)
    assert fix.total_iterations < ref.total_iterations


def test_tol_grad_exits_earlier_but_stays_stationary(problem):
    ref = embedder(problem)
    chain(ref, problem, n=3)
    fast = embedder(problem, tol_grad=1e-8)
    pts = chain(fast, problem, n=3)
    assert fast.total_iterations < ref.total_iterations
    # the adjoint's precondition: still essentially stationary
    assert fast.last_stationarity < 1e-7
    assert np.all(np.isfinite(pts[-1]))


def test_reused_adjoint_factor_gives_the_same_gradient(problem):
    """GMRES with a stale LU as preconditioner is EXACT -- it drives the true
    residual of the current matrix, which the caller verifies anyway."""
    g = problem

    def grads(**over):
        emb = embedder(g, **over)
        rng = np.random.default_rng(0)
        out = []
        w0 = torch.as_tensor(g.edge_weights, dtype=torch.float64)
        for _ in range(3):
            w0 = w0 * torch.as_tensor(
                np.exp(rng.normal(0.0, 0.01, size=w0.shape)), dtype=torch.float64
            )
            w = w0.clone().requires_grad_(True)
            pts = emb(w)
            (pts * torch.linspace(0.1, 1.0, pts.numel(), dtype=torch.float64).reshape(pts.shape)).sum().backward()
            out.append(w.grad.numpy().copy())
        return emb, out

    _, base = grads()
    emb, reused = grads(adjoint_reuse_factor=True)
    for a, b in zip(base, reused):
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)
        assert rel < 1e-8, f"reused-factor gradient drifted by {rel:.2e}"
    assert emb.adjoint_refactorizations < 3, "the factor must actually be reused"
    assert emb.adjoint_seconds > 0.0, "adjoint timing must be recorded"


def test_full_stack_matches_the_reference_solution(problem):
    """Everything on at once, against the untouched defaults."""
    base = chain(embedder(problem), problem)
    full = chain(
        embedder(
            problem,
            cache_affine=True,
            precond_every=5,
            two_loop_order="textbook",
            memory=8,
            tol_grad=1e-8,
            line_search="interp",
            adjoint_reuse_factor=True,
        ),
        problem,
    )
    for a, b in zip(base, full):
        assert np.abs(a - b).max() < 1e-6
