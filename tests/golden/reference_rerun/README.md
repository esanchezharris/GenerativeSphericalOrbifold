# Reference rerun (2026-08-12)

How `../x_final.mat` was produced: the reference MATLAB implementation
([`noamaig/spherical_orbifolds`](https://github.com/noamaig/spherical_orbifolds))
was rerun under **GNU Octave 8.4** on `my_man_maxp.mat` — the stock demo mesh,
which is byte-identical to golden `V.mat`/`F.mat` — with `orbifold_type = 4`
(cones `[4 2 2]`, k = 4), i.e. the exact configuration of the original golden
dump. Driver: `dump_golden_solution.m` (this directory).

## Octave compatibility patches applied to the reference clone

None of these change any arithmetic:

1. **`methods (Abstract)` blocks** → concrete `error('abstract')` stubs
   (Octave rejects abstract method declarations outside @-folders). Applied
   mechanically to `EmbeddingStructure.m`, `BoundaryPiece.m`,
   `BoundaryConditions.m`, `LBFGS/Precond.m` by `fix_abstract2.py` (this
   directory).
2. **`triangulation.m`** (this directory) — shim for MATLAB's `triangulation`
   object; only `freeBoundary()` is used (empty for the closed sphere mesh).
3. **`graph.m`** (this directory) — shim for MATLAB's `graph` +
   `shortestpath` (Dijkstra). Tie-breaking could in principle differ from
   MATLAB's; the bit-check below proves it did not on this input.
4. `@Solver/solve_bfgs_fast.m` line 26: `eye(3)` → `speye(3)` and the
   preconditioner matrix wrapped in `sparse(...)` — Octave's `kron`/`qr`
   return dense where MATLAB stays sparse, and `SparseLU` requires sparse.
   Identical values.

## Determinism / fidelity results (measured at dump time)

- Regenerated `A` vs golden `A.mat`: **identical sparsity pattern**, max entry
  difference 3.6e-16 (float noise in rotation-matrix arithmetic).
- Regenerated `b`, `wMat` vs golden: **bit-identical** (max diff 0.0) —
  including the `double(single(W))` truncation and the 1e-3 clamps.
- Converged energy `E_final = 7.873716855189`. The reference counts each edge
  in both directions (`find(adj)` yields ordered pairs), so this is exactly
  **2 ×** the Python solver's converged energy 3.936858427594 — which matches
  to all printed digits.
- Reference vs Python converged embeddings: per-vertex distance max 8.3e-9,
  median 3.0e-9.
- Pre-normalization max radius under Octave: 1.1585 (the Python port's
  documentation says "radii reach at most ~1.16").

Conclusion: the previously self-pinned regression value 3.9368584276 is now
reference-provenanced, and the Python port is validated against the reference
implementation's actual output end to end (`tests/test_golden_solution.py`).
