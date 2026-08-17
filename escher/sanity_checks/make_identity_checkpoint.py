"""Write an undeformed identity checkpoint (W = 0) for from-scratch SDS runs.

The literal paper method on the sphere starts from the undeformed fundamental
domain and lets SDS sculpt everything (the gempure precedent,
batch_gem_full.yaml). This writes that starting state as a resumable
checkpoint, embedding a carve-compatible config so load_checkpoint's
mismatch warnings stay quiet.

Run from the repo root::

    python escher/sanity_checks/make_identity_checkpoint.py \
        OUT_DIR=runs/r10_parity/prep/carve_identity40 KITE_N=40

Any further KEY=value args are forwarded as config overrides -- e.g.
``W_INIT_RANDN=1.0 SEED=0`` writes a seeded random-W start instead of the
undeformed identity (build_shape_run seeds torch from args.SEED, so the
draw is deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from escher.main_shape import build_shape_run
from escher.r8_chain import carve_args


def main() -> None:
    cli = OmegaConf.to_container(OmegaConf.from_cli(sys.argv[1:]))
    out_dir = Path(cli.pop("OUT_DIR", "runs/r10_parity/prep/carve_identity40"))
    kite_n = int(cli.pop("KITE_N", 40))
    out_dir.mkdir(parents=True, exist_ok=True)
    args = carve_args(
        out_dir,
        kite_n,
        DEVICE="cpu",
        # Must match the texture run's interpretation of W: under
        # cotangent-relative weights, W = 0 is the NATURAL undeformed domain
        # (uniform weights would be its harmonic distortion, 82x area spread).
        COTANGENT_RELATIVE_WEIGHTS=True,
        **cli,
    )
    escher = build_shape_run(args)
    escher.save_checkpoint(0)
    w_note = "W=0, undeformed" if float(args.get("W_INIT_RANDN", 0) or 0) == 0 else (
        f"W~randn*{args.W_INIT_RANDN}, SEED={args.SEED}"
    )
    print(f"wrote {out_dir / 'checkpoint.pt'} (KITE_N={kite_n}, {w_note})")


if __name__ == "__main__":
    main()
