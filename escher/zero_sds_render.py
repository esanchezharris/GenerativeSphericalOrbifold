"""The ZERO-SDS deliverable: carve -> bake -> tint -> render. No diffusion.

Round 7 measured, across four arms and three figures, that outline-only SDS on
a frozen texture DECOUPLES the boundary from the baked figure's edge -- the
step-0 state (bake aligned to the carve, tint 3-coloring) was already the
flat-iconic deliverable, and every SDS step degraded the correspondence. This
script IS that step-0 state, produced directly: load a carve, bake the flat
anchor onto it, write the texture into the checkpoint, and export the tinted
renders + turntable. Deterministic, first-time-right by construction.

Usage (inside the WSL venv)::

    python escher/zero_sds_render.py CARVE=<carve>/checkpoint.pt \
        COLOR_IMAGE=<anchor>/color.png OUT_DIR=runs/r7_zero/<name>
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.render_final import finalize
from escher.texture_init import bake_texture_init

DEFAULTS = OmegaConf.create(
    {"CARVE": "", "COLOR_IMAGE": "", "OUT_DIR": "", "FILL": "flat"}
)


def zero_sds_render(
    carve: str | Path,
    color_image: str | Path,
    out_dir: str | Path,
    fill: str = "flat",
) -> dict:
    carve, out_dir = Path(carve), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bake_path = out_dir / "texture_init.npy"
    bake_texture_init(carve, color_image, bake_path, fill=fill)

    state = torch.load(carve, map_location="cpu", weights_only=False)
    tex = torch.as_tensor(np.load(bake_path), dtype=torch.float32)
    res = int(state["config"].get("TEXTURE_RESOLUTION", 256))
    if tex.shape != (res, res, 3):
        raise ValueError(f"bake {tuple(tex.shape)} != checkpoint resolution {res}")
    state["texture"] = tex
    ckpt = out_dir / "checkpoint.pt"
    torch.save(state, ckpt)

    artifacts = finalize(str(ckpt), tint=True, shade=True, out_dir=str(out_dir), turntable=True)
    print(f"zero-SDS deliverable -> {out_dir}")
    return artifacts


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if not (args.CARVE and args.COLOR_IMAGE and args.OUT_DIR):
        raise SystemExit("usage: CARVE=<ckpt> COLOR_IMAGE=<png> OUT_DIR=<dir> [FILL=flat|nearest]")
    zero_sds_render(args.CARVE, args.COLOR_IMAGE, args.OUT_DIR, fill=str(args.FILL))


if __name__ == "__main__":
    main()
