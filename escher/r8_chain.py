r"""One-command deterministic figure chain, with the ESCHERIZE-SCREEN.

The round-8 pipeline end to end, no diffusion after the figure images:

1. silhouette candidates (full SD denoising, seconds)
2. per-candidate rigid alignment onto the undeformed tile (CPU seconds)
3. **escherize-screen**: the closed-form reachability solve per candidate --
   SECONDS each -- and the candidate with the best ceiling wins. This replaces
   the 40-minute carve screen with a principled reachability measurement: the
   ceiling IS the quantity the old 150-step screen was estimating.
4. realization carve of the winner (GPU ~7 min; the target is reachable by
   construction, expect hard IoU >= 0.95)
5. flat "lineal color" anchor generation (flat-ground gate)
6. correspondence-warped flat bake -> tint render + turntable

Usage (inside the WSL venv, GPU)::

    python escher/r8_chain.py NAME=butterfly \
        "SILHOUETTE_PROMPT=a plain solid black silhouette of a butterfly with spread wings, white background, minimal flat logo, centered, full body" \
        "ANCHOR_PROMPT=a butterfly with spread wings, minimal flat 2d vector icon, lineal color, white background, centered, full body"

Every stage skips itself when its artifact already exists, so a failed chain
resumes where it stopped.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.main_shape import build_shape_run, make_context, prepare_target, run_shape
from escher.render_final import finalize
from escher.texture_init import bake_texture_init
from escher.warped_bake import warped_anchor

DEFAULTS = OmegaConf.create(
    {
        "NAME": "",
        "SILHOUETTE_PROMPT": "",
        "ANCHOR_PROMPT": "",
        "OUT_ROOT": "runs/r8_chain",
        "N_CANDIDATES": 16,
        "LIMB_WEIGHT_MAX": 5.0,
        "SHAPE_STEPS": 1500,
    }
)

CARVE_CONF = [
    "configs/sphere_shape_weights_fish2.yaml",
    "configs/sphere_shape_weights_fish2_hi.yaml",
]


def carve_args(out_dir: Path, **over):
    root = Path(__file__).parent / "configs"
    args = OmegaConf.merge(
        OmegaConf.load(root / "sphere.yaml"),
        OmegaConf.load(root / "sphere_shape.yaml"),
        *(OmegaConf.load(Path(__file__).parent / c) for c in CARVE_CONF),
    )
    args.OUTPUT_DIR = str(out_dir)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def chain(a) -> dict:
    root = Path(a.OUT_ROOT) / a.NAME
    root.mkdir(parents=True, exist_ok=True)
    log: dict = {"name": a.NAME}

    # ---- 1. silhouette candidates -----------------------------------------
    targets_dir = root / "targets"
    if not (targets_dir / "contact_sheet.png").exists():
        from escher.make_target import generate

        gen_args = OmegaConf.merge(
            OmegaConf.create(
                {
                    "PROMPT": a.SILHOUETTE_PROMPT,
                    "NEGATIVE": "photo, texture, shading, gradient, outline only, cropped, text",
                    "MODEL": "Manojb/stable-diffusion-2-1-base",
                    "N": int(a.N_CANDIDATES),
                    "SEED": 0,
                    "STEPS": 30,
                    "GUIDANCE": 7.5,
                    "SIZE": 512,
                    "OUT_DIR": str(targets_dir),
                    "DEVICE": "cuda",
                }
            )
        )
        generate(gen_args)

    # ---- 2+3. escherize-screen -------------------------------------------
    screen_json = root / "escherize_screen.json"
    if screen_json.exists():
        screen = json.loads(screen_json.read_text())
    else:
        from escher.escherize import escherize

        # One reference checkpoint carries the carve config/camera for all
        # candidates (escherize reads config+framing from a checkpoint).
        ref_dir = root / "ref"
        ref_ckpt = ref_dir / "checkpoint.pt"
        if not ref_ckpt.exists():
            ref = build_shape_run(carve_args(ref_dir, DEVICE="cpu"))
            ref.save_checkpoint(0)

        rows = []
        for cand in sorted(targets_dir.glob("target_[0-9][0-9].png")):
            cdir = root / "screen" / cand.stem
            cdir.mkdir(parents=True, exist_ok=True)
            try:
                al_args = carve_args(
                    cdir, DEVICE="cpu", TARGET_MASK=str(cand), TARGET_SMOOTH_RADIUS=0
                )
                esc = build_shape_run(al_args)
                ctx = make_context(esc, al_args)
                prepare_target(esc, ctx, al_args)  # writes target_aligned.png
                res = escherize(
                    OmegaConf.create(
                        {
                            "CARVE": str(ref_ckpt),
                            "TARGET": str(cdir / "target_aligned.png"),
                            "OUT_DIR": str(cdir),
                            "CHORDS": [2, 3, 5, 8],
                            "LIMB_WEIGHT_MAX": float(a.LIMB_WEIGHT_MAX),
                            "POSITION_EPS": 0.05,
                        }
                    )
                )
                rows.append({"candidate": cand.stem, "ceiling": res["ceiling_hard_iou_vs_target"]})
            except Exception as e:  # noqa: BLE001 -- a bad candidate must not kill the screen
                rows.append({"candidate": cand.stem, "ceiling": -1.0, "error": str(e)})
        rows.sort(key=lambda r: r["ceiling"], reverse=True)
        screen = {"rows": rows, "winner": rows[0]["candidate"]}
        screen_json.write_text(json.dumps(screen, indent=2))
    winner = screen["winner"]
    log["escherize_screen_winner"] = winner
    log["ceiling"] = screen["rows"][0]["ceiling"]
    wdir = root / "screen" / winner

    # ---- 4. realization carve --------------------------------------------
    carve_dir = root / "carve"
    if not (carve_dir / "checkpoint.pt").exists():
        result = run_shape(
            carve_args(
                carve_dir,
                TARGET_MASK=str(wdir / "escherized_target.npy"),
                ALIGN_IDENTITY=True,
                TARGET_SMOOTH_RADIUS=0,
                LIMB_WEIGHT_MAX=float(a.LIMB_WEIGHT_MAX),
                SHAPE_STEPS=int(a.SHAPE_STEPS),
                KEEP_CHECKPOINTS=2,
            )
        )
        log["carve"] = {k: result[k] for k in ("iou", "valid") if k in result}

    # ---- 5. flat anchor ---------------------------------------------------
    anchor_dir = root / "anchor"
    if not (anchor_dir / "color.png").exists():
        from escher.make_target import generate_color

        generate_color(
            OmegaConf.create(
                {
                    "PROMPT": a.ANCHOR_PROMPT,
                    "NEGATIVE": "photo, texture, shading, gradient, outline only, cropped, text",
                    "MODEL": "Manojb/stable-diffusion-2-1-base",
                    "N": 8,
                    "SEED": 0,
                    "STEPS": 30,
                    "GUIDANCE": 7.5,
                    "SIZE": 512,
                    "OUT_DIR": str(anchor_dir),
                    "DEVICE": "cuda",
                }
            )
        )

    # ---- 6. warped bake + render -----------------------------------------
    final_dir = root / "final"
    if not (final_dir / "final.png").exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        state = torch.load(
            carve_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        size = int(state["config"].get("SHAPE_RENDER_SIZE") or state["config"]["RENDER_SIZE"])
        color, mask = warped_anchor(
            wdir / "correspondence.npz",
            wdir / "target_aligned.png",
            anchor_dir / "color.png",
            size,
        )
        bake_texture_init(
            carve_dir / "checkpoint.pt",
            anchor_dir / "color.png",
            final_dir / "texture_init.npy",
            fill="flat",
            aligned_override=(color, mask),
        )
        state["texture"] = torch.as_tensor(
            np.load(final_dir / "texture_init.npy"), dtype=torch.float32
        )
        torch.save(state, final_dir / "checkpoint.pt")
        finalize(
            str(final_dir / "checkpoint.pt"),
            tint=True,
            shade=True,
            out_dir=str(final_dir),
            turntable=True,
        )
    log["final"] = str(final_dir / "final.png")

    (root / "chain_log.json").write_text(json.dumps(log, indent=2))
    print(f"chain complete: {a.NAME} | ceiling {log['ceiling']:.4f} -> {final_dir}")
    return log


def main() -> None:
    a = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if not (a.NAME and a.SILHOUETTE_PROMPT and a.ANCHOR_PROMPT):
        raise SystemExit("usage: NAME=<figure> SILHOUETTE_PROMPT=... ANCHOR_PROMPT=...")
    chain(a)


if __name__ == "__main__":
    main()
