"""Generate figure-silhouette targets with FULL Stable Diffusion denoising (not SDS).

The deterministic shape phase needs one clean binary mask of the figure. SDS never
produces one -- it only ever nudges pixels -- but the very same weights run as a normal
txt2img sampler produce crisp silhouettes in seconds. Generate a batch of candidates,
binarize each (``shape_target.binarize_mask``), and write a contact sheet; the largest
candidate is auto-selected as ``target.npy``, and a second invocation with ``CHOOSE=i``
overrides that choice without touching the GPU.

Usage (inside the WSL venv)::

    python escher/make_target.py                 # generate N candidates + contact sheet
    python escher/make_target.py CHOOSE=3        # re-pick: candidate 3 -> target.npy
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from escher.shape_target import binarize_mask, figure_mask_from_flat_ground

DEFAULTS = OmegaConf.create(
    {
        "PROMPT": (
            "a plain solid black silhouette of a gingerbread man, thick rounded limbs, "
            "white background, minimal flat logo, centered, full body"
        ),
        "NEGATIVE": "photo, texture, shading, gradient, outline only, cropped, text",
        "MODEL": "Manojb/stable-diffusion-2-1-base",
        "N": 8,
        "SEED": 0,
        "STEPS": 30,
        "GUIDANCE": 7.5,
        "SIZE": 512,
        "OUT_DIR": "assets/targets/gingerbread",
        "CHOOSE": -1,  # >= 0: skip generation, just re-point target.npy
        "DEVICE": "cuda",
        # COLOR mode: full-color figure images (texture-style prompt) instead of
        # silhouettes -- the anchor images for image-anchored texture init. The
        # best candidate (largest clean figure in the plausibility band) is
        # copied to color.png.
        "COLOR": False,
    }
)


def generate_color(args) -> dict:
    """Generate color figure candidates; pick the best into ``color.png``.

    Same pipeline and auto-pick philosophy as the silhouette path: a candidate
    counts only if it binarizes to one clean figure in the plausibility band
    (the color image's own darkness against the white ground supplies the mask).
    """
    import shutil

    import torch
    from diffusers import StableDiffusionPipeline

    out_dir = Path(args.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(args.get("DEVICE", "cuda"))
    pipe = StableDiffusionPipeline.from_pretrained(
        args.MODEL, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    candidates = []
    for i in range(args.N):
        gen = torch.Generator(device=device).manual_seed(args.SEED + i)
        raw = np.asarray(
            pipe(
                args.PROMPT,
                negative_prompt=args.NEGATIVE,
                num_inference_steps=args.STEPS,
                guidance_scale=args.GUIDANCE,
                height=args.SIZE,
                width=args.SIZE,
                generator=gen,
            ).images[0]
        )
        imageio.imwrite(out_dir / f"color_{i:02d}.png", raw)
        # The bake needs one figure on a FLAT ground of any color (measured: the
        # model favors slate gray behind colorful flat-vector subjects). A
        # patterned ground cannot be segmented and is gated out by its border
        # color spread.
        try:
            mask, ground_std = figure_mask_from_flat_ground(raw)
            area = float(mask.mean())
            # Figure CHROMA gate: the deliverable's colors come from hue-tinting
            # this figure, and achromatic pixels are fixed points of the tint --
            # a white/gray figure renders as a monochrome sphere (measured:
            # seahorse and ray anchors passed the flat-ground gate with pale
            # figures and produced gray tilings).
            fig_px = np.asarray(raw, dtype=np.float64)[..., :3][mask > 0.5] / 255.0
            chroma = float(
                (fig_px.max(axis=1) - fig_px.min(axis=1)).mean()
            ) if len(fig_px) else 0.0
        except ValueError:
            area, ground_std, chroma = 0.0, 1.0, 0.0
        candidates.append((i, area, ground_std, chroma))
        print(
            f"color candidate {i}: figure area {area:.3f}, ground spread "
            f"{ground_std:.3f}, figure chroma {chroma:.3f}",
            flush=True,
        )

    valid = [
        (i, a)
        for i, a, g, ch in candidates
        if 0.08 <= a <= 0.6 and g <= 0.08 and ch >= 0.12
    ]
    if not valid:
        raise ValueError(
            "no color candidate with one clean figure on a FLAT ground -- adjust PROMPT"
        )
    best = max(valid, key=lambda t: t[1])[0]
    shutil.copyfile(out_dir / f"color_{best:02d}.png", out_dir / "color.png")
    print(f"color.png <- candidate {best}")
    return {"chosen": best, "out_dir": str(out_dir)}


def write_choice(out_dir: Path, index: int) -> None:
    mask = imageio.imread(out_dir / f"target_{index:02d}.png").astype(np.float32)
    mask = (mask / mask.max() > 0.5).astype(np.float32)
    np.save(out_dir / "target.npy", mask)
    print(f"target.npy <- candidate {index} (area {mask.mean():.3f} of frame)")


def generate(args) -> dict:
    """Generate candidates; returns ``{"chosen": index, "areas": [...], "out_dir"}``.

    Raises ``ValueError`` when no candidate binarizes cleanly (the CLI shim maps it
    to a nonzero exit); a driver can catch it per job.
    """
    import torch
    from diffusers import StableDiffusionPipeline

    out_dir = Path(args.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = str(args.get("DEVICE", "cuda"))
    pipe = StableDiffusionPipeline.from_pretrained(
        args.MODEL, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    candidates = []  # (index, raw, mask-or-None, area)
    for i in range(args.N):
        gen = torch.Generator(device=device).manual_seed(args.SEED + i)
        raw = pipe(
            args.PROMPT,
            negative_prompt=args.NEGATIVE,
            num_inference_steps=args.STEPS,
            guidance_scale=args.GUIDANCE,
            height=args.SIZE,
            width=args.SIZE,
            generator=gen,
        ).images[0]
        raw = np.asarray(raw)
        imageio.imwrite(out_dir / f"target_{i:02d}_raw.png", raw)
        try:
            mask = binarize_mask(raw)
        except ValueError:
            mask = None  # sampler produced no dark figure; keep the slot for the sheet
        if mask is not None:
            imageio.imwrite(out_dir / f"target_{i:02d}.png", (mask * 255).astype(np.uint8))
        area = float(mask.mean()) if mask is not None else 0.0
        candidates.append((i, raw, mask, area))
        print(f"candidate {i}: seed {args.SEED + i}, figure area {area:.3f}", flush=True)

    fig, axes = plt.subplots(2, args.N, figsize=(2.2 * args.N, 4.8))
    for i, raw, mask, area in candidates:
        axes[0, i].imshow(raw)
        axes[0, i].set_title(f"{i}  (area {area:.2f})", fontsize=9)
        if mask is not None:
            axes[1, i].imshow(mask, cmap="gray")
        for row in (0, 1):
            axes[row, i].set_axis_off()
    fig.suptitle(f'"{args.PROMPT}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=120, bbox_inches="tight")
    print(f"wrote {out_dir / 'contact_sheet.png'}")

    # Auto-pick the largest PLAUSIBLE figure; CHOOSE=i on a re-run overrides without the
    # GPU. Area near 1.0 is an inverted/dark-background generation (the whole frame
    # became "figure"), not a big figure -- the first sheet's naive largest-area pick
    # chose exactly that, so the band is load-bearing.
    valid = [(i, a) for i, _, m, a in candidates if m is not None and 0.08 <= a <= 0.6]
    if not valid:
        raise ValueError("no candidate binarized cleanly -- adjust PROMPT and rerun")
    best = max(valid, key=lambda t: t[1])[0]
    write_choice(out_dir, best)
    return {
        "chosen": best,
        "areas": [a for _, _, _, a in candidates],
        "out_dir": str(out_dir),
    }


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if args.CHOOSE >= 0:
        write_choice(Path(args.OUT_DIR), int(args.CHOOSE))
    else:
        try:
            generate_color(args) if args.get("COLOR", False) else generate(args)
        except ValueError as e:
            raise SystemExit(2) from e


if __name__ == "__main__":
    main()
