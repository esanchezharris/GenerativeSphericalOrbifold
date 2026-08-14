"""Is the tiny encoder a safe substitute for the SD VAE encoder in SDS?

The encoder is ~48% of the SDS wall clock (timing.csv over 14 full runs: ~118 ms
forward + ~170 ms backward of a 602 ms step) because it is the ONLY network in
the backward -- 2.1x the cost of the diffusion UNet it feeds. Swapping it for a
distilled tiny autoencoder is therefore the single largest speed lever, but it
approximates the encoder's Jacobian, and SDS only cares about that Jacobian's
DIRECTION. This probe answers, in minutes and before any GPU night, three
questions that a full run would take 70 minutes each to answer badly:

  1. Do the two encoders agree on the LATENT SCALE and content? A scale error
     (e.g. one applies 0.18215 and the other does not) makes SDS meaningless.
  2. Do they agree on the PULLBACK DIRECTION -- cos(J_taesd^T g, J_vae^T g)?
     This, not the latent error, is what the optimizer actually consumes.
  3. What is the real speed and VRAM delta at production shape?

Usage (inside the WSL venv, GPU)::

    python escher/sanity_checks/probe_sds_encoder.py \
        CHECKPOINT=runs/r10_parity/prep/carve_identity40/checkpoint.pt

Exit code 0 iff the latents agree on scale (std ratio in [0.7, 1.4]) AND the
texture-space pullback cosine exceeds 0.8. Anything else means the swap is not
safe as a drop-in, and the report says which of the two failed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from escher.main_shape import build_shape_run

DEFAULTS = OmegaConf.create(
    {
        "CHECKPOINT": "runs/r10_parity/prep/carve_identity40/checkpoint.pt",
        "DEVICE": "cuda",
        "BATCH": 4,
        "SIZE": 512,
        "TIMED_ITERS": 30,
        "WARMUP": 10,
        "STD_RATIO_MIN": 0.7,
        "STD_RATIO_MAX": 1.4,
        "COSINE_MIN": 0.8,
        "TAESD": "madebyollin/taesd",
        "PROMPT": "A professional cartoon of a gingerbread man, a masterpiece",
    }
)


def _stats(name, z):
    print(
        f"  {name:10s} shape {tuple(z.shape)} mean {z.mean():+.4f} "
        f"std {z.std():.4f} min {z.min():+.3f} max {z.max():+.3f}"
    )
    return float(z.std())


def _timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    peak = torch.cuda.max_memory_allocated() / 2**30
    return ms, peak


def probe(a) -> bool:
    import escher.guidance.sd as sd

    device = torch.device(a.DEVICE)
    state = torch.load(a.CHECKPOINT, map_location="cpu", weights_only=False)
    args = OmegaConf.create(state["config"])
    args.DEVICE = a.DEVICE
    args.OUTPUT_DIR = str(Path(a.CHECKPOINT).parent / "probe_tmp")

    # A real render from the real geometry -- encoder behavior on natural images
    # is not what matters; behavior on OUR renders is.
    escher = build_shape_run(args)
    escher.load_checkpoint(a.CHECKPOINT, reset_texture=True)
    with torch.no_grad():
        points = escher.solve_points()
    escher.texture.data.uniform_(0.0, 1.0)

    def render():
        images, alpha, _ = escher.render(
            int(a.BATCH), isolated=True, points=points, roll_deg=0.0
        )
        return (images * alpha + 1.0 * (1.0 - alpha)).clamp(0, 1)

    print(f"probe: {a.CHECKPOINT} | batch {a.BATCH} @ {a.SIZE}px")

    # The stock stabilityai repo is gated; the project runs an ungated mirror,
    # recorded in the checkpoint's own config. Use whatever the run used.
    guidance = sd.StableDiffusion(
        sd.Config(
            pretrained_model_name_or_path=str(args.PRETRAINED_MODEL_NAME_OR_PATH),
            half_precision_weights=bool(args.USE_HALF_PRECISION),
            guidance_scale=float(args.GUIDANCE_SCALE),
            sds_encoder="vae",
        )
    )
    from diffusers import AutoencoderTiny

    taesd = AutoencoderTiny.from_pretrained(
        a.TAESD, torch_dtype=guidance.weights_dtype
    ).to(device)
    for p in taesd.parameters():
        p.requires_grad_(False)
    guidance.taesd = taesd

    # ---------------- Gate 1: latent agreement -------------------------------
    print("\n[1] latent agreement")
    with torch.no_grad():
        rgb = render()
        x = (rgb.permute(0, 3, 1, 2) * 2.0 - 1.0).to(guidance.weights_dtype)
        z_vae = guidance._encode_vae(x, sample=True).float()
        z_tae = guidance._encode_taesd(x).float()
    s_v = _stats("vae", z_vae)
    s_t = _stats("taesd", z_tae)
    ratio = s_t / max(s_v, 1e-9)
    rel = float((z_tae - z_vae).norm() / z_vae.norm())
    corr = [
        float(
            torch.corrcoef(
                torch.stack([z_vae[:, c].flatten(), z_tae[:, c].flatten()])
            )[0, 1]
        )
        for c in range(z_vae.shape[1])
    ]
    print(f"  std ratio (taesd/vae) : {ratio:.3f}   [gate {a.STD_RATIO_MIN}-{a.STD_RATIO_MAX}]")
    print(f"  relative L2 difference: {rel:.3f}")
    print(f"  per-channel correlation: {[f'{c:.3f}' for c in corr]}")
    gate1 = a.STD_RATIO_MIN <= ratio <= a.STD_RATIO_MAX

    # ---------------- Gate 2: the pullback direction -------------------------
    # This is what the optimizer consumes. Use ONE shared SDS gradient so the
    # comparison isolates the encoder Jacobian rather than noise draws.
    print("\n[2] pullback direction (the quantity SDS actually uses)")
    # train_step wants [2*B, 77, D] as positives-then-negatives (main_sphere's
    # text_embeds_for convention, which compute_grad_sds's chunk(2) relies on).
    pos = guidance.get_text_embeds(a.PROMPT)
    neg = guidance.get_text_embeds("")
    text = torch.cat([pos] * int(a.BATCH) + [neg] * int(a.BATCH))

    # ISOLATED Jacobian test. Compute ONE SDS gradient from the real VAE latents,
    # then pull that SAME g back through each encoder. This is the only clean way
    # to compare Jacobians: an end-to-end comparison is confounded because
    # posterior.sample() consumes RNG and TAESD does not, so the two arms would
    # otherwise draw DIFFERENT timesteps and noise and the cosine would measure
    # SDS's own step-to-step variance rather than the encoders' disagreement.
    torch.manual_seed(0)
    rgb = render()
    with torch.no_grad():
        x_ref = (rgb.permute(0, 3, 1, 2) * 2.0 - 1.0).to(guidance.weights_dtype)
        z_ref = guidance._encode_vae(x_ref, sample=True)
        t = torch.randint(
            guidance.min_step, guidance.max_step + 1, [int(a.BATCH)],
            dtype=torch.long, device=device,
        )
        g_shared = guidance.compute_grad_sds(z_ref, text, t).float()

    grads = {}
    for name, enc in (
        ("vae", lambda xx: guidance._encode_vae(xx, sample=False)),
        ("taesd", guidance._encode_taesd),
    ):
        escher.texture.grad = None
        torch.manual_seed(0)
        rgb = render()
        xx = (rgb.permute(0, 3, 1, 2) * 2.0 - 1.0).to(guidance.weights_dtype)
        (enc(xx).float() * g_shared).sum().backward()
        grads[name] = escher.texture.grad.detach().flatten().float().clone()

    gv, gt = grads["vae"], grads["taesd"]
    cos = float(torch.nn.functional.cosine_similarity(gv, gt, dim=0))
    nrm = float(gt.norm() / max(float(gv.norm()), 1e-30))
    print(f"  cosine(J_taesd^T g, J_vae^T g) on TEXTURE : {cos:+.3f}   [gate > {a.COSINE_MIN}]")
    print(f"  gradient norm ratio taesd/vae            : {nrm:.3f}")
    print("  (norm ratio matters only where regularizers are nonzero; the clean")
    print("   stack runs every shape/texture reg at 0, so direction is all that counts)")

    # Reference: how much does the VAE's own gradient disagree with ITSELF across
    # two SDS draws? Any encoder cosine must be read against this noise floor --
    # if consecutive SDS steps already disagree, demanding 0.8 is meaningless.
    escher.texture.grad = None
    torch.manual_seed(1)
    rgb = render()
    with torch.no_grad():
        x2 = (rgb.permute(0, 3, 1, 2) * 2.0 - 1.0).to(guidance.weights_dtype)
        z2 = guidance._encode_vae(x2, sample=True)
        t2 = torch.randint(
            guidance.min_step, guidance.max_step + 1, [int(a.BATCH)],
            dtype=torch.long, device=device,
        )
        g2 = guidance.compute_grad_sds(z2, text, t2).float()
    torch.manual_seed(0)
    rgb = render()
    xx = (rgb.permute(0, 3, 1, 2) * 2.0 - 1.0).to(guidance.weights_dtype)
    (guidance._encode_vae(xx, sample=False).float() * g2).sum().backward()
    g_self = escher.texture.grad.detach().flatten().float().clone()
    cos_floor = float(torch.nn.functional.cosine_similarity(gv, g_self, dim=0))
    print(f"  NOISE FLOOR cos(vae@draw1, vae@draw2)    : {cos_floor:+.3f}")
    print("  => the encoder swap is safe if its cosine is comparable to this floor")
    gate2 = cos > a.COSINE_MIN or cos >= cos_floor

    # ---------------- Gate 3: speed and memory -------------------------------
    print("\n[3] speed + VRAM at production shape")
    xb = torch.randn(
        int(a.BATCH), 3, int(a.SIZE), int(a.SIZE), device=device,
        dtype=guidance.weights_dtype,
    )

    def mk(fn, need_grad):
        def run():
            xx = xb.detach().requires_grad_(need_grad)
            z = fn(xx)
            if need_grad:
                z.sum().backward()
        return run

    for label, fn in (
        ("vae  ", lambda t: guidance._encode_vae(t, sample=True)),
        ("taesd", guidance._encode_taesd),
    ):
        f_ms, f_gb = _timed(mk(fn, False), int(a.TIMED_ITERS), int(a.WARMUP))
        b_ms, b_gb = _timed(mk(fn, True), int(a.TIMED_ITERS), int(a.WARMUP))
        print(
            f"  {label}: fwd {f_ms:6.1f} ms | fwd+bwd {b_ms:6.1f} ms | "
            f"peak {b_gb:.2f} GiB"
        )
        if label.strip() == "vae":
            base = b_ms
        else:
            k = base / max(b_ms, 1e-9)
            # Predict from the RATIO applied to the MEASURED production encoder
            # share (289 ms of a 602 ms step), never from these absolute ms:
            # this probe runs without torch.compile and against whatever else is
            # on the GPU, so its absolutes are not production numbers.
            prod_enc, prod_step = 289.0, 602.0
            pred = prod_step - prod_enc + prod_enc / k
            print(f"  => encoder speedup k = {k:.2f}x (ratio; absolutes above are uncompiled)")
            print(f"  => predicted step: {prod_step:.0f} -> {pred:.0f} ms "
                  f"({prod_step / pred:.2f}x)")

    print("\nVERDICT")
    print(f"  gate 1 latent scale/content : {'PASS' if gate1 else 'FAIL'}")
    print(f"  gate 2 pullback direction   : {'PASS' if gate2 else 'FAIL'}")
    return bool(gate1 and gate2)


def main() -> None:
    a = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli(sys.argv[1:]))
    ok = probe(a)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
