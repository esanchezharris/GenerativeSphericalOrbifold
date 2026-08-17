"""SDS_ENCODER dispatch: the swap must be opt-in and the fallback must be honest.

The encoder is ~48% of SDS wall clock (it is the only network in the backward),
so swapping it for a distilled tiny autoencoder is the largest speed lever --
but it approximates the encoder's Jacobian, so the default must stay the real
VAE and the straight-through mode must genuinely carry SD's forward value with
TAESD's gradient.
"""

from __future__ import annotations

import pytest
import torch

import escher.guidance.sd as sd


class FakePosterior:
    def __init__(self, value):
        self._v = value

    def sample(self):
        return self._v * 1.0

    @property
    def mean(self):
        return self._v * 100.0  # deliberately distinct from sample()


class FakeEncoder:
    """Stands in for AutoencoderKL: .encode(x).latent_dist, .config.scaling_factor."""

    def __init__(self, coeff, scaling):
        self.coeff = coeff
        self.config = type("C", (), {"scaling_factor": scaling})()

    def encode(self, x):
        return type("O", (), {"latent_dist": FakePosterior(x * self.coeff)})()


class FakeTiny:
    """Stands in for AutoencoderTiny: .encode(x).latents, scaling_factor 1.0."""

    def __init__(self, coeff, scaling=1.0):
        self.coeff = coeff
        self.config = type("C", (), {"scaling_factor": scaling})()

    def encode(self, x):
        return type("O", (), {"latents": x * self.coeff})()


def make(mode, vae_coeff=2.0, vae_scale=0.5, tiny_coeff=3.0):
    g = sd.StableDiffusion.__new__(sd.StableDiffusion)
    g.cfg = sd.Config(sds_encoder=mode)
    g.weights_dtype = torch.float32
    g.vae = FakeEncoder(vae_coeff, vae_scale)
    g.taesd = FakeTiny(tiny_coeff)
    return g


def test_default_is_the_real_vae():
    """Every historical config and run must reproduce unchanged."""
    assert sd.Config().sds_encoder == "vae"


def test_vae_mode_samples_the_posterior_and_applies_its_scaling():
    g = make("vae")
    x = torch.ones(1, 3, 8, 8)
    # imgs*2-1 = 1.0 for ones input; sample -> *2.0 coeff; *0.5 scaling
    assert torch.allclose(g.encode_images(x), torch.full((1, 3, 8, 8), 1.0))


def test_vae_mean_mode_uses_the_mean_not_a_sample():
    g = make("vae_mean")
    x = torch.ones(1, 3, 8, 8)
    # mean is 100x the sample in the fake, so this must differ sharply
    assert torch.allclose(g.encode_images(x), torch.full((1, 3, 8, 8), 100.0))


def test_taesd_mode_reads_latents_and_its_own_scaling_factor():
    g = make("taesd")
    x = torch.ones(1, 3, 8, 8)
    # tiny coeff 3.0 * scaling 1.0
    assert torch.allclose(g.encode_images(x), torch.full((1, 3, 8, 8), 3.0))


def test_taesd_bwd_has_sd_value_and_taesd_gradient():
    """The straight-through contract -- the only nontrivial math in the swap.

    Forward must be EXACTLY the SD latent (the UNet sees no distribution shift
    and the score is evaluated at the true point); the gradient must come only
    from the tiny encoder.
    """
    g = make("taesd_bwd", vae_coeff=2.0, vae_scale=0.5, tiny_coeff=3.0)
    x = torch.ones(1, 3, 8, 8, requires_grad=True)

    out = g.encode_images(x)
    # value: the vae path (coeff 2.0 * scaling 0.5 = 1.0), NOT the tiny path (3.0)
    assert torch.allclose(out, torch.full((1, 3, 8, 8), 1.0))

    out.sum().backward()
    # d/dx of (2*x-1)*3.0 = 6.0 -- the TINY coefficient, not the vae's 1.0
    assert torch.allclose(x.grad, torch.full((1, 3, 8, 8), 6.0))


def test_identity_interpolate_is_skipped_bit_exactly():
    """RENDER_SIZE 512 resized to 512 was a no-op running every step.

    Bit-exactness matters: this is on the default path, so it must not perturb
    historical reproductions at all.
    """
    g = make("vae")
    captured = {}

    def spy(imgs):
        captured["x"] = imgs
        return torch.zeros(imgs.shape[0], 4, 64, 64)

    g.encode_images = spy
    g.device = torch.device("cpu")
    g.min_step, g.max_step = 20, 980
    g.alphas = torch.linspace(0.999, 0.001, 1000)
    g.cfg.weighting_strategy = "sds"
    g.grad_clip_val = None

    def fake_grad(latents, text, t):
        return torch.zeros_like(latents)

    g.compute_grad_sds = fake_grad

    rgb = torch.rand(2, 512, 512, 3)
    g.train_step(rgb, torch.zeros(4, 77, 1024))
    assert torch.equal(captured["x"], rgb.permute(0, 3, 1, 2)), "512 must pass through untouched"

    rgb_small = torch.rand(2, 256, 256, 3)
    g.train_step(rgb_small, torch.zeros(4, 77, 1024))
    assert captured["x"].shape[-2:] == (512, 512), "non-512 must still be resized"


def test_sds_encoder_is_plumbed_from_config(monkeypatch, tmp_path):
    """The knob must actually reach sd.Config from the YAML."""
    import escher.main_sphere as ms

    captured = {}

    class FakeSD:
        def __init__(self, cfg):
            captured["cfg"] = cfg
            self.text_encoder = None
            self.cfg = cfg

        def get_text_embeds(self, p):
            return torch.zeros(1, 77, 1024)

    # _init_guidance imports the module locally, so patch it at its source.
    monkeypatch.setattr(sd, "StableDiffusion", FakeSD)
    from .test_main_shape import tiny_args

    args = tiny_args(tmp_path)
    args.SDS_ENCODER = "taesd"
    esc = ms.SphereEscher.__new__(ms.SphereEscher)
    esc.args = args
    esc.device = torch.device("cpu")
    esc._init_guidance()
    assert captured["cfg"].sds_encoder == "taesd"
