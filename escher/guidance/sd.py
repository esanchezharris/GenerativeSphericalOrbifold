from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

"""
Updated from ThreeStudio : Apache-2.0 license
"""
# Basic types
from typing import Any, Optional

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float, Int

# PyTorch Tensor type
from torch import Tensor


def parse_version(ver: str):
    return version.parse(ver)


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


@dataclass
class Config:
    textual_inversion: str = ""
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    enable_memory_efficient_attention: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False
    guidance_scale: float = 100.0
    grad_clip: Optional[Any] = field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    use_sjc: bool = False
    var_red: bool = True
    weighting_strategy: str = "sds"

    token_merging: bool = False
    token_merging_params: Optional[dict] = field(default_factory=dict)

    # torch.compile the UNet and the VAE encoder (the only two networks on the SDS
    # hot path -- the UNet runs once per step under no_grad, the VAE encoder is the
    # sole network in the backward). First call pays a minutes-long compile; only
    # worth it for full-length runs. vae.encoder is compiled (not vae) because
    # AutoencoderKL.encode calls self.encoder directly, bypassing a wrapped forward.
    torch_compile: bool = False

    # Independent noise draws averaged per SDS step (same timestep): 1/N gradient
    # variance at N UNet evals. A run-to-run-consistency lever.
    noise_samples: int = 1

    # Which encoder maps rendered pixels -> SD latents. MEASURED (timing.csv over
    # 14 full runs): the encoder is ~48% of wall clock -- ~118 ms forward plus
    # ~170 ms backward of a 602 ms step -- because it is the ONLY network in the
    # backward. That is 2.1x the diffusion UNet it feeds (129 ms).
    #   "vae"       AutoencoderKL, posterior SAMPLE -- every historical run
    #   "vae_mean"  AutoencoderKL, posterior MEAN -- control isolating the sampling noise
    #   "taesd"     AutoencoderTiny: ~1.2M params, no GroupNorm, no attention,
    #               and its config.scaling_factor is 1.0 because it emits SD's
    #               ALREADY-SCALED latents (verified against diffusers 0.39)
    #   "taesd_bwd" SD latents in the forward (no_grad) + TAESD's Jacobian in the
    #               backward via straight-through -- the quality fallback
    sds_encoder: str = "vae"
    taesd_model_name_or_path: str = "madebyollin/taesd"


class StableDiffusion(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure()

    def configure(self) -> None:
        print(f"Loading Stable Diffusion ... from {self.cfg.pretrained_model_name_or_path}!")

        self.weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32

        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        
        if self.cfg.textual_inversion != "":
            print(f"Loading textual inversion from {self.cfg.textual_inversion}")
            self.pipe.load_textual_inversion(self.cfg.textual_inversion)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                print("PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                print("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            # NHWC is the tensor-core layout for an fp16 conv stack, and since the
            # encoder is the hot network (48% of the step) it wants this at least
            # as much as the UNet does.
            self.pipe.vae.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        # for p in self.tokenizer.parameters():
        #     p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        # The tiny distilled encoder, loaded only when armed. self.vae is KEPT
        # either way: it is 168 MiB of fp16 weights (the win is in ACTIVATIONS,
        # not parameters), decode_latents still needs it, and "taesd_bwd" runs
        # both encoders.
        self.taesd = None
        if self.cfg.sds_encoder in ("taesd", "taesd_bwd"):
            from diffusers import AutoencoderTiny

            print(f"Loading TAESD encoder from {self.cfg.taesd_model_name_or_path}")
            self.taesd = AutoencoderTiny.from_pretrained(
                self.cfg.taesd_model_name_or_path,
                torch_dtype=self.weights_dtype,
            ).to(self.device)
            for p in self.taesd.parameters():
                p.requires_grad_(False)
            if self.cfg.enable_channels_last_format:
                self.taesd.to(memory_format=torch.channels_last)

        if self.cfg.torch_compile:
            self.unet = torch.compile(self.unet)
            # Compile whichever encoders are actually on the hot path.
            if self.cfg.sds_encoder in ("vae", "vae_mean", "taesd_bwd"):
                self.vae.encoder = torch.compile(self.vae.encoder)
            if self.taesd is not None:
                self.taesd.encoder = torch.compile(self.taesd.encoder)

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        print(f"Loaded Stable Diffusion!")

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt, negative_prompt: [str]

        # positive
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        """Rendered pixels -> SD latents. The single dispatch point for SDS_ENCODER.

        Both encoder branches multiply by their OWN ``config.scaling_factor`` rather
        than a literal: AutoencoderKL's is 0.18215 and AutoencoderTiny's is 1.0
        (TAESD emits already-scaled latents), so reading it off the active model
        keeps the convention self-documenting instead of relying on a comment.
        """
        input_dtype = imgs.dtype
        x = (imgs * 2.0 - 1.0).to(self.weights_dtype)  # both encoders take [-1, 1]
        mode = self.cfg.sds_encoder
        if mode == "taesd":
            latents = self._encode_taesd(x)
        elif mode == "taesd_bwd":
            # Straight-through: the VALUE is exactly the SD latent (so the UNet sees
            # no distribution shift and the score is evaluated at the true point);
            # only the pullback J^T is TAESD's.
            with torch.no_grad():
                z_sd = self._encode_vae(x, sample=True)
            z_t = self._encode_taesd(x)
            latents = z_t + (z_sd - z_t).detach()
        else:
            latents = self._encode_vae(x, sample=(mode != "vae_mean"))
        return latents.to(input_dtype)

    def _encode_vae(self, x: Tensor, sample: bool) -> Tensor:
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample() if sample else posterior.mean
        return z * self.vae.config.scaling_factor

    def _encode_taesd(self, x: Tensor) -> Tensor:
        # AutoencoderTiny returns .latents (no latent_dist -- it is deterministic,
        # there is no posterior to sample). Same 8x downscale as the SD VAE.
        return self.taesd.encode(x).latents * self.taesd.config.scaling_factor

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(latents, (latent_height, latent_width), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        # Averaging several independent noise draws at the SAME timestep reduces
        # per-step gradient variance by 1/N at N UNet evals (noise_samples > 1 is
        # a run-to-run-consistency lever, not a quality lever per se).
        n = max(int(self.cfg.noise_samples), 1)
        grad = 0.0
        for _ in range(n):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            grad = grad + w * (noise_pred - noise)
        return grad / n

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma**2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

            Ds = zs - sigma * noise_pred

            if self.cfg.var_red:
                grad = -(Ds - y) / sigma
            else:
                grad = -(Ds - zs) / sigma

        return grad

    def train_step(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        rgb_as_latents=False,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False)
        else:
            # Skip the resize when the render is already 512 (RENDER_SIZE 512 is
            # the production setting, so this fired every step for nothing). A
            # bilinear resize to the SAME size with align_corners=False maps out[i]
            # to src[i] with weights (1, 0), i.e. it is the identity -- so this is
            # bit-exact, not merely statistically equivalent.
            rgb_BCHW_512 = rgb_BCHW
            if rgb_BCHW.shape[-2:] != (512, 512):
                rgb_BCHW_512 = F.interpolate(
                    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
                )
            # encode image into latents with the configured encoder
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad = self.compute_grad_sjc(latents, text_embeddings, t)
        else:
            grad = self.compute_grad_sds(latents, text_embeddings, t)

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return loss, t
        # return {
        #     "sds": loss,
        #     "grad_norm": grad.norm(),
        #     "timestep": t,
        # }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

    def set_step_range(self, min_percent: float, max_percent: float) -> None:
        """Move the SDS timestep sampling window; ``train_step`` reads these per call.

        Annealing ``max_percent`` downward over training shifts SDS from layout-scale
        edits (high noise) to detail refinement (low noise).
        """
        self.min_step = int(self.num_train_timesteps * min_percent)
        self.max_step = int(self.num_train_timesteps * max_percent)
