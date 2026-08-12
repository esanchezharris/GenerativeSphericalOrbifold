"""Per-step SDS schedule arming, shared by the planar and spherical pipelines.

Two facts this module encodes, both measured:

- ``update_step`` is the ONLY code that arms ``grad_clip_val`` from the
  ``CLIP_GRADIENTS_IN_SDS`` schedule, and NEITHER pipeline ever called it -- SDS
  gradient clipping was silently OFF in every historical run, planar and spherical.
- Annealing the max sampled timestep downward moves SDS from layout-scale composition
  (high noise) to detail polish (low noise); ``set_step_range`` is the hook
  (``StableDiffusion`` only -- DeepFloyd lacks it, hence the ``hasattr`` guard).
"""

from __future__ import annotations

__all__ = ["annealed_max_step", "arm_sds"]


def annealed_max_step(args, iteration: int) -> float:
    """Linear anneal of the SDS max-timestep fraction over ``[0, SDS_ANNEAL_END]``."""
    t = min(max(iteration / max(args.SDS_ANNEAL_END, 1), 0.0), 1.0)
    return float(
        args.get("SDS_MAX_START", 0.98)
        + (args.get("SDS_MAX_END", 0.5) - args.get("SDS_MAX_START", 0.98)) * t
    )


def arm_sds(guidance, args, iteration: int) -> None:
    """Call once per optimization step, before ``train_step``."""
    guidance.update_step(0, iteration)
    if not hasattr(guidance, "set_step_range"):
        return
    # SDS_MIN_STEP raises the timestep FLOOR: diffusion deposits high-frequency
    # detail at low t (spectral autoregression), so never sampling below the
    # floor keeps SDS in the layout band -- the round-7 mark-refinement guard.
    # 0/unset = the model default = every prior run.
    floor = float(args.get("SDS_MIN_STEP", 0.0) or 0.0)
    min_step = floor if floor > 0 else guidance.cfg.min_step_percent
    if args.get("SDS_ANNEAL_END", 0) > 0:
        guidance.set_step_range(
            min_step, max(annealed_max_step(args, iteration), min_step)
        )
    elif floor > 0:
        guidance.set_step_range(min_step, guidance.cfg.max_step_percent)
