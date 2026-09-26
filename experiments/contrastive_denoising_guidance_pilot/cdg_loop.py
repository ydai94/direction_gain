"""Custom Qwen-Image denoising loop with contrastive denoising guidance (CDG).

The stock diffusers ``QwenImagePipeline.__call__`` runs exactly one conditional
and one unconditional transformer forward per step and offers no hook to modify
the prediction *before* the scheduler step (``callback_on_step_end`` fires after
it). CDG needs the difference of the anti-stereotype and stereotype model
predictions injected at each step, so this module re-implements the pipeline's
pre/post-loop setup faithfully and adds the contrastive residual inside the loop.

Per-step math (confirmed against pipeline_qwenimage.py:670-751):

    cond_n  = T(latents, t, neutral)          # raw conditional
    uncond  = T(latents, t, " ")              # unconditional
    comb        = uncond + cfg * (cond_n - uncond)
    pred_n_cfg  = comb * (||cond_n|| / ||comb||)        # per-token norm rescale

When the schedule weight w_t > 0 (and rho > 0):

    cond_a  = T(latents, t, anti)             # raw conditional
    cond_s  = T(latents, t, stereo)           # raw conditional
    r_t     = cond_a - cond_s                 # residual (uncond cancels)
    g_t     = cond_n - uncond                 # CFG guidance direction
    r_use   = r_t (optionally - proj_{g_t} r_t)
    lam     = rho * ||g_t|| / (||r_t|| + eps)            # per-token (dim=-1)
    pred    = pred_n_cfg + w_t * lam * r_use
else:
    pred    = pred_n_cfg

``pred`` is fed to the FlowMatchEulerDiscreteScheduler (velocity prediction:
``prev = sample + dt * pred``). With rho == 0 the loop reduces exactly to the
stock CFG path, which the smoke test relies on for an equivalence check.

Decisions (from plan): residual = raw conditional (uncond cancels); norms are
per-token along the channel dim; guidance vector g_t = cond_n - uncond.
"""

from __future__ import annotations

import numpy as np
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    calculate_shift,
    retrieve_timesteps,
)


VALID_SCHEDULES = ("all", "mid", "late", "early", "ramp_down", "ramp_mid", "none")


def schedule_weight(i: int, num_steps: int, schedule: str) -> float:
    """w_t for step index ``i`` (0-based) out of ``num_steps``.

    Step order is high->low noise (i=0 first/noisiest, i=N-1 last/cleanest).

    all       -> 1 for every step.
    mid       -> 1 for the middle 40% of steps (0.30 <= i/N < 0.70).
    late      -> 1 for the final 50% of steps (i/N >= 0.50).
    early     -> 1 for the first 50% of steps (i/N < 0.50).
    ramp_down -> 1 at i=0, linearly to 0 at i=N-1 (strongest early).
    ramp_mid  -> sin(pi*i/(N-1))**2 bump centered mid-denoising.
    none      -> 0 for every step (loop reduces to stock CFG; smoke test).
    """
    if schedule == "all":
        return 1.0
    if schedule == "none":
        return 0.0
    frac = i / num_steps
    if schedule == "mid":
        return 1.0 if (0.30 <= frac < 0.70) else 0.0
    if schedule == "late":
        return 1.0 if (frac >= 0.50) else 0.0
    if schedule == "early":
        return 1.0 if (frac < 0.50) else 0.0
    denom = max(num_steps - 1, 1)
    if schedule == "ramp_down":
        return float(1.0 - i / denom)
    if schedule == "ramp_mid":
        return float(np.sin(np.pi * i / denom) ** 2)
    raise ValueError(f"unknown schedule: {schedule!r}; expected one of {VALID_SCHEDULES}")


def _per_token_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over the channel dim, keepdim -> (B, T, 1)."""
    return torch.norm(x, dim=-1, keepdim=True)


def compute_timesteps(pipe, num_steps: int = 50,
                      height: int | None = None, width: int | None = None):
    """Return the exact timestep schedule generate_cdg would use.

    image_seq_len is derived analytically from height/width (it equals
    latents.shape[1] after prepare_latents+_pack_latents), so this matches the
    in-loop schedule without running the model. Used for schedule_debug.
    """
    vsf = pipe.vae_scale_factor
    height = height or pipe.default_sample_size * vsf
    width = width or pipe.default_sample_size * vsf
    h = 2 * (height // (vsf * 2))
    w = 2 * (width // (vsf * 2))
    image_seq_len = (h // 2) * (w // 2)
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_steps, pipe._execution_device, sigmas=sigmas, mu=mu,
    )
    return timesteps


@torch.no_grad()
def generate_cdg(
    pipe,
    embeds: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
    seed: int,
    *,
    schedule: str,
    orthogonalize: bool,
    rho: float,
    num_steps: int = 50,
    cfg: float = 4.0,
    eps: float = 1e-8,
    height: int | None = None,
    width: int | None = None,
    diag: list | None = None,
    diag_meta: dict | None = None,
):
    """Run CDG denoising and return a single PIL image.

    ``embeds`` maps {"n","a","s","uncond"} -> (prompt_embeds, prompt_embeds_mask)
    as produced by ``pipe.encode_prompt`` (mask may be None when all-ones).
    Only "n" and "uncond" are required when the schedule never fires.
    """
    if schedule not in VALID_SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule!r}")

    device = pipe._execution_device
    transformer = pipe.transformer
    scheduler = pipe.scheduler

    pe_n, pm_n = embeds["n"]
    pe_u, pm_u = embeds["uncond"]
    dtype = pe_n.dtype

    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor

    # --- pre-loop setup (mirrors __call__:619-665) ---
    generator = torch.Generator(device=device).manual_seed(int(seed))
    num_channels_latents = transformer.config.in_channels // 4
    latents = pipe.prepare_latents(
        1, num_channels_latents, height, width, dtype, device, generator, None,
    )
    img_shapes = [[(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]]

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_steps = retrieve_timesteps(
        scheduler, num_steps, device, sigmas=sigmas, mu=mu,
    )

    # Qwen-Image is not guidance-distilled -> guidance is None (mirrors __call__:653-665).
    guidance = None
    if getattr(transformer.config, "guidance_embeds", False):
        raise RuntimeError("guidance-distilled transformer not supported by CDG loop")

    def forward(pe, pm, ts, ctx):
        with transformer.cache_context(ctx):
            return transformer(
                hidden_states=latents,
                timestep=ts / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=pm,
                encoder_hidden_states=pe,
                img_shapes=img_shapes,
                attention_kwargs={},
                return_dict=False,
            )[0]

    # --- denoising loop ---
    scheduler.set_begin_index(0)
    for i, t in enumerate(timesteps):
        ts = t.expand(latents.shape[0]).to(latents.dtype)

        cond_n = forward(pe_n, pm_n, ts, "cond")
        uncond = forward(pe_u, pm_u, ts, "uncond")
        comb = uncond + cfg * (cond_n - uncond)
        pred_base = comb * (_per_token_norm(cond_n) / _per_token_norm(comb))
        pred = pred_base

        w_t = schedule_weight(i, num_steps, schedule)
        active = w_t > 0.0 and rho > 0.0
        r_t = lam = delta = None
        if active:
            pe_a, pm_a = embeds["a"]
            pe_s, pm_s = embeds["s"]
            cond_a = forward(pe_a, pm_a, ts, "cond_a")
            cond_s = forward(pe_s, pm_s, ts, "cond_s")
            r_t = cond_a - cond_s
            g_t = cond_n - uncond
            r_use = r_t
            if orthogonalize:
                denom = (g_t * g_t).sum(dim=-1, keepdim=True) + eps
                coef = (r_t * g_t).sum(dim=-1, keepdim=True) / denom
                r_use = r_t - coef * g_t
            lam = rho * _per_token_norm(g_t) / (_per_token_norm(r_t) + eps)
            delta = (w_t * lam * r_use).to(pred_base.dtype)
            pred = pred_base + delta

        if diag is not None:
            g_diag = cond_n - uncond
            base_fro = torch.norm(pred_base) + eps
            rec = dict(diag_meta or {})
            rec.update({
                "step_idx": int(i),
                "timestep_value": float(t.item()),
                "schedule_weight": float(w_t),
                "norm_g_mean": float(_per_token_norm(g_diag).mean().item()),
                "pred_base_norm_mean": float(_per_token_norm(pred_base).mean().item()),
            })
            if active:
                rec.update({
                    "norm_r_mean": float(_per_token_norm(r_t).mean().item()),
                    "lambda_mean": float(lam.mean().item()),
                    "lambda_median": float(lam.median().item()),
                    "lambda_max": float(lam.max().item()),
                    "delta_norm_mean": float(_per_token_norm(delta).mean().item()),
                    "delta_over_pred_base": float((torch.norm(delta) / base_fro).item()),
                    "delta_over_g": float((torch.norm(delta) / (torch.norm(g_diag) + eps)).item()),
                    "pred_hat_over_base_diff": float((torch.norm(pred - pred_base) / base_fro).item()),
                })
            else:
                rec.update({
                    "norm_r_mean": float("nan"),
                    "lambda_mean": float("nan"),
                    "lambda_median": float("nan"),
                    "lambda_max": float("nan"),
                    "delta_norm_mean": 0.0,
                    "delta_over_pred_base": 0.0,
                    "delta_over_g": 0.0,
                    "pred_hat_over_base_diff": 0.0,
                })
            diag.append(rec)

        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    # --- decode (mirrors __call__:739-751) ---
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")
    return image[0]


def encode_prompts(pipe, prompts: dict[str, str]) -> dict[str, tuple]:
    """Encode each prompt via the pipeline's own encoder (post-template-drop).

    Returns {key -> (prompt_embeds, prompt_embeds_mask)}; mask may be None.
    Using ``pipe.encode_prompt`` keeps these comparable to the stock baselines.
    """
    out = {}
    for key, prompt in prompts.items():
        pe, pm = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
        out[key] = (pe, pm)
    return out
