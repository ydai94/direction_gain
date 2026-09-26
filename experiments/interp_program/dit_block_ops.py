"""Shared low-level DiT-block ops for the Exp 68/69/70 DiT-block Q1/Q2/Q3 battery.

Two primitives, both built on exp38's ``BlockHook`` and ``cdg_loop`` WITHOUT editing
either (import-only). The DiT has 60 double-stream blocks; the image-stream residual
at block k is ``out[1]`` with shape (B, T_img, 3072).

- ``capture_blocks``  : one standard CFG denoise, a capture hook on EVERY grid block,
  storing the mean-pooled image-stream residual ``hid`` at chosen step indices. One
  denoise yields the whole {block x step} feature set for a prompt, plus the final
  (unpacked) latent. Serves DETECTION (pooled vectors -> Cohen's d) and LEVERAGE
  (A/B pooled means -> the fixed block-space direction r_k).
- ``generate_leverage_block`` : fixed-``r`` single-fire-step small injection at one
  block, returning the raveled unpacked latent (the DiT analog of exp50's encoder
  finite-difference edit). Same-seed clean vs perturbed isolates the edit's effect.

Everything here is @torch.no_grad and self-consistent: clean and perturbed latents are
produced by the SAME loop (never mixed with exp50's stock-pipe latents), so the concept
axis d_latent = z_A - z_B and the response dz = z_L - z_A live in one representation.
"""
from __future__ import annotations

import numpy as np
import torch

from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

from experiments.contrastive_denoising_guidance_pilot import cdg_loop
from experiments.interp_program.exp38_dit_depth import BlockHook

_EPS = 1e-8


def _prep(pipe, seed, num_steps):
    """Mirror the cdg_loop / exp38 timestep + latent setup. Returns the pieces the
    denoise loops need (latents, timesteps, img_shapes, height, width)."""
    device = pipe._execution_device
    scheduler = pipe.scheduler
    transformer = pipe.transformer
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = height
    generator = torch.Generator(device=device).manual_seed(int(seed))
    num_channels_latents = transformer.config.in_channels // 4
    dtype = next(transformer.parameters()).dtype
    latents = pipe.prepare_latents(1, num_channels_latents, height, width,
                                   dtype, device, generator, None)
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
    timesteps, _ = retrieve_timesteps(scheduler, num_steps, device, sigmas=sigmas, mu=mu)
    return latents, timesteps, img_shapes, height, width


def _unpack_ravel(pipe, latents, height, width):
    """Unpacked latent -> flat fp32 numpy (the exp50 'image latent', pre-VAE)."""
    lat = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    return np.asarray(lat.detach().float().cpu().numpy(), dtype=np.float32).ravel()


@torch.no_grad()
def capture_blocks(pipe, embed, uncond, seed, blocks, steps, *,
                   num_steps=50, cfg=4.0, return_latent=True):
    """Run one CFG denoise of prompt ``embed=(pe,pm)`` (uncond=(pe_u,pm_u)); capture the
    mean-pooled image-stream residual at every block in ``blocks`` at each step index in
    ``steps``. Returns (feats, latent_ravel) where feats[(k, step)] = (D,) fp32 vector.
    Capture is gated to the CONDITIONAL forward only (uncond runs with hooks off)."""
    pe, pm = embed
    pe_u, pm_u = uncond
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    latents, timesteps, img_shapes, height, width = _prep(pipe, seed, num_steps)
    steps = set(int(s) for s in steps)

    def forward(pe_, pm_, ts, ctx):
        with transformer.cache_context(ctx):
            return transformer(
                hidden_states=latents, timestep=ts / 1000, guidance=None,
                encoder_hidden_states_mask=pm_, encoder_hidden_states=pe_,
                img_shapes=img_shapes, attention_kwargs={}, return_dict=False)[0]

    hooks = {k: BlockHook() for k in blocks}
    handles = [transformer.transformer_blocks[k].register_forward_hook(h) for k, h in hooks.items()]
    feats = {}
    try:
        scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            ts = t.expand(latents.shape[0]).to(latents.dtype)
            grab = i in steps
            for h in hooks.values():
                h.mode = "capture" if grab else "off"
                h.slot = "cur"
            cond = forward(pe, pm, ts, "cond")
            if grab:
                for k, h in hooks.items():
                    hid = h.store["cur"]                      # (1, T_img, D)
                    feats[(k, i)] = hid[0].mean(0).float().cpu().numpy().astype(np.float32)
                for h in hooks.values():
                    h.mode = "off"
            uncond_pred = forward(pe_u, pm_u, ts, "uncond")
            comb = uncond_pred + cfg * (cond - uncond_pred)
            pred = comb * (cdg_loop._per_token_norm(cond) / cdg_loop._per_token_norm(comb))
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    finally:
        for hd in handles:
            hd.remove()
    lat = _unpack_ravel(pipe, latents, height, width) if return_latent else None
    return feats, lat


@torch.no_grad()
def generate_leverage_block(pipe, embed, uncond, seed, block_k, r_fixed, rho_lev, fire_step, *,
                            num_steps=50, cfg=4.0):
    """Finite-difference DiT edit: inject the FIXED direction ``r_fixed`` (a (D,) tensor,
    broadcast over image tokens) at ``block_k`` only at step ``fire_step`` with relative
    strength ``rho_lev`` (exp38 lam formula), full denoise, return the raveled unpacked
    latent. ``rho_lev=0`` -> no injection -> clean latent. Same seed as the clean run."""
    pe, pm = embed
    pe_u, pm_u = uncond
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    latents, timesteps, img_shapes, height, width = _prep(pipe, seed, num_steps)
    r_t = None if r_fixed is None else r_fixed.to(latents.device)

    def forward(pe_, pm_, ts, ctx):
        with transformer.cache_context(ctx):
            return transformer(
                hidden_states=latents, timestep=ts / 1000, guidance=None,
                encoder_hidden_states_mask=pm_, encoder_hidden_states=pe_,
                img_shapes=img_shapes, attention_kwargs={}, return_dict=False)[0]

    H = BlockHook()
    handle = transformer.transformer_blocks[block_k].register_forward_hook(H)
    try:
        scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            ts = t.expand(latents.shape[0]).to(latents.dtype)
            fire = (i == int(fire_step)) and (rho_lev > 0.0) and (r_t is not None)
            if fire:
                H.r = r_t; H.rho = float(rho_lev); H.w = 1.0; H.mode = "inject"
                cond = forward(pe, pm, ts, "cond")
                H.mode = "off"
            else:
                cond = forward(pe, pm, ts, "cond")
            uncond_pred = forward(pe_u, pm_u, ts, "uncond")
            comb = uncond_pred + cfg * (cond - uncond_pred)
            pred = comb * (cdg_loop._per_token_norm(cond) / cdg_loop._per_token_norm(comb))
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    finally:
        handle.remove()
    return _unpack_ravel(pipe, latents, height, width)
