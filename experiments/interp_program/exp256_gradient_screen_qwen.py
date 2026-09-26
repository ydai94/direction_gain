#!/usr/bin/env python3
"""Qwen all-layer first-order gradient screen for Exp256."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from exp256_common import (
    ALL_INTERNAL_LAYERS,
    ALPHA_OUTPUT,
    BASE,
    CFG,
    INTERNAL_SITES,
    NUM_STEPS,
    N_SHARDS,
    OUT,
    TIMESTEPS,
    TRAJECTORY_SEEDS,
    apply_centered_projection,
    atomic_json,
    atomic_parquet,
    tensor_sha256,
)

from experiments.causal_patching import encoder_runner as ER
from experiments.causal_patching.projection_patcher import ProjectionPatcher
from experiments.causal_patching.run_three_methods import load_pipe
from experiments.contrastive_denoising_guidance_pilot import cdg_loop
import experiments.interp_program.dit_block_ops as DBO


GENERIC = ("person", "man", "woman", "people", "girl", "boy", "child", "worker")
EPSILONS = (0.25, 0.125, 0.0625)
MAX_PEAK_BYTES = 75 * 1024**3


def diff_rows(left: torch.Tensor, right: torch.Tensor) -> list[int]:
    length = min(left.shape[0], right.shape[0])
    distance = (left[:length] - right[:length]).norm(dim=-1).float().cpu().numpy()
    threshold = max(1e-3, 0.05 * float(distance.max()))
    return [int(index) for index in np.where(distance > threshold)[0]]


def safe_case_id(value: object) -> str:
    case_id = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
        raise ValueError(f"unsafe case ID {case_id!r}")
    return case_id


def load_registered_cases() -> pd.DataFrame:
    source = pd.read_parquet(BASE / "results/exp244_routing/cohort_frozen.parquet")
    source["cid12"] = source["cid12"].astype(str)
    # Preserve the canonical 12-character cohort ID as `id`; the source table's
    # original full ID is metadata and must not trigger pandas' id_x/id_y suffixes.
    source = source.rename(columns={"id": "source_id"})
    cohort = pd.read_csv(
        BASE / "results/exp251_best_fixed_site/test_cohort_frozen.csv",
        dtype={"id": str},
    )
    cohort = cohort.loc[cohort["model"].eq("qwen")].copy()
    if len(cohort) != 235 or cohort["id"].nunique() != 235:
        raise AssertionError("expected 235 unique Qwen cases")
    if source["cid12"].duplicated().any():
        raise AssertionError("Exp244 source IDs are not unique")
    merged = cohort.merge(source, left_on="id", right_on="cid12", how="left", validate="one_to_one")
    required = ["id", "prompt_stereotype", "prompt_anti_stereotype", "prompt_neutral", "triplet_hash"]
    if merged[required].isna().any().any():
        raise AssertionError("missing registered prompt text")
    if merged["id"].nunique() != 235:
        raise AssertionError("registered case join changed canonical IDs")
    return merged.sort_values(["triplet_hash", "id"]).reset_index(drop=True)


def load_direction_cache() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    path = OUT / "directions_235.npz"
    with np.load(path, allow_pickle=False) as data:
        ids = [str(value) for value in data["ids"]]
        directions = np.asarray(data["directions"], dtype=np.float32)
        mu = np.asarray(data["mu"], dtype=np.float32)
        norms = np.asarray(data["direction_norms"], dtype=np.float32)
    if len(ids) != 235 or directions.shape != (235, 29, 3584):
        raise AssertionError("invalid prepared direction cache")
    if mu.shape != (29, 3584) or norms.shape != (235, 29):
        raise AssertionError("invalid mu/direction-norm cache")
    if not np.isfinite(directions).all() or not np.isfinite(mu).all() or not np.isfinite(norms).all():
        raise AssertionError("non-finite direction cache")
    return ids, directions, mu, norms


def driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    except Exception:
        return "unavailable"


def encode(pipe, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # Public encode_prompt intentionally collapses an all-ones Qwen mask to
        # None in diffusers 0.37.1. The frozen DiT path retains the native mask.
        pe, pm = pipe._get_qwen_prompt_embeds(prompt=prompt, device=pipe._execution_device)
    if pe.ndim != 3 or pm is None or pm.ndim != 2 or pe.shape[:2] != pm.shape:
        raise AssertionError(
            f"invalid native Qwen prompt encoding: pe={tuple(pe.shape)}, "
            f"pm={None if pm is None else tuple(pm.shape)}"
        )
    return pe.detach(), pm.detach()


def build_output_delta(pipe, case: pd.Series, pe_n: torch.Tensor) -> tuple[torch.Tensor, float]:
    pe_s, _ = encode(pipe, str(case.prompt_stereotype))
    pe_a, _ = encode(pipe, str(case.prompt_anti_stereotype))
    changed = diff_rows(pe_s[0], pe_a[0])
    if not changed:
        raise AssertionError(f"no changed output rows for {case.id}")
    length = min(pe_s.shape[1], pe_a.shape[1])
    distances = (pe_s[0, :length] - pe_a[0, :length]).norm(dim=-1).float().cpu().numpy()
    top = sorted(changed, key=lambda index: -distances[index])[:2]
    direction = torch.stack([(pe_a[0][index] - pe_s[0][index]).float() for index in top]).mean(0)

    candidates: list[str] = []
    for key in ("head", "target"):
        value = case.get(key)
        if isinstance(value, str):
            candidates.extend(word for word in re.findall(r"[A-Za-z]+", value) if len(word) > 2)
    candidates.extend(GENERIC)
    anchor_rows: list[int] = []
    neutral = str(case.prompt_neutral)
    for word in candidates:
        if not re.search(rf"\b{re.escape(word)}\b", neutral, flags=re.I):
            continue
        replacement, _ = encode(
            pipe,
            re.sub(rf"\b{re.escape(word)}\b", "thing", neutral, count=1, flags=re.I),
        )
        anchor_rows = diff_rows(pe_n[0], replacement[0])
        if anchor_rows:
            break
    if not anchor_rows:
        raise AssertionError(f"no output anchor rows for {case.id}")
    edited = pe_n.clone()
    for column in anchor_rows[:2]:
        if column < edited.shape[1]:
            edited[0, column] = edited[0, column] + ALPHA_OUTPUT * direction.to(edited.dtype)
    return (edited - pe_n).detach(), float(direction.norm())


def velocity(
    transformer,
    z: torch.Tensor,
    timestep: torch.Tensor,
    pe: torch.Tensor,
    pm: torch.Tensor,
    img_shapes,
    *,
    context: str,
) -> torch.Tensor:
    ts = timestep.expand(z.shape[0]).to(z.dtype)
    with transformer.cache_context(context):
        value = transformer(
            hidden_states=z,
            timestep=ts / 1000,
            guidance=None,
            encoder_hidden_states_mask=pm,
            encoder_hidden_states=pe,
            img_shapes=img_shapes,
            attention_kwargs={},
            return_dict=False,
        )[0]
    return value[0]


@torch.no_grad()
def trajectory(pipe, pe_n: torch.Tensor, pm_n: torch.Tensor, pe_u: torch.Tensor, pm_u: torch.Tensor, seed: int):
    transformer, scheduler = pipe.transformer, pipe.scheduler
    latents, timesteps, img_shapes, _, _ = DBO._prep(pipe, seed, NUM_STEPS)
    scheduler.set_begin_index(0)
    keep: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, timestep in enumerate(timesteps):
        if index in TIMESTEPS:
            keep[index] = (latents.clone(), timestep)
        if index > max(TIMESTEPS):
            break
        ts = timestep.expand(latents.shape[0]).to(latents.dtype)
        with transformer.cache_context("cond"):
            cond = transformer(
                hidden_states=latents,
                timestep=ts / 1000,
                guidance=None,
                encoder_hidden_states_mask=pm_n,
                encoder_hidden_states=pe_n,
                img_shapes=img_shapes,
                attention_kwargs={},
                return_dict=False,
            )[0]
        with transformer.cache_context("uncond"):
            unc = transformer(
                hidden_states=latents,
                timestep=ts / 1000,
                guidance=None,
                encoder_hidden_states_mask=pm_u,
                encoder_hidden_states=pe_u,
                img_shapes=img_shapes,
                attention_kwargs={},
                return_dict=False,
            )[0]
        combined = unc + CFG * (cond - unc)
        prediction = combined * (cdg_loop._per_token_norm(cond) / cdg_loop._per_token_norm(combined))
        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    if set(keep) != set(TIMESTEPS):
        raise AssertionError("trajectory did not capture both registered timesteps")
    return keep, img_shapes


def compute_cell_gradient(
    transformer,
    z: torch.Tensor,
    timestep: torch.Tensor,
    img_shapes,
    pe_n: torch.Tensor,
    pm_n: torch.Tensor,
    pe_s: torch.Tensor,
    pm_s: torch.Tensor,
    pe_a: torch.Tensor,
    pm_a: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    with torch.no_grad():
        target = (
            velocity(transformer, z, timestep, pe_a, pm_a, img_shapes, context=f"{label}_anti")
            - velocity(transformer, z, timestep, pe_s, pm_s, img_shapes, context=f"{label}_stereo")
        ).float().flatten()
        target_norm = float(target.norm())
        if not np.isfinite(target_norm) or target_norm <= 1e-8:
            raise AssertionError(f"invalid target norm for {label}")
        target_hat = target / target_norm

    pe_leaf = pe_n.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        value = velocity(
            transformer,
            z,
            timestep,
            pe_leaf,
            pm_n,
            img_shapes,
            context=f"{label}_grad",
        ).float().flatten()
        objective = torch.dot(value, target_hat)
        gradient = torch.autograd.grad(objective, pe_leaf, retain_graph=False, create_graph=False)[0]
    gradient = gradient.detach().float().cpu()
    if not torch.isfinite(gradient).all():
        raise AssertionError(f"non-finite upstream gradient for {label}")
    del pe_leaf, value, objective
    return gradient, target_hat.detach(), target_norm


def save_checkpoint(
    root: Path,
    case_id: str,
    gradient: torch.Tensor,
    metadata: dict[str, object],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / f"{case_id}.safetensors"
    json_path = root / f"{case_id}.json"
    tensor_tmp = tensor_path.with_suffix(".safetensors.tmp")
    save_file({"g_out": gradient.contiguous().cpu()}, str(tensor_tmp))
    os.replace(tensor_tmp, tensor_path)
    metadata = dict(metadata)
    metadata["checkpoint_sha256"] = __import__("hashlib").sha256(tensor_path.read_bytes()).hexdigest()
    atomic_json(metadata, json_path)


def load_checkpoint(root: Path, case_id: str, expected_cells: int) -> tuple[torch.Tensor, dict[str, object]]:
    tensor_path = root / f"{case_id}.safetensors"
    json_path = root / f"{case_id}.json"
    if not tensor_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(case_id)
    metadata = json.loads(json_path.read_text())
    observed_hash = __import__("hashlib").sha256(tensor_path.read_bytes()).hexdigest()
    if observed_hash != metadata.get("checkpoint_sha256"):
        raise AssertionError(f"checkpoint hash mismatch for {case_id}")
    if int(metadata["n_cells"]) != expected_cells or metadata["case_id"] != case_id:
        raise AssertionError(f"checkpoint metadata mismatch for {case_id}")
    gradient = load_file(str(tensor_path), device="cpu")["g_out"].float()
    if list(gradient.shape) != list(metadata["gradient_shape"]) or not torch.isfinite(gradient).all():
        raise AssertionError(f"invalid checkpoint tensor for {case_id}")
    return gradient, metadata


def compute_upstream_case(
    pipe,
    case: pd.Series,
    checkpoint_root: Path,
    *,
    smoke: bool,
) -> tuple[dict[str, object], dict[str, object] | None]:
    case_id = safe_case_id(case.id)
    expected_cells = 1 if smoke else len(TRAJECTORY_SEEDS) * len(TIMESTEPS)
    try:
        _, metadata = load_checkpoint(checkpoint_root, case_id, expected_cells)
        print(f"[Exp256 upstream] reuse {case_id}", flush=True)
        return metadata, None
    except FileNotFoundError:
        pass

    started = time.time()
    pe_n, pm_n = encode(pipe, str(case.prompt_neutral))
    pe_s, pm_s = encode(pipe, str(case.prompt_stereotype))
    pe_a, pm_a = encode(pipe, str(case.prompt_anti_stereotype))
    encodings = {"neutral": pe_n, "stereo": pe_s, "anti": pe_a}
    if any(pe.shape[0] != 1 or pe.shape[-1] != pe_n.shape[-1] for pe in encodings.values()):
        raise AssertionError(
            f"conditioning batch/hidden mismatch for {case_id}: "
            f"{ {name: tuple(pe.shape) for name, pe in encodings.items()} }"
        )
    output_delta, output_direction_norm = build_output_delta(pipe, case, pe_n)
    pe_u, pm_u = encode(pipe, " ")

    seeds = (0,) if smoke else TRAJECTORY_SEEDS
    time_indices = (10,) if smoke else TIMESTEPS
    gradients = []
    cell_rows = []
    smoke_context = None
    for seed in seeds:
        keep, img_shapes = trajectory(pipe, pe_n, pm_n, pe_u, pm_u, seed)
        for timestep_index in time_indices:
            z, timestep = keep[timestep_index]
            label = f"{case_id}_s{seed}_t{timestep_index}"
            gradient, target_hat, target_norm = compute_cell_gradient(
                pipe.transformer,
                z,
                timestep,
                img_shapes,
                pe_n,
                pm_n,
                pe_s,
                pm_s,
                pe_a,
                pm_a,
                label,
            )
            gradients.append(gradient)
            cell_rows.append(
                {
                    "probe_seed": seed,
                    "t_idx": timestep_index,
                    "target_norm": target_norm,
                    "gradient_norm": float(gradient.norm()),
                }
            )
            if smoke:
                smoke_context = {
                    "z": z.detach(),
                    "timestep": timestep,
                    "img_shapes": img_shapes,
                    "target_hat": target_hat.detach(),
                    "pe_n": pe_n,
                    "pm_n": pm_n,
                    "output_delta": output_delta,
                }
    gradient_mean = torch.stack(gradients).mean(dim=0)
    output_score = float(torch.dot(gradient_mean.flatten(), output_delta.detach().float().cpu().flatten()))
    metadata = {
        "case_id": case_id,
        "triplet_hash": str(case.triplet_hash),
        "n_cells": len(cell_rows),
        "cells": cell_rows,
        "gradient_shape": list(gradient_mean.shape),
        "gradient_dtype": str(gradient_mean.dtype),
        "gradient_norm": float(gradient_mean.norm()),
        "neutral_embed_sha256": tensor_sha256(pe_n),
        "output_score": output_score,
        "output_delta_norm": float(output_delta.float().norm()),
        "output_direction_norm": output_direction_norm,
        "elapsed_seconds": time.time() - started,
    }
    save_checkpoint(checkpoint_root, case_id, gradient_mean, metadata)
    del pe_s, pe_a, pe_u, gradients, gradient_mean
    torch.cuda.empty_cache()
    return metadata, smoke_context


def capture_clean_layers(pipe, prompt: str, layers: tuple[int, ...]) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    result = ER.clean_forward(pipe.text_encoder, pipe.tokenizer, prompt, record_layers=layers, device="cuda")
    pe, pm = ER.post_process_for_dit(
        result["last_hidden_state"], result["attention_mask"], target_dtype=torch.bfloat16
    )
    return result["recorded"], pe, pm


@torch.no_grad()
def scalar_for_embed(pipe, context: dict[str, object], pe: torch.Tensor, pm: torch.Tensor, label: str) -> float:
    value = velocity(
        pipe.transformer,
        context["z"],
        context["timestep"],
        pe,
        pm,
        context["img_shapes"],
        context=label,
    ).float().flatten()
    return float(torch.dot(value, context["target_hat"]))


def smoke_finite_differences(
    pipe,
    case: pd.Series,
    context: dict[str, object],
    directions: np.ndarray,
    mu: np.ndarray,
) -> dict[str, object]:
    records, pe_clean, pm_clean = capture_clean_layers(pipe, str(case.prompt_neutral), (17, 22, 28))
    if tensor_sha256(pe_clean) != tensor_sha256(context["pe_n"]):
        raise AssertionError("manual encoder post-processing does not match pipeline encoding")

    diagnostics: dict[str, object] = {}
    output_delta = context["output_delta"]
    diagnostics["out"] = {
        "projection_exact": True,
        "finite_differences": {
            str(epsilon): (
                scalar_for_embed(
                    pipe,
                    context,
                    context["pe_n"] + epsilon * output_delta,
                    context["pm_n"],
                    f"smoke_out_plus_{epsilon}",
                )
                - scalar_for_embed(
                    pipe,
                    context,
                    context["pe_n"] - epsilon * output_delta,
                    context["pm_n"],
                    f"smoke_out_minus_{epsilon}",
                )
            )
            / (2 * epsilon)
            for epsilon in EPSILONS
        },
    }

    inner = ER._text_decoder(pipe.text_encoder)
    encoded = ER.tokenize_wrapped(pipe.tokenizer, str(case.prompt_neutral), device="cuda")
    patch_mask = torch.ones(encoded["input_ids"].shape, dtype=torch.bool, device="cuda")
    for layer in (17, 22, 28):
        hidden = records[layer]
        direction = torch.from_numpy(np.ascontiguousarray(directions[layer]))
        center = torch.from_numpy(np.ascontiguousarray(mu[layer]))
        expected = apply_centered_projection(hidden, direction, center, patch_mask=patch_mask)
        patcher = ProjectionPatcher()
        patcher.mode = "project"
        patcher.direction = direction
        patcher.mu = center
        patcher.alpha = 2.0
        patcher.patch_mask = patch_mask.cpu()
        actual = patcher._modify(hidden)
        projection_exact = bool(torch.equal(expected, actual))
        delta = expected - hidden
        fd = {}
        for epsilon in EPSILONS:
            plus = ER.patched_forward(
                pipe.text_encoder,
                pipe.tokenizer,
                str(case.prompt_neutral),
                layer,
                hidden + epsilon * delta,
                patch_mask,
                device="cuda",
            )
            minus = ER.patched_forward(
                pipe.text_encoder,
                pipe.tokenizer,
                str(case.prompt_neutral),
                layer,
                hidden - epsilon * delta,
                patch_mask,
                device="cuda",
            )
            pe_plus, pm_plus = ER.post_process_for_dit(
                plus["last_hidden_state"], plus["attention_mask"], target_dtype=torch.bfloat16
            )
            pe_minus, pm_minus = ER.post_process_for_dit(
                minus["last_hidden_state"], minus["attention_mask"], target_dtype=torch.bfloat16
            )
            fd[str(epsilon)] = (
                scalar_for_embed(pipe, context, pe_plus, pm_plus, f"smoke_l{layer}_plus_{epsilon}")
                - scalar_for_embed(pipe, context, pe_minus, pm_minus, f"smoke_l{layer}_minus_{epsilon}")
            ) / (2 * epsilon)
        diagnostics[str(layer)] = {
            "projection_exact": projection_exact,
            "finite_differences": fd,
        }
    return diagnostics


def release_dit(pipe) -> None:
    transformer = pipe.transformer
    vae = getattr(pipe, "vae", None)
    pipe.transformer = None
    if hasattr(pipe, "vae"):
        pipe.vae = None
    del transformer, vae
    gc.collect()
    torch.cuda.empty_cache()


def encoder_vjp_scores(
    pipe,
    case: pd.Series,
    gradient_cpu: torch.Tensor,
    metadata: dict[str, object],
    directions: np.ndarray,
    mu: np.ndarray,
    direction_norms: np.ndarray,
    shard: int,
) -> list[dict[str, object]]:
    case_id = safe_case_id(case.id)
    inner = ER._text_decoder(pipe.text_encoder)
    pipe.text_encoder.requires_grad_(False)
    encoded = ER.tokenize_wrapped(pipe.tokenizer, str(case.prompt_neutral), device="cuda")
    with torch.no_grad():
        embedding_leaf = inner.embed_tokens(encoded["input_ids"]).detach()
    embedding_leaf.requires_grad_(True)

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(layer: int):
        def hook(_module, _args, output):
            captured[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    for layer in ALL_INTERNAL_LAYERS:
        handles.append(inner.layers[layer - 1].register_forward_hook(hook_for(layer)))
    try:
        with torch.enable_grad():
            output = inner(
                inputs_embeds=embedding_leaf,
                attention_mask=encoded["attention_mask"],
                output_hidden_states=False,
                use_cache=False,
            )
            pe_graph, _ = ER.post_process_for_dit(
                output.last_hidden_state,
                encoded["attention_mask"],
                target_dtype=torch.bfloat16,
            )
            if tensor_sha256(pe_graph) != metadata["neutral_embed_sha256"]:
                raise AssertionError(f"neutral encoder reproduction mismatch for {case_id}")
            if tuple(pe_graph.shape) != tuple(gradient_cpu.shape):
                raise AssertionError(f"upstream gradient shape mismatch for {case_id}")
            objective = (pe_graph.float() * gradient_cpu.to("cuda", dtype=torch.float32)).sum()
            hidden_tuple = tuple(captured[layer] for layer in ALL_INTERNAL_LAYERS)
            gradients = torch.autograd.grad(
                objective,
                hidden_tuple,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    rows: list[dict[str, object]] = [
        {
            "id": case_id,
            "site": "out",
            "score": float(metadata["output_score"]),
            "rank_internal": np.nan,
            "grad_norm": float(metadata["gradient_norm"]),
            "delta_norm": float(metadata["output_delta_norm"]),
            "direction_norm": float(metadata["output_direction_norm"]),
            "n_cells": int(metadata["n_cells"]),
            "finite": True,
            "zero_direction": False,
            "shard": int(shard),
        }
    ]
    internal_rows = []
    for layer, gradient in zip(ALL_INTERNAL_LAYERS, gradients):
        hidden = captured[layer].detach()
        direction = torch.from_numpy(np.ascontiguousarray(directions[layer]))
        center = torch.from_numpy(np.ascontiguousarray(mu[layer]))
        patched = apply_centered_projection(hidden, direction, center)
        delta = patched - hidden
        score = float(torch.dot(gradient.detach().float().flatten(), delta.float().flatten()))
        grad_norm = float(gradient.detach().float().norm())
        delta_norm = float(delta.float().norm())
        finite = bool(np.isfinite([score, grad_norm, delta_norm]).all())
        if not finite:
            raise AssertionError(f"non-finite encoder score for {case_id} L{layer}")
        if str(layer) in INTERNAL_SITES and delta_norm <= 0:
            raise AssertionError(f"registered candidate has zero displacement for {case_id} L{layer}")
        internal_rows.append(
            {
                "id": case_id,
                "site": str(layer),
                "score": score,
                "rank_internal": 0,
                "grad_norm": grad_norm,
                "delta_norm": delta_norm,
                "direction_norm": float(direction_norms[layer]),
                "n_cells": int(metadata["n_cells"]),
                "finite": finite,
                "zero_direction": bool(direction_norms[layer] <= 1e-8),
                "shard": int(shard),
            }
        )
    ordered = sorted(internal_rows, key=lambda row: (-float(row["score"]), int(row["site"])))
    for rank, row in enumerate(ordered, start=1):
        row["rank_internal"] = rank
    rows.extend(internal_rows)
    if any(parameter.grad is not None for parameter in pipe.text_encoder.parameters()):
        raise AssertionError("text-encoder parameter gradient was populated")
    del output, pe_graph, objective, gradients, captured, embedding_leaf
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def evaluate_smoke(
    rows: list[dict[str, object]],
    diagnostics: dict[str, object],
    peak_bytes: int,
) -> dict[str, object]:
    lookup = {str(row["site"]): row for row in rows}
    checks = {}
    for site in ("out", "17", "22", "28"):
        score = float(lookup[site]["score"])
        scale = max(float(lookup[site]["grad_norm"]) * float(lookup[site]["delta_norm"]), 1e-12)
        candidates = []
        for epsilon, estimate in diagnostics[site]["finite_differences"].items():
            estimate = float(estimate)
            error = abs(estimate - score) / scale
            sign_required = abs(score) / scale > 1e-4
            sign_match = (not sign_required) or np.sign(estimate) == np.sign(score)
            candidates.append(
                {
                    "epsilon": float(epsilon),
                    "estimate": estimate,
                    "score": score,
                    "normalized_error": error,
                    "sign_required": sign_required,
                    "sign_match": bool(sign_match),
                    "pass": bool(error <= 0.05 and sign_match),
                }
            )
        checks[site] = {
            "projection_exact": bool(diagnostics[site]["projection_exact"]),
            "finite_difference": candidates,
            "pass": bool(diagnostics[site]["projection_exact"] and any(item["pass"] for item in candidates)),
        }
    memory_pass = peak_bytes < MAX_PEAK_BYTES
    pass_all = bool(all(check["pass"] for check in checks.values()) and memory_pass)
    return {
        "status": "PASS" if pass_all else "FAIL",
        "site_checks": checks,
        "peak_allocated_bytes": int(peak_bytes),
        "peak_limit_bytes": int(MAX_PEAK_BYTES),
        "memory_pass": memory_pass,
        "parameter_grads_unset": True,
    }


def main(mode: str, shard: int, nshards: int) -> None:
    if mode not in {"smoke", "cohort"}:
        raise ValueError(mode)
    if mode == "cohort" and nshards != N_SHARDS:
        raise ValueError(f"cohort execution requires nshards={N_SHARDS}")
    if not 0 <= shard < nshards:
        raise ValueError("invalid shard")

    cases = load_registered_cases()
    cache_ids, all_directions, mu, all_direction_norms = load_direction_cache()
    if cache_ids != cases["id"].astype(str).tolist():
        raise AssertionError("direction-cache order differs from frozen cohort order")
    if mode == "smoke":
        selected = cases.iloc[[0]].copy()
        shard = 0
        nshards = 1
    else:
        selected = cases.iloc[shard::nshards].copy()
    if selected.empty:
        raise AssertionError("empty gradient shard")

    run_root = OUT / mode
    checkpoint_root = run_root / "upstream"
    score_root = run_root / "score_shards"
    score_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    pipe = load_pipe()
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.eval()
    pipe.text_encoder.eval()

    metadata_by_id: dict[str, dict[str, object]] = {}
    smoke_context = None
    smoke_diagnostics = None
    for position, (_, case) in enumerate(selected.iterrows(), start=1):
        metadata, context = compute_upstream_case(
            pipe,
            case,
            checkpoint_root,
            smoke=mode == "smoke",
        )
        metadata_by_id[str(case.id)] = metadata
        if mode == "smoke":
            smoke_context = context
            case_index = cache_ids.index(str(case.id))
            smoke_diagnostics = smoke_finite_differences(
                pipe,
                case,
                smoke_context,
                all_directions[case_index],
                mu,
            )
        print(f"[Exp256 upstream {mode} shard={shard}] {position}/{len(selected)} {case.id}", flush=True)

    if any(parameter.grad is not None for parameter in pipe.transformer.parameters()):
        raise AssertionError("DiT parameter gradient was populated")
    release_dit(pipe)

    rows: list[dict[str, object]] = []
    for position, (_, case) in enumerate(selected.iterrows(), start=1):
        case_id = str(case.id)
        expected_cells = 1 if mode == "smoke" else 6
        gradient, metadata = load_checkpoint(checkpoint_root, case_id, expected_cells)
        case_index = cache_ids.index(case_id)
        rows.extend(
            encoder_vjp_scores(
                pipe,
                case,
                gradient,
                metadata,
                all_directions[case_index],
                mu,
                all_direction_norms[case_index],
                shard,
            )
        )
        print(f"[Exp256 encoder {mode} shard={shard}] {position}/{len(selected)} {case_id}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["id", "site"]).reset_index(drop=True)
    expected_rows = len(selected) * 29
    if len(frame) != expected_rows or frame[["id", "site"]].duplicated().any():
        raise AssertionError(f"score rows {len(frame)} != {expected_rows}")
    output_path = score_root / ("scores_smoke.parquet" if mode == "smoke" else f"scores_shard{shard}.parquet")
    atomic_parquet(frame, output_path)

    peak_bytes = int(torch.cuda.max_memory_allocated())
    summary = {
        "mode": mode,
        "shard": shard,
        "nshards": nshards,
        "n_cases": len(selected),
        "n_rows": len(frame),
        "case_ids": selected["id"].astype(str).tolist(),
        "elapsed_seconds": time.time() - started,
        "peak_allocated_bytes": peak_bytes,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
        "driver_version": driver_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "score_path": str(output_path),
    }
    if mode == "smoke":
        if smoke_diagnostics is None:
            raise AssertionError("smoke diagnostics missing")
        summary["engineering_gate"] = evaluate_smoke(rows, smoke_diagnostics, peak_bytes)
        atomic_json(summary, run_root / "smoke_summary.json")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["engineering_gate"]["status"] != "PASS":
            raise SystemExit(2)
    else:
        atomic_json(summary, score_root / f"summary_shard{shard}.json")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "cohort"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=N_SHARDS)
    arguments = parser.parse_args()
    main(arguments.mode, arguments.shard, arguments.nshards)
