#!/usr/bin/env python3
"""All-layer exact-additive backward S screen for Exp264B."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from exp264b_common import (
    ALPHA,
    BASE,
    LAYERS,
    N_ANCHORS,
    OUT,
    TIMESTEPS,
    TRAJECTORY_SEEDS,
    atomic_json,
    atomic_parquet,
    bool_mask,
    choose_random_layer,
    load_cases,
    native_add_delta,
    output_rows_for_pole,
    parse_int_list,
    pole_diff_rows,
    sha256_file,
    sha256_tensor,
    unit_direction,
    validate_case_coordinates,
)

sys.path.insert(0, str(BASE))

N_SHARDS = 12


def safe_id(value: object) -> str:
    import re

    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"unsafe full_id {text!r}")
    return text


def direction_bundle(pipe, row: pd.Series):
    from experiments.causal_patching import encoder_runner as ER

    stereo = str(row.prompt_stereotype)
    anti = str(row.prompt_anti_stereotype)
    s = ER.clean_forward(pipe.text_encoder, pipe.tokenizer, stereo, record_layers=LAYERS)
    a = ER.clean_forward(pipe.text_encoder, pipe.tokenizer, anti, record_layers=LAYERS)
    pe_s, pm_s = ER.post_process_for_dit(
        s["last_hidden_state"], s["attention_mask"], target_dtype=torch.bfloat16
    )
    pe_a, pm_a = ER.post_process_for_dit(
        a["last_hidden_state"], a["attention_mask"], target_dtype=torch.bfloat16
    )
    if pm_s.shape != pe_s.shape[:2] or pm_a.shape != pe_a.shape[:2]:
        raise AssertionError("invalid pole conditioning mask")
    d_s_wrapped, d_a_wrapped = pole_diff_rows(pipe.tokenizer, stereo, anti)
    d_s_output, d_a_output = output_rows_for_pole(pipe.tokenizer, stereo, anti)
    directions = np.empty((29, pe_s.shape[-1]), dtype=np.float32)
    norms = np.empty(29, dtype=np.float32)
    out_direction, norms[0] = unit_direction(pe_s, d_s_output, pe_a, d_a_output)
    directions[0] = out_direction.detach().cpu().numpy()
    for layer in LAYERS:
        direction, norms[layer] = unit_direction(
            s["recorded"][layer], d_s_wrapped, a["recorded"][layer], d_a_wrapped
        )
        directions[layer] = direction.detach().cpu().numpy()
    if not np.isfinite(directions).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise AssertionError("invalid direction bundle")
    return directions, norms, pe_s, pm_s, pe_a, pm_a, d_s_wrapped, d_a_wrapped


def save_upstream(case_id: str, gradient: torch.Tensor, directions: np.ndarray, norms: np.ndarray, metadata):
    root = OUT / "s_screen" / "upstream"
    root.mkdir(parents=True, exist_ok=True)
    gradient_path = root / f"{case_id}.safetensors"
    direction_path = root / f"{case_id}.directions.npz"
    metadata_path = root / f"{case_id}.json"
    gradient_tmp = gradient_path.with_suffix(".safetensors.tmp")
    save_file({"g_out": gradient.detach().contiguous().cpu().float()}, str(gradient_tmp))
    os.replace(gradient_tmp, gradient_path)
    direction_tmp = direction_path.with_suffix(".npz.tmp")
    with direction_tmp.open("wb") as handle:
        np.savez(handle, directions=directions.astype(np.float32), raw_norms=norms.astype(np.float32))
    os.replace(direction_tmp, direction_path)
    payload = dict(metadata)
    payload["gradient_sha256"] = sha256_file(gradient_path)
    payload["direction_sha256"] = sha256_file(direction_path)
    atomic_json(payload, metadata_path)


def load_upstream(case_id: str):
    root = OUT / "s_screen" / "upstream"
    gradient_path = root / f"{case_id}.safetensors"
    direction_path = root / f"{case_id}.directions.npz"
    metadata_path = root / f"{case_id}.json"
    if not all(path.is_file() for path in (gradient_path, direction_path, metadata_path)):
        raise FileNotFoundError(case_id)
    metadata = json.loads(metadata_path.read_text())
    if metadata["gradient_sha256"] != sha256_file(gradient_path):
        raise AssertionError("upstream gradient hash mismatch")
    if metadata["direction_sha256"] != sha256_file(direction_path):
        raise AssertionError("upstream direction hash mismatch")
    gradient = load_file(str(gradient_path), device="cpu")["g_out"].float()
    with np.load(direction_path, allow_pickle=False) as bundle:
        directions = np.asarray(bundle["directions"], dtype=np.float32)
        norms = np.asarray(bundle["raw_norms"], dtype=np.float32)
    if directions.shape != (29, 3584) or norms.shape != (29,):
        raise AssertionError("upstream direction shape mismatch")
    if tuple(gradient.shape) != tuple(metadata["gradient_shape"]):
        raise AssertionError("upstream gradient shape mismatch")
    return gradient, directions, norms, metadata


def compute_upstream(pipe, core, row: pd.Series) -> None:
    case_id = safe_id(row.full_id)
    try:
        gradient, directions, norms, metadata = load_upstream(case_id)
        if metadata["triplet_key"] != str(row.triplet_key) or int(metadata["n_cells"]) != 6:
            raise AssertionError("upstream metadata mismatch")
        print(f"[exp264b-S] reuse upstream {case_id}", flush=True)
        del gradient, directions, norms
        return
    except FileNotFoundError:
        pass
    wrapped_rows, output_rows = validate_case_coordinates(row)
    directions, norms, pe_s, pm_s, pe_a, pm_a, d_s, d_a = direction_bundle(pipe, row)
    pe_n, pm_n = core.encode(pipe, str(row.prompt_neutral))
    output_mask = bool_mask(pe_n.shape[1], output_rows, device="cuda")
    _, output_delta = native_add_delta(
        pe_n, torch.from_numpy(directions[0]).to("cuda"), output_mask
    )
    pe_u, pm_u = core.encode(pipe, " ")
    gradients = []
    cells = []
    started = time.time()
    for seed in TRAJECTORY_SEEDS:
        keep, img_shapes = core.trajectory(pipe, pe_n, pm_n, pe_u, pm_u, seed)
        for t_idx in TIMESTEPS:
            z, timestep = keep[t_idx]
            gradient, _target_hat, target_norm = core.compute_cell_gradient(
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
                f"exp264b_{case_id}_s{seed}_t{t_idx}",
            )
            gradients.append(gradient)
            cells.append(
                {
                    "probe_seed": seed,
                    "t_idx": t_idx,
                    "gradient_norm": float(gradient.norm()),
                    "target_norm": float(target_norm),
                }
            )
        del keep
    gradient_mean = torch.stack(gradients).mean(dim=0)
    metadata = {
        "case_id": case_id,
        "triplet_key": str(row.triplet_key),
        "n_cells": len(cells),
        "cells": cells,
        "gradient_shape": list(gradient_mean.shape),
        "gradient_norm": float(gradient_mean.norm()),
        "neutral_conditioning_sha256": sha256_tensor(pe_n),
        "output_delta_norm": float(output_delta.float().norm()),
        "wrapped_anchor_rows": wrapped_rows,
        "output_anchor_rows": output_rows,
        "pole_diff_wrapped_stereo": d_s,
        "pole_diff_wrapped_anti": d_a,
        "elapsed_seconds": time.time() - started,
    }
    save_upstream(case_id, gradient_mean, directions, norms, metadata)
    del gradients, gradient_mean, pe_s, pe_a, pe_n, pe_u
    gc.collect()
    torch.cuda.empty_cache()


def internal_conditioning(pipe, prompt: str, layer: int, direction: np.ndarray, wrapped_rows: list[int]):
    from experiments.causal_patching import encoder_runner as ER
    from experiments.causal_patching.projection_patcher import ProjectionPatcher

    inner = ER._text_decoder(pipe.text_encoder)
    encoded = ER.tokenize_wrapped(pipe.tokenizer, prompt, device="cuda")
    mask = bool_mask(encoded["input_ids"].shape[1], wrapped_rows)
    with ProjectionPatcher() as patcher:
        patcher.mode = "add"
        patcher.direction = torch.from_numpy(np.ascontiguousarray(direction))
        patcher.alpha = ALPHA
        patcher.patch_mask = mask
        patcher.install(inner.layers[layer - 1])
        with torch.no_grad():
            result = inner(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=False,
                use_cache=False,
            )
    return ER.post_process_for_dit(
        result.last_hidden_state, encoded["attention_mask"], target_dtype=torch.bfloat16
    )


def score_case(pipe, row: pd.Series, shard: int):
    from experiments.causal_patching import encoder_runner as ER

    case_id = safe_id(row.full_id)
    gradient_cpu, directions, raw_norms, metadata = load_upstream(case_id)
    wrapped_rows, output_rows = validate_case_coordinates(row)
    inner = ER._text_decoder(pipe.text_encoder)
    encoded = ER.tokenize_wrapped(pipe.tokenizer, str(row.prompt_neutral), device="cuda")
    embedding_leaf = inner.embed_tokens(encoded["input_ids"]).detach().requires_grad_(True)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _args, output):
            captured[layer] = output[0] if isinstance(output, tuple) else output
        return hook

    for layer in LAYERS:
        handles.append(inner.layers[layer - 1].register_forward_hook(make_hook(layer)))
    try:
        with torch.enable_grad():
            result = inner(
                inputs_embeds=embedding_leaf,
                attention_mask=encoded["attention_mask"],
                output_hidden_states=False,
                use_cache=False,
            )
            pe_graph, pm_graph = ER.post_process_for_dit(
                result.last_hidden_state,
                encoded["attention_mask"],
                target_dtype=torch.bfloat16,
            )
            if sha256_tensor(pe_graph) != metadata["neutral_conditioning_sha256"]:
                raise AssertionError(f"neutral conditioning reproduction failed for {case_id}")
            objective = (pe_graph.float() * gradient_cpu.to("cuda").float()).sum()
            gradients = torch.autograd.grad(
                objective,
                tuple(captured[layer] for layer in LAYERS),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    output_mask = bool_mask(pe_graph.shape[1], output_rows, device="cuda")
    pe_output, delta_output = native_add_delta(
        pe_graph, torch.from_numpy(directions[0]).to("cuda"), output_mask
    )
    rows = [
        {
            "full_id": case_id,
            "triplet_key": str(row.triplet_key),
            "site": "A",
            "layer": 0,
            "s": float(torch.dot(gradient_cpu.flatten(), delta_output.detach().float().cpu().flatten())),
            "grad_norm": float(gradient_cpu.norm()),
            "delta_norm": float(delta_output.detach().float().norm()),
            "relative_edit_norm": float(
                delta_output.detach().float().norm()
                / max(float(pe_graph.detach()[0, output_rows].float().norm()), 1e-8)
            ),
            "raw_direction_norm": float(raw_norms[0]),
            "finite": True,
            "zero_direction": False,
            "shard": int(shard),
        }
    ]
    internal_rows = []
    wrapped_mask = bool_mask(encoded["input_ids"].shape[1], wrapped_rows, device="cuda")
    for layer, gradient in zip(LAYERS, gradients):
        _edited, delta = native_add_delta(
            captured[layer], torch.from_numpy(directions[layer]).to("cuda"), wrapped_mask
        )
        score = float(torch.dot(gradient.detach().float().flatten(), delta.detach().float().flatten()))
        delta_norm = float(delta.detach().float().norm())
        grad_norm = float(gradient.detach().float().norm())
        finite = bool(np.isfinite([score, delta_norm, grad_norm]).all())
        internal_rows.append(
            {
                "full_id": case_id,
                "triplet_key": str(row.triplet_key),
                "site": f"L{layer}",
                "layer": int(layer),
                "s": score,
                "grad_norm": grad_norm,
                "delta_norm": delta_norm,
                "relative_edit_norm": float(
                    delta_norm
                    / max(float(captured[layer].detach()[0, wrapped_rows].float().norm()), 1e-8)
                ),
                "raw_direction_norm": float(raw_norms[layer]),
                "finite": finite,
                "zero_direction": bool(raw_norms[layer] <= 1e-8 or delta_norm <= 0),
                "shard": int(shard),
            }
        )
    rows.extend(internal_rows)
    eligible = [item for item in internal_rows if item["finite"] and not item["zero_direction"]]
    if len(eligible) != len(LAYERS):
        raise AssertionError(f"not all internal sites eligible for {case_id}")
    backward = sorted(eligible, key=lambda item: (-float(item["s"]), int(item["layer"])))[0]
    backward_layer = int(backward["layer"])
    random_layer = choose_random_layer(str(row.triplet_key), [int(item["layer"]) for item in eligible])

    pe_clean = pe_graph.detach()
    site_diagnostics = {}
    for label, layer in (("B", backward_layer), ("R", random_layer)):
        edited, edited_mask = internal_conditioning(
            pipe, str(row.prompt_neutral), layer, directions[layer], wrapped_rows
        )
        if not torch.equal(edited_mask, pm_graph):
            raise AssertionError("internal output mask changed")
        delta_final = edited - pe_clean
        anchor_norm = float(delta_final[0, output_rows].float().norm())
        all_rows = list(range(delta_final.shape[1]))
        off_rows = [index for index in all_rows if index not in set(output_rows)]
        off_norm = float(delta_final[0, off_rows].float().norm()) if off_rows else 0.0
        total_norm = float(delta_final.float().norm())
        site_diagnostics[label] = {
            "layer": layer,
            "conditioning_sha256": sha256_tensor(edited),
            "final_delta_norm": total_norm,
            "final_anchor_norm": anchor_norm,
            "final_off_anchor_norm": off_norm,
            "spill_fraction": off_norm / max(total_norm, 1e-8),
        }
    candidate = {
        "full_id": case_id,
        "triplet_key": str(row.triplet_key),
        "backward_layer": backward_layer,
        "backward_s": float(backward["s"]),
        "random_layer": random_layer,
        "random_s": float(next(item["s"] for item in eligible if int(item["layer"]) == random_layer)),
        "output_s": float(rows[0]["s"]),
        "b_equals_r": bool(backward_layer == random_layer),
        "a_conditioning_sha256": sha256_tensor(pe_output.detach()),
        "b_conditioning_sha256": site_diagnostics["B"]["conditioning_sha256"],
        "r_conditioning_sha256": site_diagnostics["R"]["conditioning_sha256"],
        "b_spill_fraction": site_diagnostics["B"]["spill_fraction"],
        "r_spill_fraction": site_diagnostics["R"]["spill_fraction"],
        "direction_file": str(OUT / "s_screen" / "upstream" / f"{case_id}.directions.npz"),
        "direction_sha256": metadata["direction_sha256"],
        "shard": int(shard),
    }
    if any(parameter.grad is not None for parameter in pipe.text_encoder.parameters()):
        raise AssertionError("text-encoder parameter gradient populated")
    del result, pe_graph, objective, gradients, captured, embedding_leaf
    gc.collect()
    torch.cuda.empty_cache()
    return rows, candidate


def main(shard: int, nshards: int) -> None:
    import exp256_gradient_screen_qwen as core
    from experiments.causal_patching.run_three_methods import load_pipe

    if nshards != N_SHARDS or not 0 <= shard < nshards:
        raise ValueError(f"Exp264B S requires {N_SHARDS} shards")
    cases = load_cases(anchors_only=True)
    selected = cases.iloc[shard::nshards].copy()
    if selected.empty:
        raise AssertionError("empty S shard")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipe = load_pipe()
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        compute_upstream(pipe, core, row)
        print(f"[exp264b-S upstream {shard}] {position}/{len(selected)} {row.full_id}", flush=True)
    if any(parameter.grad is not None for parameter in pipe.transformer.parameters()):
        raise AssertionError("DiT parameter gradient populated")
    transformer = pipe.transformer
    vae = getattr(pipe, "vae", None)
    pipe.transformer = None
    if hasattr(pipe, "vae"):
        pipe.vae = None
    del transformer, vae
    gc.collect()
    torch.cuda.empty_cache()

    score_rows = []
    candidates = []
    for position, (_, row) in enumerate(selected.iterrows(), start=1):
        rows, candidate = score_case(pipe, row, shard)
        score_rows.extend(rows)
        candidates.append(candidate)
        print(
            f"[exp264b-S encoder {shard}] {position}/{len(selected)} {row.full_id} "
            f"B=L{candidate['backward_layer']} R=L{candidate['random_layer']}",
            flush=True,
        )
    score_frame = pd.DataFrame(score_rows).sort_values(["full_id", "layer"])
    candidate_frame = pd.DataFrame(candidates).sort_values("full_id")
    if len(score_frame) != len(selected) * 29 or len(candidate_frame) != len(selected):
        raise AssertionError("S shard rectangle mismatch")
    score_path = OUT / "s_screen" / "shards" / f"scores_shard{shard:02d}.parquet"
    candidate_path = OUT / "s_screen" / "shards" / f"candidates_shard{shard:02d}.parquet"
    atomic_parquet(score_frame, score_path)
    atomic_parquet(candidate_frame, candidate_path)
    summary = {
        "experiment": "Exp264B",
        "stage": "S",
        "shard": shard,
        "nshards": nshards,
        "n_cases": len(selected),
        "n_score_rows": len(score_frame),
        "elapsed_seconds": time.time() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
        "score_sha256": sha256_file(score_path),
        "candidate_sha256": sha256_file(candidate_path),
    }
    atomic_json(summary, OUT / "s_screen" / "shards" / f"summary_shard{shard:02d}.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=N_SHARDS)
    args = parser.parse_args()
    main(args.shard, args.nshards)
