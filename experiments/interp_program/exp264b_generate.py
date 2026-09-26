#!/usr/bin/env python3
"""Generate a checkpointed shard of every frozen Exp264B C/F/A/B/R arm."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exp264b_common import (
    ALPHA,
    BASE,
    IMAGE_SEEDS,
    NUM_STEPS,
    OUT,
    atomic_json,
    atomic_parquet,
    bool_mask,
    load_cases,
    native_add_delta,
    sha256_file,
    sha256_tensor,
    validate_case_coordinates,
)
from exp264b_screen import internal_conditioning, load_upstream

sys.path.insert(0, str(BASE))

N_SHARDS = 24


def full_output_edit(neutral: torch.Tensor, stereotype: torch.Tensor, anti: torch.Tensor):
    max_length = max(stereotype.shape[1], anti.shape[1])
    stereo = F.pad(stereotype.float(), (0, 0, 0, max_length - stereotype.shape[1]))
    anti_padded = F.pad(anti.float(), (0, 0, 0, max_length - anti.shape[1]))
    direction = anti_padded - stereo
    if direction.shape[1] < neutral.shape[1]:
        direction = F.pad(direction, (0, 0, 0, neutral.shape[1] - direction.shape[1]))
    else:
        direction = direction[:, : neutral.shape[1], :]
    direction = direction.to(device=neutral.device, dtype=neutral.dtype)
    edited = neutral + ALPHA * direction
    if direction.shape != neutral.shape or float(direction.float().norm()) <= 1e-8:
        raise AssertionError("invalid full-output displacement")
    return edited, direction


def main(shard: int, nshards: int) -> None:
    import exp256_gradient_screen_qwen as core
    from experiments.causal_patching.run_three_methods import load_pipe

    if nshards != N_SHARDS or not 0 <= shard < nshards:
        raise ValueError(f"Exp264B generation requires {N_SHARDS} shards")
    prepare_summary = json.loads((OUT / "generation_prepare_summary.json").read_text())
    manifest_path = OUT / "generation_manifest_frozen.parquet"
    if prepare_summary["manifest_sha256"] != sha256_file(manifest_path):
        raise AssertionError("generation manifest changed")
    manifest = pd.read_parquet(manifest_path)
    cases = load_cases(anchors_only=True)
    candidates = pd.read_parquet(OUT / "candidates_frozen.parquet")
    case_table = cases.merge(candidates, on=["full_id", "triplet_key"], validate="one_to_one")
    selected_cases = case_table.sort_values(["sample_rank", "full_id"]).iloc[shard::nshards].copy()
    if selected_cases.empty:
        raise AssertionError("empty generation shard")
    selected_ids = set(selected_cases["full_id"].astype(str))
    selected_manifest = manifest.loc[manifest["full_id"].astype(str).isin(selected_ids)].copy()
    if len(selected_manifest) != len(selected_cases) * 10:
        raise AssertionError("generation shard manifest mismatch")
    checkpoint = OUT / "generation_shards" / f"manifest_shard{shard:02d}.parquet"
    done_frame = pd.read_parquet(checkpoint) if checkpoint.is_file() else pd.DataFrame()
    done = set(done_frame["image_path"].astype(str)) if len(done_frame) else set()
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    torch.set_grad_enabled(False)
    pipe = load_pipe()
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    new_rows = []
    for position, case in enumerate(selected_cases.itertuples(index=False), start=1):
        full_id = str(case.full_id)
        _gradient, directions, _norms, _metadata = load_upstream(full_id)
        wrapped_rows, output_rows = validate_case_coordinates(case._asdict())
        pe_n, pm_n = core.encode(pipe, str(case.prompt_neutral))
        pe_s, _pm_s = core.encode(pipe, str(case.prompt_stereotype))
        pe_a, _pm_a = core.encode(pipe, str(case.prompt_anti_stereotype))
        pe_f, _full_direction = full_output_edit(pe_n, pe_s, pe_a)
        output_mask = bool_mask(pe_n.shape[1], output_rows, device="cuda")
        edit_a, _delta_a = native_add_delta(
            pe_n, torch.from_numpy(directions[0]).to("cuda"), output_mask
        )
        edit_b, mask_b = internal_conditioning(
            pipe, str(case.prompt_neutral), int(case.backward_layer),
            directions[int(case.backward_layer)], wrapped_rows,
        )
        edit_r, mask_r = internal_conditioning(
            pipe, str(case.prompt_neutral), int(case.random_layer),
            directions[int(case.random_layer)], wrapped_rows,
        )
        conditionings = {
            "C": (pe_n, pm_n),
            "F": (pe_f, pm_n),
            "A": (edit_a, pm_n),
            "B": (edit_b, mask_b),
            "R": (edit_r, mask_r),
        }
        expected = {
            "A": str(case.a_conditioning_sha256),
            "B": str(case.b_conditioning_sha256),
            "R": str(case.r_conditioning_sha256),
        }
        observed = {arm: sha256_tensor(conditionings[arm][0]) for arm in ("A", "B", "R")}
        if observed != expected:
            raise AssertionError(f"generation conditioning differs from S/N for {full_id}")
        case_manifest = selected_manifest.loc[selected_manifest["full_id"].astype(str).eq(full_id)]
        for item in case_manifest.sort_values(["arm", "seed"]).itertuples(index=False):
            image_path = Path(item.image_path)
            if str(image_path) in done:
                if not image_path.is_file() or image_path.stat().st_size <= 0:
                    raise AssertionError("checkpoint names absent image")
                continue
            image_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_embeds, prompt_mask = conditionings[str(item.arm)]
            image = pipe(
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_mask,
                num_inference_steps=NUM_STEPS,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                generator=torch.Generator("cuda").manual_seed(int(item.seed)),
            ).images[0]
            image.save(image_path)
            if image.size != (1024, 1024) or image_path.stat().st_size <= 0:
                raise AssertionError(f"invalid generated image {image_path}")
            record = item._asdict()
            record["conditioning_sha256"] = sha256_tensor(prompt_embeds)
            record["image_sha256"] = sha256_file(image_path)
            record["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "interactive")
            new_rows.append(record)
            combined = pd.concat([done_frame, pd.DataFrame(new_rows)], ignore_index=True)
            atomic_parquet(combined.drop_duplicates("image_path"), checkpoint)
            print(
                f"[exp264b-generate {shard}] {position}/{len(selected_cases)} "
                f"{full_id} {item.arm} s{item.seed}",
                flush=True,
            )
    combined = pd.concat([done_frame, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates("image_path").sort_values(["sample_rank", "arm", "seed"])
    if len(combined) != len(selected_manifest):
        raise AssertionError("generation shard incomplete")
    atomic_parquet(combined, checkpoint)
    payload = {
        "experiment": "Exp264B",
        "stage": "generation",
        "shard": shard,
        "nshards": nshards,
        "n_cases": len(selected_cases),
        "n_images": len(combined),
        "elapsed_seconds": time.time() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "manifest_sha256": sha256_file(checkpoint),
    }
    atomic_json(payload, checkpoint.with_suffix(".summary.json"))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=N_SHARDS)
    args = parser.parse_args()
    main(args.shard, args.nshards)
