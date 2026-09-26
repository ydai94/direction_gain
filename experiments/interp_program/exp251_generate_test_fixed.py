#!/usr/bin/env python3
"""Generate one development-selected fixed internal-site arm on the frozen test cohort."""

from __future__ import annotations
import os

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BASE = Path(os.environ.get("DG_ROOT", "."))
OUT = BASE / "results/exp251_best_fixed_site"
sys.path.insert(0, str(BASE))

SEEDS = {"flux": [0, 1, 2], "qwen": [0, 1], "sd3": [0, 1]}


def norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_model(model: str):
    if model == "flux":
        from experiments.interp_flux.flux_concept_control import _ProjHook, _layer_reps, _load_flux

        pipe = _load_flux()
        pipe.set_progress_bar_config(disable=True)

        def prepare(row: pd.Series, layer: int):
            stereo = _layer_reps(pipe, [str(row.prompt_stereotype)])[0]
            anti = _layer_reps(pipe, [str(row.prompt_anti_stereotype)])[0]
            direction = stereo - anti
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            return direction[layer + 1]

        def generate(row: pd.Series, layer: int, prepared, seed: int):
            handle = pipe.text_encoder.model.layers[layer].register_forward_hook(
                _ProjHook(prepared, 2.0)
            )
            try:
                return pipe(
                    prompt=str(row.prompt_neutral),
                    num_inference_steps=8,
                    generator=torch.Generator("cuda").manual_seed(seed),
                ).images[0]
            finally:
                handle.remove()

        return pipe, prepare, generate

    if model == "qwen":
        from experiments.causal_patching import encoder_runner as ER
        from experiments.causal_patching.projection_patcher import ProjectionPatcher
        from experiments.causal_patching.run_three_methods import load_pipe
        from experiments.interp_program.exp21_perprompt_encoder import load_perprompt_directions

        _, mu, _, _ = load_perprompt_directions()
        pipe = load_pipe()
        inner = ER._text_decoder(pipe.text_encoder)

        def layer_reps(prompt: str) -> np.ndarray:
            encoded = ER.tokenize_wrapped(pipe.tokenizer, prompt, device="cuda")
            output = inner(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            mask = encoded["attention_mask"][0].bool()
            return np.stack(
                [hidden[0][mask].float().mean(0).cpu().numpy() for hidden in output.hidden_states]
            )

        def prepare(row: pd.Series, layer: int):
            raw = layer_reps(str(row.prompt_stereotype)) - layer_reps(
                str(row.prompt_anti_stereotype)
            )
            return raw[layer] / (np.linalg.norm(raw[layer]) + 1e-8)

        def generate(row: pd.Series, layer: int, prepared, seed: int):
            encoded = ER.tokenize_wrapped(pipe.tokenizer, str(row.prompt_neutral), device="cuda")
            token_count = encoded["input_ids"].shape[1]
            with ProjectionPatcher() as patcher:
                patcher.mode = "project"
                patcher.direction = torch.from_numpy(
                    np.ascontiguousarray(prepared.astype(np.float32))
                )
                patcher.mu = torch.from_numpy(np.ascontiguousarray(mu[layer].astype(np.float32)))
                patcher.alpha = 2.0
                patcher.patch_mask = torch.ones((1, token_count), dtype=torch.bool)
                patcher.install(inner.layers[layer - 1])
                with torch.no_grad():
                    output = inner(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        output_hidden_states=False,
                        use_cache=False,
                    )
            prompt_embeds, prompt_mask = ER.post_process_for_dit(
                output.last_hidden_state,
                encoded["attention_mask"],
                target_dtype=torch.bfloat16,
            )
            return pipe(
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_mask,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                negative_prompt=" ",
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]

        return pipe, prepare, generate

    from experiments.interp_sd3.sd3_concept_control import ENC_CFG, _get_blocks, _load_sd3
    from experiments.interp_sd3.sd3_t5_detect_control import _ProjHook, _t5_dir

    pipe = _load_sd3()
    pipe.set_progress_bar_config(disable=True)
    blocks = _get_blocks(pipe, ENC_CFG["t5"])

    def prepare(row: pd.Series, layer: int):
        direction = _t5_dir(
            pipe, str(row.prompt_stereotype), str(row.prompt_anti_stereotype)
        )
        return direction[layer + 1]

    def generate(row: pd.Series, layer: int, prepared, seed: int):
        handle = blocks[layer].register_forward_hook(_ProjHook(prepared, 8.0))
        try:
            return pipe(
                prompt=str(row.prompt_neutral),
                num_inference_steps=28,
                guidance_scale=7.0,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
        finally:
            handle.remove()

    return pipe, prepare, generate


def main(model: str, shard: int, nshards: int, limit: int | None, output_root: Path) -> None:
    if not 0 <= shard < nshards:
        raise ValueError(f"invalid shard {shard}/{nshards}")
    sites = pd.read_csv(OUT / "fixed_sites_frozen.csv")
    selected = sites.loc[sites.model.eq(model), "fixed_internal_site"]
    if len(selected) != 1:
        raise AssertionError(f"{model}: fixed site not uniquely frozen")
    layer = int(selected.iloc[0])

    frozen = pd.read_csv(OUT / "test_cohort_frozen.csv", dtype={"id": str})
    ids = frozen.loc[frozen.model.eq(model), "id"].map(norm_id).tolist()
    cohort = pd.read_parquet(
        BASE / "results/exp244_routing/cohort_frozen.parquet"
    ).copy()
    cohort["id"] = cohort.cid12.map(norm_id)
    cohort = cohort.set_index("id").loc[ids].reset_index()
    if cohort.id.tolist() != ids:
        raise AssertionError(f"{model}: frozen test cohort order mismatch")
    mine = [index for index in range(len(cohort)) if index % nshards == shard]
    if limit is not None:
        mine = mine[:limit]

    png = output_root / f"test_fixed_{model}_png"
    png.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    _, prepare, generate = load_model(model)
    rows: list[dict[str, object]] = []
    for position, index in enumerate(mine, start=1):
        row = cohort.iloc[index]
        case_id = norm_id(row.id)
        prepared = prepare(row, layer)
        for seed in SEEDS[model]:
            image_path = png / f"{case_id}__fixed_L{layer:02d}__s{seed}.png"
            rows.append(
                {
                    "id": case_id,
                    "layer": layer,
                    "arm": "fixed",
                    "seed": seed,
                    "image_path": str(image_path),
                    "prompt_stereotype": row.prompt_stereotype,
                    "prompt_anti_stereotype": row.prompt_anti_stereotype,
                }
            )
            if not image_path.exists():
                generate(row, layer, prepared, seed).save(image_path)
        manifest = output_root / f"test_fixed_{model}_manifest_shard{shard}.parquet"
        atomic_parquet(pd.DataFrame(rows), manifest)
        print(f"[e251-test-{model}] {position}/{len(mine)} {case_id} L{layer}", flush=True)
    manifest = output_root / f"test_fixed_{model}_manifest_shard{shard}.parquet"
    atomic_parquet(pd.DataFrame(rows), manifest)
    expected = len(mine) * len(SEEDS[model])
    if len(rows) != expected or not all(Path(path).exists() for path in pd.DataFrame(rows).image_path):
        raise AssertionError(f"{model}: fixed test coverage failure")
    print(f"[e251-test-{model}] complete rows={len(rows)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(SEEDS), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.model, args.shard, args.nshards, args.limit, args.output_root)
