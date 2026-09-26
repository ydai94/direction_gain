#!/usr/bin/env python3
"""Generate only manifest-selected missing Qwen arms for Exp254."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from exp254_common import BASE, EXPECTED_HASHES, OUT, atomic_parquet, require_hash

sys.path.insert(0, str(BASE))
from experiments.interp_program.exp251_generate_test_fixed import load_model


def generate_selected(manifest_path: Path, output: Path, shard: int, nshards: int) -> None:
    if not 0 <= shard < nshards:
        raise ValueError(f"invalid shard {shard}/{nshards}")
    source = BASE / "experiments/interp_program/exp251_generate_test_fixed.py"
    require_hash(source, EXPECTED_HASHES[source.name])
    manifest = pd.read_parquet(manifest_path)
    required = {"id", "layer", "seed", "image_path", "prompt_neutral", "prompt_stereotype", "prompt_anti_stereotype", "recipe_id"}
    if not required.issubset(manifest.columns):
        raise AssertionError(f"manifest missing columns {sorted(required-set(manifest.columns))}")
    selected = manifest.iloc[[i for i in range(len(manifest)) if i % nshards == shard]].copy()
    if selected.empty:
        atomic_parquet(selected, output / f"generated_manifest_shard{shard}.parquet")
        print(f"Exp254 shard {shard}: clean no-op")
        return
    if set(selected.recipe_id) != {"qwen_projection_alpha2_steps50_cfg4_negspace"}:
        raise AssertionError("recipe mismatch")
    torch.set_grad_enabled(False)
    _, prepare, generate = load_model("qwen")
    completed = []
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        image_path = Path(row.image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            prepared = prepare(pd.Series(row._asdict()), int(row.layer))
            generate(pd.Series(row._asdict()), int(row.layer), prepared, int(row.seed)).save(image_path)
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise AssertionError(f"invalid generated image {image_path}")
        completed.append(row._asdict())
        atomic_parquet(pd.DataFrame(completed), output / f"generated_manifest_shard{shard}.parquet")
        print(f"[Exp254 qwen] {position}/{len(selected)} {row.id} L{row.layer} s{row.seed}", flush=True)
    print(f"Exp254 shard {shard} complete rows={len(completed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=OUT / "generation_manifest.parquet")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    generate_selected(args.manifest, args.output, args.shard, args.nshards)
