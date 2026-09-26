#!/usr/bin/env python3
"""Shared deterministic utilities for Exp254 Qwen top-three routing."""

from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path(os.environ.get("DG_ROOT", "."))
INTERP = BASE / "results"
OUT = INTERP / "exp254_qwen_top3"
PROBE_DIR = INTERP / "exp245_routing_qwen"
EXP251 = INTERP / "exp251_best_fixed_site"
COHORT = INTERP / "exp244_routing/cohort_frozen.parquet"

QWEN_IDS = 235
SEEDS = (0, 1)
TIMESTEPS = (10, 16)
ALL_SITES = ("out", "8", "10", "12", "17", "22", "26", "28")
RESTRICTED_LAYERS = (17, 22, 28)
BOOTSTRAP_SEED = 25420260828
BOOTSTRAP_REPLICATES = 10_000

EXPECTED_HASHES = {
    "test_cohort_frozen.csv": "e692687793f9cbc861567d73e1ca79b82cebf29465c657dd246a0f2de9004a23",
    "routes_frozen.csv": "eff934dbf5cd11a2278c7c90ef18f596aa9a7ee6a9cbf80eb68b593f356196ad",
    "scores.parquet": "b492e05c582134e42a2b6b539220b55d4071eeef3010242fd7dc119f9d601aa0",
    "test_fixed_qwen_scores.parquet": "c95cf7a63514dc3d41c1692f54374037944fe4bf49906f8152584e09a1353526",
    "probe245q_shard0.csv": "3fc0e709bcce9ee534e5ba63dafb40b3d1c1bbd3e405fc020fd23942b6ab2073",
    "probe245q_shard1.csv": "285e295df97bb16ac4422e8f6d7e47454c45bdfca44eb4ef5dda384662b83ddc",
    "probe245q_shard2.csv": "12e6168bcc838533e54c4e0d1d3fcca590185a1b872e678ead99661ead95d8f2",
    "probe245q_shard3.csv": "4e2ac7b53bd5c6578fffb2cc9e35acd3e47175187717c548d6b43deeaa49e0be",
    "probe245q_shard4.csv": "4e27d188b35a2429a2885b8c3d617cac0734095e49f9daba5a3ef962007a8610",
    "probe245q_shard5.csv": "aae7c240607fc58253b7826d3212f57bde66f915e632183bedf25a18ce277445",
    "exp251_generate_test_fixed.py": "5d2b4cec407247f895241ae19574f5530b6ea3a842e5a336f9b610d584b97c1a",
    "exp251_score.py": "09c54a101f572979c6807b8f7225ba3698e1f03a1ad78016422b4b6e7bf2f265",
}


def norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise AssertionError(f"hash mismatch {path}: {observed} != {expected}")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_qwen_cohort() -> pd.DataFrame:
    frozen_path = EXP251 / "test_cohort_frozen.csv"
    require_hash(frozen_path, EXPECTED_HASHES[frozen_path.name])
    frozen = pd.read_csv(frozen_path, dtype={"id": str})
    frozen = frozen.loc[frozen.model.eq("qwen")].copy()
    frozen["id"] = frozen.id.map(norm_id)
    if len(frozen) != QWEN_IDS or frozen.id.duplicated().any() or "984" in set(frozen.id):
        raise AssertionError("Qwen cohort must be 235 unique cases with ID 984 excluded")
    source = pd.read_parquet(COHORT).copy()
    source["id"] = source.cid12.map(norm_id)
    source = source.set_index("id")
    missing = set(frozen.id) - set(source.index)
    if missing:
        raise AssertionError(f"cohort prompts missing for {len(missing)} IDs")
    prompts = source.loc[frozen.id, ["prompt_neutral", "prompt_stereotype", "prompt_anti_stereotype"]]
    prompts = prompts.reset_index(drop=True)
    return pd.concat([frozen.reset_index(drop=True), prompts], axis=1)
