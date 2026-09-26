#!/usr/bin/env python3
"""Shared constants and deterministic helpers for Exp261A."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


BASE = Path(os.environ.get('DG_ROOT', '.'))
OUT = BASE / "results/exp261a_full_output_top1_n"
ARCHIVE = Path(
    "${PROJECT_ROOT}/stereoimage/experiment_zips/"
    "exp_11_gt_kg_gt_pair_sv.zip"
)
ARCHIVE_ROOT = "exp_11_gt_kg_gt_pair_sv"
BENCHMARK = BASE / "data/benchmark_prompts.csv"
EXP244 = BASE / "results/exp244_routing"
EXP258B = BASE / "results/exp258b_canonical_g_gradient_repair"
EXP259A = BASE / "results/exp259a_all_layer_s_top1"
EXP260A = BASE / "results/exp260a_s_top2_finite_n"

N_CASES = 1831
N_DEVELOPMENT = 235
N_NONDEVELOPMENT = 1596
N_NONDEVELOPMENT_UNIQUE = 1592
LAYERS = tuple(range(1, 29))
TRAJECTORY_SEEDS = (0, 1, 2)
TIMESTEPS = (10, 16)
IMAGE_SEEDS = (0, 1)
NUM_STEPS = 50
ALPHA = 2.0
CFG = 4.0
N_BOOT = 10_000
N_S_SHARDS = 20
N_N_SHARDS = 32
N_GENERATION_SHARDS = 24
N_SCORE_SHARDS = 8
INTERNAL_RECIPE_ID = "qwen_projection_alpha2_steps50_cfg4_negspace"
OUTPUT_RECIPE_ID = "stereoimage_exp11_full_positional_output_alpha2_steps50_cfg4_negspace"
BENCHMARK_SHA256 = "fdf9d9445df9a8ecf9ef1950040c8848608dcf2eb65f54b9141aa9d2f20dc5d7"


def norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def seed32(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"Exp261A::{label}".encode()).digest()[:4], "big")


def triplet_hash(row: pd.Series | Mapping[str, object]) -> str:
    payload = [
        str(row["prompt_neutral"]),
        str(row["prompt_stereotype"]),
        str(row["prompt_anti_stereotype"]),
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def output_member(full_id: str, seed: int) -> str:
    return f"{ARCHIVE_ROOT}/{full_id}/steered_alpha_2.0/seed_{int(seed)}.png"


def choose_n_route(site_rows: pd.DataFrame) -> pd.Series:
    """Choose maximum N; exact ties prefer output, then smaller internal layer."""
    work = site_rows.copy()
    work["priority"] = work["site"].map(lambda value: -1 if str(value) == "out" else int(value))
    return work.sort_values(["N", "priority"], ascending=[False, True], kind="mergesort").iloc[0]


def validate_n_cells(frame: pd.DataFrame, expected_sites: pd.DataFrame) -> None:
    keys = ["full_id", "site", "probe_seed", "t_idx"]
    required = set(keys + ["cos", "mag", "gain", "rnorm"])
    if not required.issubset(frame.columns):
        raise AssertionError(f"N cells missing {sorted(required-set(frame.columns))}")
    if frame[keys].duplicated().any():
        raise AssertionError("duplicate N cell keys")
    expected = {
        (str(row.full_id), str(row.site), seed, timestep)
        for row in expected_sites.itertuples(index=False)
        for seed in TRAJECTORY_SEEDS
        for timestep in TIMESTEPS
    }
    observed = set(
        zip(
            frame["full_id"].astype(str), frame["site"].astype(str),
            frame["probe_seed"].astype(int), frame["t_idx"].astype(int),
        )
    )
    if observed != expected:
        raise AssertionError(f"N key mismatch missing={len(expected-observed)} extra={len(observed-expected)}")
    numeric = frame[["cos", "mag", "gain", "rnorm"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (frame["mag"] <= 0).any() or (frame["rnorm"] <= 0).any():
        raise AssertionError("non-finite or non-positive N cell")
    if not np.allclose(frame["gain"], frame["cos"] * frame["mag"], rtol=1e-7, atol=1e-7):
        raise AssertionError("gain identity failed")


def paired_bootstrap(values: Sequence[float], label: str) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise AssertionError("bootstrap needs a non-empty finite vector")
    seed = seed32(label)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(N_BOOT, len(array)))
    return array[indices].mean(axis=1), seed
