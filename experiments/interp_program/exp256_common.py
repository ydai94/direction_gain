#!/usr/bin/env python3
"""Shared constants and fail-closed helpers for Exp256."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("DG_ROOT", "."))
OUT = BASE / "results/exp256_gradient_screening"
EXP255 = BASE / "results/exp255_g_site_selection/development"

SITES = ("out", "17", "22", "28")
INTERNAL_SITES = ("17", "22", "28")
ALL_INTERNAL_LAYERS = tuple(range(1, 29))
SITE_TIE_ORDER = {site: index for index, site in enumerate(SITES)}
INTERNAL_TIE_ORDER = {site: index for index, site in enumerate(INTERNAL_SITES)}

TRAJECTORY_SEEDS = (0, 1, 2)
TIMESTEPS = (10, 16)
NUM_STEPS = 50
ALPHA_INTERNAL = 2.0
ALPHA_OUTPUT = 2.0
CFG = 4.0
N_BOOT = 10_000
N_SHARDS = 12

INPUT_HASHES = {
    "results/exp244_routing/cohort_frozen.parquet":
        "8b39f806d23112221c9676c8d0029c8153bcfcf69a2fbc3362cd44c1cad306b3",
    "results/exp251_best_fixed_site/test_cohort_frozen.csv":
        "e692687793f9cbc861567d73e1ca79b82cebf29465c657dd246a0f2de9004a23",
    "data/benchmark_prompts.csv":
        "fdf9d9445df9a8ecf9ef1950040c8848608dcf2eb65f54b9141aa9d2f20dc5d7",
    "cache/layer_probing/prompt_index.parquet":
        "a5013afdc94f638421d78029cf80b810560daf20758fde6690a8c1c8488bfc52",
    "cache/layer_probing/reps_mean.h5":
        "4b7266967403e2ce24c82cf8b72b764adfb6c404ab65a842ad4bde92edd61c47",
    "results/exp255_g_site_selection/development/development_oof.parquet":
        "32fe12c8cf579b44d37acd393156f58726ab31be4f86dd8349a2898750e94820",
    "results/exp255_g_site_selection/development/internal_case_endpoints.csv":
        "b8feb6c09f679ca5b3df1da6c2b24bb72b9d9ac84bb66bfcd8b7ac380a001939",
    "results/exp254_qwen_top3/case_endpoints.csv":
        "7e1dd61207865c9a0ad0690b2062eaaf3757cefba4a12e9d000e2587ecd5a849",
}


def seed32(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"Exp256::{label}".encode("utf-8")).digest()[:4], "big")


def normalized_triplet_hash(stereo: object, anti: object, neutral: object) -> str:
    """Match the frozen benchmark triplet-key construction used by Exp251/255."""
    text = "\n".join(
        " ".join(str(value).strip().lower().split())
        for value in (stereo, anti, neutral)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(relative_path: str) -> str:
    path = BASE / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    expected = INPUT_HASHES[relative_path]
    if observed != expected:
        raise AssertionError(f"hash mismatch for {relative_path}: {observed} != {expected}")
    return observed


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


def tensor_sha256(tensor: torch.Tensor) -> str:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def apply_centered_projection(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
    mu: torch.Tensor,
    *,
    alpha: float = ALPHA_INTERNAL,
    patch_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure equivalent of ProjectionPatcher(mode='project')."""
    import torch

    direction = direction.to(device=hidden_states.device, dtype=hidden_states.dtype)
    mu = mu.to(device=hidden_states.device, dtype=hidden_states.dtype)
    d_row = direction.view(1, 1, -1)
    scalar = ((hidden_states - mu.view(1, 1, -1)) * d_row).sum(dim=-1, keepdim=True)
    delta = float(alpha) * scalar * d_row
    if patch_mask is None:
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    mask = patch_mask[:, : hidden_states.shape[1]].to(hidden_states.device).unsqueeze(-1)
    return hidden_states - mask.to(hidden_states.dtype) * delta


def choose_site(predictions: Mapping[str, float], candidates: Iterable[str]) -> str:
    candidate_set = tuple(str(site) for site in candidates)
    if not candidate_set or not set(candidate_set).issubset(SITE_TIE_ORDER):
        raise ValueError(f"invalid candidate set {candidate_set}")
    return max(candidate_set, key=lambda site: (float(predictions[site]), -SITE_TIE_ORDER[site]))


def top_internal(scores: Mapping[str, float], k: int) -> tuple[str, ...]:
    if k not in {1, 2}:
        raise ValueError("Exp256 supports K=1 or K=2")
    if set(scores) != set(INTERNAL_SITES):
        raise ValueError(f"expected scores for {INTERNAL_SITES}, got {tuple(scores)}")
    ordered = sorted(
        INTERNAL_SITES,
        key=lambda site: (-float(scores[site]), INTERNAL_TIE_ORDER[site]),
    )
    return tuple(ordered[:k])


def interval_from_fixed_resamples(
    values: np.ndarray, fixed_resample_indices: np.ndarray
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    estimates = values[fixed_resample_indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def conditional_recall_interval(
    recalled: np.ndarray,
    eligible: np.ndarray,
    fixed_resample_indices: np.ndarray,
    random_reference: float,
) -> dict[str, float]:
    recalled = np.asarray(recalled, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if recalled.shape != eligible.shape or not eligible.any():
        raise AssertionError("invalid conditional recall inputs")
    point = float(recalled[eligible].mean())
    boot_values = []
    for indices in fixed_resample_indices:
        selected = eligible[indices]
        if selected.any():
            boot_values.append(float(recalled[indices][selected].mean()))
    if len(boot_values) != len(fixed_resample_indices):
        raise AssertionError("fixed resampling produced an empty internal-route draw")
    boot = np.asarray(boot_values, dtype=float)
    return {
        "estimate": point,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "random_reference": float(random_reference),
        "uplift_ci_low": float(np.quantile(boot - random_reference, 0.025)),
        "uplift_ci_high": float(np.quantile(boot - random_reference, 0.975)),
        "n_eligible": int(eligible.sum()),
    }


def exact_random_shortlist_outcome(row: Mapping[str, float], k: int, endpoint: str = "J") -> float:
    """Uniform expectation over every K-of-3 internal shortlist plus output."""
    from itertools import combinations

    values = []
    predictions = {site: float(row[f"pred_R_{site}"]) for site in SITES}
    for subset in combinations(INTERNAL_SITES, k):
        selected = choose_site(predictions, ("out",) + subset)
        values.append(float(row[f"{endpoint}_{selected}"]))
    return float(np.mean(values))


def gain_retention_80(j_screen: np.ndarray, j_output: np.ndarray, j_all4g: np.ndarray) -> np.ndarray:
    return (
        np.asarray(j_screen, dtype=float)
        - np.asarray(j_output, dtype=float)
        - 0.8 * (np.asarray(j_all4g, dtype=float) - np.asarray(j_output, dtype=float))
    )


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"{label} missing columns: {missing}")
