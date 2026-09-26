#!/usr/bin/env python3
"""Post-hoc descriptive J contrast for Exp261A Qwen internal routes only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/remote_runs/exp261a_finalize/run_30057323"
OUT = ROOT / "results/exp261a_full_output_top1_n"
ENDPOINTS = ("Y", "H", "J")
N_BOOT = 10_000
COLUMNS = [
    "full_id",
    "split",
    "triplet_key",
    "selected_site",
    "Y_policy",
    "H_policy",
    "J_policy",
    "Y_output",
    "H_output",
    "J_output",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed32(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def summarize(frame: pd.DataFrame, label: str) -> dict[str, object]:
    if len(frame) == 0:
        raise AssertionError(f"empty internal-only subset: {label}")
    delta_j = (frame["J_policy"] - frame["J_output"]).to_numpy(np.float64)
    if not np.isfinite(delta_j).all():
        raise AssertionError(f"nonfinite J delta: {label}")
    seed = seed32(label)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta_j), size=(N_BOOT, len(delta_j)))
    draws = delta_j[indices].mean(axis=1)
    means = {}
    for endpoint in ENDPOINTS:
        policy = frame[f"{endpoint}_policy"].to_numpy(np.float64)
        output = frame[f"{endpoint}_output"].to_numpy(np.float64)
        means[endpoint] = {
            "policy": float(policy.mean()),
            "always_output": float(output.mean()),
            "delta": float((policy - output).mean()),
        }
    return {
        "n_units": int(len(frame)),
        "means": means,
        "J_policy_minus_output": {
            "estimate": float(delta_j.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "bootstrap_seed": seed,
            "n_bootstrap": N_BOOT,
        },
        "win_tie_loss": {
            "win": int((delta_j > 0).sum()),
            "tie": int((delta_j == 0).sum()),
            "loss": int((delta_j < 0).sum()),
        },
    }


def collapse_primary(frame: pd.DataFrame) -> pd.DataFrame:
    nondevelopment = frame.loc[frame["split"].eq("nondevelopment")].copy()
    if len(nondevelopment) != 1596:
        raise AssertionError("nondevelopment source-row count changed")
    route_counts = nondevelopment.groupby("triplet_key")["route"].nunique()
    if not route_counts.eq(1).all():
        raise AssertionError("duplicate triplet rows disagree on route")
    aggregations: dict[str, tuple[str, str]] = {
        "route": ("route", "first")
    }
    for endpoint in ENDPOINTS:
        aggregations[f"{endpoint}_policy"] = (f"{endpoint}_policy", "mean")
        aggregations[f"{endpoint}_output"] = (f"{endpoint}_output", "mean")
    primary = nondevelopment.groupby("triplet_key", as_index=False).agg(**aggregations)
    if len(primary) != 1592:
        raise AssertionError("primary unique-triplet count changed")
    return primary


def main() -> None:
    case_path = ARCHIVE / "case_results.parquet"
    registered_path = ARCHIVE / "results.json"
    frame = pd.read_parquet(case_path, columns=COLUMNS)
    registered = json.loads(registered_path.read_text())
    if len(frame) != 1831 or frame["full_id"].duplicated().any():
        raise AssertionError("Qwen source-row coverage changed")
    frame["selected_site"] = frame["selected_site"].astype(str)
    internal_layers = frame.loc[frame["selected_site"].ne("out"), "selected_site"]
    if not internal_layers.map(lambda value: value.isdigit() and 1 <= int(value) <= 28).all():
        raise AssertionError("unexpected Qwen selected-site labels")
    frame["route"] = np.where(frame["selected_site"].eq("out"), "out", "internal")

    output_rows = frame.loc[frame["route"].eq("out")]
    for endpoint in ENDPOINTS:
        delta = output_rows[f"{endpoint}_policy"] - output_rows[f"{endpoint}_output"]
        if not np.array_equal(delta.to_numpy(np.float64), np.zeros(len(delta))):
            raise AssertionError(f"output-routed arm is not identical for {endpoint}")

    primary = collapse_primary(frame)
    observed_primary = float((primary["J_policy"] - primary["J_output"]).mean())
    registered_primary = registered["summaries"][
        "nondevelopment_unique_triplets_primary"
    ]["J_policy_minus_output"]["estimate"]
    if not np.isclose(observed_primary, registered_primary, rtol=0, atol=1e-15):
        raise AssertionError("registered Qwen primary estimate did not reproduce")

    development = frame.loc[frame["split"].eq("development")]
    if len(development) != 235:
        raise AssertionError("development count changed")
    subsets = {
        "primary_nondevelopment_unique_internal_only": primary.loc[
            primary["route"].eq("internal")
        ],
        "development_source_rows_internal_only": development.loc[
            development["route"].eq("internal")
        ],
        "all_source_rows_internal_only": frame.loc[
            frame["route"].eq("internal")
        ],
    }
    summaries = {
        name: summarize(subset, f"Exp261A-posthoc-qwen-{name}")
        for name, subset in subsets.items()
    }
    primary_total = float((primary["J_policy"] - primary["J_output"]).sum())
    internal_primary = subsets["primary_nondevelopment_unique_internal_only"]
    internal_total = float(
        (internal_primary["J_policy"] - internal_primary["J_output"]).sum()
    )
    if not np.isclose(primary_total, internal_total, rtol=0, atol=1e-15):
        raise AssertionError("internal-only delta does not recover total Qwen delta")

    result = {
        "experiment": "Exp261A",
        "model": "Qwen-Image",
        "analysis": "posthoc_internal_route_conditional_J",
        "status": "ANALYZED_POSTHOC_DESCRIPTIVE",
        "estimand_note": (
            "Descriptive paired policy-minus-output contrast conditional on the "
            "outcome-blind router selecting internal; not the registered primary estimand."
        ),
        "input_path": str(case_path.relative_to(ROOT)),
        "input_sha256": sha256_file(case_path),
        "registered_results_sha256": sha256_file(registered_path),
        "subsets": summaries,
        "identity_checks": {
            "output_routed_delta_exactly_zero": True,
            "internal_only_delta_sum_equals_full_primary_delta_sum": True,
            "registered_primary_estimate_reproduced": True,
        },
    }
    output_path = OUT / "posthoc_internal_only.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
