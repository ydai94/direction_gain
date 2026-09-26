#!/usr/bin/env python3
"""Frozen constants and deterministic helpers for Exp264B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from exp264_common import choose_local_route, choose_random_layer, pole_diff_rows


BASE = Path(os.environ.get('DG_ROOT', '.'))
SOURCE = BASE / "results/exp264a_qwen_anchor_top1"
OUT = BASE / "results/exp264b_qwen_anchor_pilot"
CASES = SOURCE / "cases_frozen.parquet"
CASES_SHA256 = "6e475988dbda54b1644e061147ad667e02987aef440f13ba3add07c7d4da9423"
LAYERS = tuple(range(1, 29))
SMOKE_LAYERS = (1, 14, 28)
TRAJECTORY_SEEDS = (0, 1, 2)
TIMESTEPS = (10, 16)
IMAGE_SEEDS = (0, 1)
ALPHA = 2.0
CFG = 4.0
NUM_STEPS = 50
N_CASES = 64
N_ANCHORS = 45
N_FALLBACK = 19
N_BOOT = 10_000
MAX_PEAK_BYTES = 75 * 1024**3
EPSILONS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(__import__("torch").uint8).numpy().tobytes())
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


def parse_int_list(value: object) -> list[int]:
    parsed = json.loads(str(value)) if isinstance(value, str) else list(value)
    result = [int(item) for item in parsed]
    if len(result) != len(set(result)) or result != sorted(result):
        raise AssertionError(f"invalid coordinate list {result}")
    return result


def load_cases(*, anchors_only: bool = False) -> pd.DataFrame:
    if sha256_file(CASES) != CASES_SHA256:
        raise AssertionError("Exp264A frozen case hash changed")
    frame = pd.read_parquet(CASES)
    frame["full_id"] = frame["full_id"].astype(str)
    frame["triplet_key"] = frame["triplet_key"].astype(str)
    if (
        len(frame) != N_CASES
        or frame["full_id"].duplicated().any()
        or frame["triplet_key"].duplicated().any()
        or int(frame["anchor_available"].astype(bool).sum()) != N_ANCHORS
    ):
        raise AssertionError("Exp264A frozen coverage changed")
    if anchors_only:
        frame = frame.loc[frame["anchor_available"].astype(bool)].copy()
        if len(frame) != N_ANCHORS:
            raise AssertionError("anchor subset count changed")
    return frame.sort_values(["sample_rank", "full_id"], kind="mergesort").reset_index(drop=True)


def validate_case_coordinates(row: Mapping[str, object]) -> tuple[list[int], list[int]]:
    wrapped = parse_int_list(row["wrapped_rows"])
    output = parse_int_list(row["output_rows"])
    if not bool(row["anchor_available"]):
        if wrapped or output:
            raise AssertionError("fallback case has anchor coordinates")
        return wrapped, output
    if not wrapped or len(wrapped) != len(output):
        raise AssertionError("anchor coordinate length mismatch")
    if wrapped != list(range(wrapped[0], wrapped[-1] + 1)):
        raise AssertionError("wrapped anchor coordinates are noncontiguous")
    if output != list(range(output[0], output[-1] + 1)):
        raise AssertionError("output anchor coordinates are noncontiguous")
    if any(index < 0 or index >= int(row["wrapped_length"]) for index in wrapped):
        raise AssertionError("wrapped anchor coordinate out of bounds")
    if any(index < 0 or index >= int(row["output_length"]) for index in output):
        raise AssertionError("output anchor coordinate out of bounds")
    return wrapped, output


def bool_mask(length: int, rows: Sequence[int], *, device: str = "cpu"):
    import torch

    mask = torch.zeros((1, int(length)), dtype=torch.bool, device=device)
    if rows:
        mask[0, torch.as_tensor(list(rows), dtype=torch.long, device=device)] = True
    return mask


def unit_direction(hidden_stereo, rows_stereo: Sequence[int], hidden_anti, rows_anti: Sequence[int]):
    import torch

    if not rows_stereo or not rows_anti:
        raise AssertionError("direction rows cannot be empty")
    stereo = hidden_stereo[0, list(rows_stereo)].float().mean(dim=0)
    anti = hidden_anti[0, list(rows_anti)].float().mean(dim=0)
    raw = anti - stereo
    norm = float(raw.norm())
    if not np.isfinite(norm) or norm <= 1e-8 or not torch.isfinite(raw).all():
        raise AssertionError("nonfinite or zero direction")
    return (raw / norm).contiguous(), norm


def native_add_delta(hidden, direction, mask, alpha: float = ALPHA):
    import torch

    d = direction.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
    mask3 = mask[:, : hidden.shape[1]].to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    edited = hidden + float(alpha) * mask3 * d
    delta = edited - hidden
    row_norm = delta.float().norm(dim=-1)[0]
    observed = torch.where(row_norm > 0)[0].detach().cpu().tolist()
    expected = torch.where(mask[0, : hidden.shape[1]].cpu())[0].tolist()
    if observed != expected:
        raise AssertionError(f"executed rows {observed} != mask rows {expected}")
    return edited, delta


def output_rows_for_pole(tokenizer, stereo: str, anti: str) -> tuple[list[int], list[int]]:
    stereo_rows, anti_rows = pole_diff_rows(tokenizer, stereo, anti)
    drop = 34
    stereo_out = [row - drop for row in stereo_rows if row >= drop]
    anti_out = [row - drop for row in anti_rows if row >= drop]
    if len(stereo_out) != len(stereo_rows) or len(anti_out) != len(anti_rows):
        raise AssertionError("pole direction unexpectedly touches dropped template rows")
    return stereo_out, anti_out


def paired_bootstrap(values: Sequence[float], label: str) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise AssertionError("bootstrap requires a finite nonempty vector")
    seed = int.from_bytes(hashlib.sha256(f"Exp264B::{label}".encode()).digest()[:4], "big")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(N_BOOT, len(array)))].mean(axis=1)
    return draws, seed


__all__ = [
    "ALPHA", "BASE", "CASES", "CASES_SHA256", "CFG", "EPSILONS", "IMAGE_SEEDS",
    "LAYERS", "MAX_PEAK_BYTES", "N_ANCHORS", "N_BOOT", "N_CASES", "N_FALLBACK",
    "NUM_STEPS", "OUT", "SMOKE_LAYERS", "TIMESTEPS", "TRAJECTORY_SEEDS",
    "atomic_json", "atomic_parquet", "bool_mask", "choose_local_route",
    "choose_random_layer", "load_cases", "native_add_delta", "output_rows_for_pole",
    "paired_bootstrap", "parse_int_list", "pole_diff_rows", "sha256_file",
    "sha256_tensor", "unit_direction", "validate_case_coordinates",
]
