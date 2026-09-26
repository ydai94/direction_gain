#!/usr/bin/env python3
"""Deterministic, outcome-blind utilities for Exp264A."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BASE = Path(os.environ.get('DG_ROOT', '.'))
OUT = BASE / "results/exp264a_qwen_anchor_top1"
SOURCE_CASES = BASE / "results/exp261a_full_output_top1_n/cases_frozen.parquet"
BENCHMARK = BASE / "data/benchmark_prompts.csv"
TOKENIZER_DIR = Path(os.path.join(os.environ.get("MODEL_ROOT", "models"), "Qwen-Image/tokenizer"))
SAMPLE_SALT = "Exp264A::qwen-anchor-pilot::20260905"
RANDOM_LAYER_SALT = "Exp264A::random-layer-v1"
LAYERS = tuple(range(1, 29))
TRAJECTORY_SEEDS = (0, 1, 2)
TIMESTEPS = (10, 16)
IMAGE_SEEDS = (0, 1)
N_CASES = 64
MIN_RELIABLE_ANCHORS = 48
ALPHA = 2.0
N_BOOT = 10_000
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 34
TOKENIZER_MAX_LENGTH = 1024


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


def stable_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def seed32(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"Exp264A::{label}".encode()).digest()[:4], "big")


def unique_complete_phrase(text: str, phrase: object) -> tuple[str, int, int, str]:
    value = "" if phrase is None or pd.isna(phrase) else str(phrase).strip()
    if not value:
        return "missing", -1, -1, ""
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", flags=re.IGNORECASE)
    matches = list(pattern.finditer(str(text)))
    if len(matches) == 0:
        return "missing", -1, -1, value
    if len(matches) > 1:
        return "ambiguous", -1, -1, value
    match = matches[0]
    return "unique", int(match.start()), int(match.end()), str(text)[match.start():match.end()]


def resolve_anchor_text(row: Mapping[str, object]) -> dict[str, object]:
    neutral = str(row["prompt_neutral"])
    head_status, start, end, surface = unique_complete_phrase(neutral, row.get("head"))
    if head_status == "unique":
        return {"anchor_status": "mechanical_available", "anchor_source": "head",
                "anchor_text": surface, "char_start": start, "char_end": end,
                "failure_reason": ""}
    if head_status == "ambiguous":
        return {"anchor_status": "unavailable", "anchor_source": "head",
                "anchor_text": "", "char_start": -1, "char_end": -1,
                "failure_reason": "head_ambiguous"}
    target_status, start, end, surface = unique_complete_phrase(neutral, row.get("target"))
    if target_status == "unique":
        return {"anchor_status": "mechanical_available", "anchor_source": "target",
                "anchor_text": surface, "char_start": start, "char_end": end,
                "failure_reason": ""}
    reason = "target_ambiguous" if target_status == "ambiguous" else "head_and_target_missing"
    return {"anchor_status": "unavailable", "anchor_source": "",
            "anchor_text": "", "char_start": -1, "char_end": -1,
            "failure_reason": reason}


def hamilton_quotas(counts: pd.Series, total: int) -> pd.Series:
    counts = counts.astype(int).sort_index()
    if total < len(counts) or (counts <= 0).any():
        raise ValueError("Hamilton allocation requires positive strata and total >= n_strata")
    ideal = counts / counts.sum() * total
    quota = np.floor(ideal).astype(int)
    quota[quota < 1] = 1
    while int(quota.sum()) < total:
        remainder = ideal - quota
        key = sorted(remainder.index, key=lambda item: (-float(remainder.loc[item]), str(item)))[0]
        quota.loc[key] += 1
    while int(quota.sum()) > total:
        candidates = [item for item in quota.index if quota.loc[item] > 1]
        if not candidates:
            raise ValueError("cannot satisfy lower-bound Hamilton allocation")
        key = sorted(candidates, key=lambda item: (float(ideal.loc[item] - quota.loc[item]), str(item)))[0]
        quota.loc[key] -= 1
    return quota.astype(int)


def deterministic_sample(cases: pd.DataFrame, total: int = N_CASES) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"full_id", "triplet_key", "source", "bias_type", "degenerate_contrast"}
    if missing := required - set(cases.columns):
        raise AssertionError(f"source cases missing {sorted(missing)}")
    work = cases.copy()
    work["full_id"] = work["full_id"].astype(str)
    work["triplet_key"] = work["triplet_key"].astype(str)
    aliases = (
        work.sort_values("full_id", kind="mergesort")
        .groupby("triplet_key", sort=True)["full_id"]
        .agg(list)
        .rename("duplicate_aliases")
    )
    canonical = work.sort_values("full_id", kind="mergesort").drop_duplicates("triplet_key", keep="first")
    canonical = canonical.loc[~canonical["degenerate_contrast"].astype(bool)].copy()
    canonical["stratum"] = canonical["source"].astype(str) + "||" + canonical["bias_type"].astype(str)
    counts = canonical.groupby("stratum").size()
    quotas = hamilton_quotas(counts, total)
    selected = []
    for stratum, quota in quotas.items():
        part = canonical.loc[canonical["stratum"].eq(stratum)].copy()
        part["sample_hash"] = part["triplet_key"].map(lambda value: stable_hash(f"{SAMPLE_SALT}|{value}"))
        part = part.sort_values(["sample_hash", "triplet_key", "full_id"], kind="mergesort").head(int(quota))
        selected.append(part)
    sample = pd.concat(selected, ignore_index=True).sort_values(
        ["stratum", "sample_hash", "triplet_key"], kind="mergesort"
    ).reset_index(drop=True)
    sample["sample_rank"] = np.arange(len(sample), dtype=int)
    sample["sample_salt"] = SAMPLE_SALT
    sample["duplicate_aliases"] = sample["triplet_key"].map(aliases).map(json.dumps)
    if len(sample) != total or sample["triplet_key"].duplicated().any():
        raise AssertionError("deterministic sample coverage failure")
    quota_frame = pd.DataFrame({"stratum": quotas.index, "population": counts.loc[quotas.index],
                                "quota": quotas.to_numpy(int)})
    return sample, quota_frame


def token_coordinates(tokenizer, prompt: str, char_start: int, char_end: int) -> dict[str, object]:
    prefix, suffix = PROMPT_TEMPLATE.split("{}")
    wrapped = prefix + prompt + suffix
    prompt_start = len(prefix)
    absolute_start, absolute_end = prompt_start + int(char_start), prompt_start + int(char_end)
    encoded = tokenizer(
        [wrapped], max_length=TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_DROP_IDX,
        padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"][0].tolist()
    attention = encoded["attention_mask"][0].tolist()
    offsets = encoded["offset_mapping"][0].tolist()
    wrapped_rows = [
        index for index, ((start, end), valid) in enumerate(zip(offsets, attention))
        if valid and end > absolute_start and start < absolute_end
    ]
    if not wrapped_rows:
        raise AssertionError("anchor character span mapped to no tokens")
    if wrapped_rows != list(range(wrapped_rows[0], wrapped_rows[-1] + 1)):
        raise AssertionError("anchor token rows are not contiguous")
    valid_rows = [index for index, valid in enumerate(attention) if valid]
    valid_ord = {row: ordinal for ordinal, row in enumerate(valid_rows)}
    output_rows = [valid_ord[row] - PROMPT_TEMPLATE_DROP_IDX for row in wrapped_rows]
    if min(output_rows) < 0:
        raise AssertionError("anchor unexpectedly lies in dropped template rows")
    token_ids = [int(input_ids[row]) for row in wrapped_rows]
    return {
        "wrapped_rows": wrapped_rows,
        "output_rows": output_rows,
        "token_ids": token_ids,
        "decoded_tokens": [tokenizer.decode([token_id]) for token_id in token_ids],
        "wrapped_length": int(sum(attention)),
        "output_length": int(sum(attention) - PROMPT_TEMPLATE_DROP_IDX),
        "truncated": bool(offsets[wrapped_rows[-1]][1] < absolute_end),
    }


def pole_diff_rows(tokenizer, stereo: str, anti: str) -> tuple[list[int], list[int]]:
    prefix, suffix = PROMPT_TEMPLATE.split("{}")

    def encode(prompt: str):
        wrapped = prefix + prompt + suffix
        encoded = tokenizer(
            [wrapped], max_length=TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_DROP_IDX,
            padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True,
        )
        ids = encoded["input_ids"][0].tolist()
        valid = encoded["attention_mask"][0].bool().tolist()
        offsets = encoded["offset_mapping"][0].tolist()
        content_start, content_end = len(prefix), len(prefix) + len(prompt)
        content_rows = {
            index for index, ((start, end), keep) in enumerate(zip(offsets, valid))
            if keep and end > content_start and start < content_end
        }
        return ids, content_rows

    stereo_ids, stereo_content = encode(stereo)
    anti_ids, anti_content = encode(anti)
    matcher = difflib.SequenceMatcher(a=stereo_ids, b=anti_ids, autojunk=False)
    stereo_diff: list[int] = []
    anti_diff: list[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            stereo_diff.extend(index for index in range(i1, i2) if index in stereo_content)
        if tag in {"replace", "insert"}:
            anti_diff.extend(index for index in range(j1, j2) if index in anti_content)
    if not stereo_diff or not anti_diff:
        raise AssertionError("pole alignment produced an empty direction side")
    return sorted(set(stereo_diff)), sorted(set(anti_diff))


def choose_random_layer(triplet_key: str, eligible: Sequence[int]) -> int:
    layers = sorted({int(layer) for layer in eligible})
    if not layers:
        raise ValueError("random layer requires at least one eligible layer")
    digest = hashlib.sha256(f"{RANDOM_LAYER_SALT}|{triplet_key}".encode()).digest()
    return layers[int.from_bytes(digest[:8], "big") % len(layers)]


def choose_local_route(site_n: Mapping[str, float], candidate: str) -> str:
    if "A" not in site_n or candidate not in site_n:
        raise KeyError("route requires A and candidate")
    return candidate if float(site_n[candidate]) > float(site_n["A"]) else "A"


def paired_bootstrap(values: Iterable[float], label: str) -> tuple[np.ndarray, int]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise AssertionError("bootstrap requires a finite nonempty vector")
    seed = seed32(label)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(N_BOOT, len(array)))].mean(axis=1)
    return draws, seed
