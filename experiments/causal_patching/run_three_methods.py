"""Orchestrator for the Week 1 Three-Method Pareto experiment.

Supports four methods (one per --method invocation), each with its own
hyperparameter sweep:

  method=pca         k ∈ {1, 2, 3, 5, 10}      (subtractive projection at L26 HEAD)
  method=category    α ∈ {1, 2, 3, 5}          (additive at L26 HEAD)
  method=per_prompt  α ∈ {1, 2, 3, 5}          (additive at L26 HEAD)
  method=sv_tail_gt  α ∈ {0.5, 1, 1.5, 2, 3}   (additive at encoder OUTPUT,
                                                 broadcast — paper-1 recipe)

For methods at L26 HEAD: intervention is hooked into
text_encoder.model.language_model.layers[25] via ProjectionPatcher
(mode "project_k" for PCA, "add" for category and per_prompt).

For sv_tail_gt: NO patcher. Per-pair SV computed at runtime via
pipe.encode_prompt(stereotype_tails) - pipe.encode_prompt(anti_stereotype_tails),
mean-pooled. SV broadcast-added to the neutral prompt's
post-processed embeddings (paper-1 generate_with_tail_steering).

All methods: input prompt is the NEUTRAL prompt; output is one image per
(triplet, hyperparam, variant, seed). Same 50-triplet sample, same
DiT params (NUM_STEPS=50, CFG=4.0).
"""

import argparse
import hashlib
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.layer_probing import config_local as C
from experiments.causal_patching import encoder_runner as ER
from experiments.causal_patching.projection_patcher import ProjectionPatcher


DEVICE = "cuda"
LAYER = 26
NUM_STEPS = 50
CFG = 4.0
DEFAULT_SEEDS = (0, 1, 2)

EXP_DIR = os.path.join(C.PROJECT_DIR, "experiments", "causal_patching")
SAMPLE_PATH = os.path.join(EXP_DIR, "sample_triplets.parquet")
TOKEN_LABELS_V2_PATH = os.path.join(C.CACHE_DIR, "token_labels_v2.parquet")

M1_DIR = os.path.join(C.PROJECT_DIR, "results", "three_methods", "method1_pca")
M3_DIR = os.path.join(C.PROJECT_DIR, "results", "three_methods", "method3_category")
M9_DIR = os.path.join(C.PROJECT_DIR, "results", "three_methods", "method9_kg")
SV_DIR = os.path.join(C.PROJECT_DIR, "results", "three_methods", "sv_tail_gt_sweep")

IMG_ROOT = os.path.join(C.PROJECT_DIR, "experiment_outputs/three_methods")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipe():
    print("[load] DiffusionPipeline ...")
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        C.QWEN_IMAGE_DIR, torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def seed_from_id(triplet_id: str) -> int:
    return int(hashlib.md5(triplet_id.encode()).hexdigest()[:8], 16) % (2 ** 32)


def random_unit_for_id(triplet_id: str, hidden_dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed_from_id(triplet_id))
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    return v / np.linalg.norm(v)


def random_k_orthonormal_for_id(triplet_id: str, hidden_dim: int, k: int) -> np.ndarray:
    """Per-triplet random k-orthonormal basis. Returns (k, D)."""
    rng = np.random.default_rng(seed_from_id(triplet_id))
    A = rng.standard_normal((hidden_dim, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.T  # (k, hidden_dim) orthonormal rows


def load_head_pos_by_row():
    tl = pd.read_parquet(TOKEN_LABELS_V2_PATH)
    tl_nh = tl[(tl["variant"] == "neutral") & (tl["label"] == "HEAD")]
    return (tl_nh.groupby("row_idx")["tok_pos"]
            .apply(lambda s: sorted(int(x) for x in s)).to_dict())


def clean_embeds(pipe, neutral_prompt):
    """Encoder forward + post_process. Returns (pe (1, T', D), pm (1, T'))."""
    enc = ER.tokenize_wrapped(pipe.tokenizer, neutral_prompt, device=DEVICE)
    inner = ER._text_decoder(pipe.text_encoder)
    with torch.no_grad():
        out = inner(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            output_hidden_states=False, use_cache=False,
        )
    pe, pm = ER.post_process_for_dit(
        out.last_hidden_state, enc["attention_mask"],
        target_dtype=torch.bfloat16,
    )
    return pe, pm


def patched_embeds(pipe, neutral_prompt, configure_patcher_fn):
    """Encoder forward with patcher at LAYER. Returns (pe, pm).

    configure_patcher_fn(p) sets p.mode + relevant params (direction/basis/mu/alpha/patch_mask).
    """
    enc = ER.tokenize_wrapped(pipe.tokenizer, neutral_prompt, device=DEVICE)
    inner = ER._text_decoder(pipe.text_encoder)
    with ProjectionPatcher() as p:
        configure_patcher_fn(p)
        p.install(inner.layers[LAYER - 1])
        with torch.no_grad():
            out = inner(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                output_hidden_states=False, use_cache=False,
            )
    pe, pm = ER.post_process_for_dit(
        out.last_hidden_state, enc["attention_mask"],
        target_dtype=torch.bfloat16,
    )
    return pe, pm


def head_mask_for_prompt(pipe, neutral_prompt, ri_neutral, head_pos_by_row):
    """Build (mask (1, T_recip), T_recip, n_masked) for the wrapped neutral prompt."""
    enc = ER.tokenize_wrapped(pipe.tokenizer, neutral_prompt, device=DEVICE)
    T_recip = enc["input_ids"].shape[1]
    positions = head_pos_by_row.get(ri_neutral, [])
    valid = [p for p in positions if p < T_recip]
    mask = torch.zeros((1, T_recip), dtype=torch.bool)
    for p in valid:
        mask[0, p] = True
    return mask, T_recip, len(valid)


def generate(pipe, pe, pm, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    res = pipe(
        prompt_embeds=pe, prompt_embeds_mask=pm,
        negative_prompt=" ", num_inference_steps=NUM_STEPS, true_cfg_scale=CFG,
        generator=g,
    )
    return res.images[0]


def manifest_path_for(method):
    return os.path.join(C.PROJECT_DIR, "results", "three_methods", method, "manifest.parquet")


def img_dir_for(method):
    return os.path.join(IMG_ROOT, method)


def append_manifest(manifest_path, new_rows):
    """Atomic-ish append: read existing, concat, write to tmp, rename.

    Caller must ensure os.path.dirname(manifest_path) exists.
    """
    if os.path.exists(manifest_path):
        existing = pd.read_parquet(manifest_path).to_dict("records")
    else:
        existing = []
    combined = existing + new_rows
    tmp_path = manifest_path + ".tmp"
    pd.DataFrame(combined).to_parquet(tmp_path, index=False)
    os.replace(tmp_path, manifest_path)
    return len(combined)


def ensure_method_dirs(method):
    """Ensure both images_dir and results_dir exist for a method."""
    os.makedirs(img_dir_for(method), exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path_for(method)), exist_ok=True)


def existing_image_paths(manifest_path):
    if not os.path.exists(manifest_path):
        return set()
    return set(pd.read_parquet(manifest_path)["image_path"].tolist())


# ---------------------------------------------------------------------------
# Method 1: PCA k-subspace
# ---------------------------------------------------------------------------

def run_pca(pipe, sample, head_pos_by_row, k_values, variants, seeds):
    method = "method1_pca"
    images_dir = img_dir_for(method)
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = manifest_path_for(method)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    seen = existing_image_paths(manifest_path)
    print(f"[m1 pca] seeds={list(seeds)} k_values={list(k_values)} variants={list(variants)}")
    print(f"[m1 pca] images_dir={images_dir}")
    print(f"[m1 pca] existing manifest rows: {len(seen)}")

    d = np.load(os.path.join(M1_DIR, "directions.npz"), allow_pickle=True)
    pc_a = torch.from_numpy(np.asarray(d["pc_a"])).float()  # (30, D)
    pc_b = torch.from_numpy(np.asarray(d["pc_b"])).float()
    mu_t = torch.from_numpy(np.asarray(d["mu_L26_HEAD"])).float()
    fold_a_ids = set(d["fold_a_ids"].tolist())
    fold_b_ids = set(d["fold_b_ids"].tolist())
    print(f"[m1 pca] PCs loaded: pc_a {tuple(pc_a.shape)}  pc_b {tuple(pc_b.shape)}  "
          f"||mu||={mu_t.norm().item():.3f}")
    print(f"[m1 pca] fold_a_ids={len(fold_a_ids)}  fold_b_ids={len(fold_b_ids)}")

    new_rows = []
    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_no_head = 0

    for tri_idx, row in sample.iterrows():
        tid = row["id"]
        bias_type = row["bias_type"]
        neutral_prompt = row["prompt_neutral"]
        ri_neutral = int(row["row_idx_neutral"])

        mask, T_recip, n_masked = head_mask_for_prompt(
            pipe, neutral_prompt, ri_neutral, head_pos_by_row
        )
        if n_masked == 0:
            n_no_head += 1
            print(f"  [skip] {tid[:8]}: no HEAD positions on neutral")
            continue

        if tid in fold_a_ids:
            pc_use = pc_b
            fold = "A"
        elif tid in fold_b_ids:
            pc_use = pc_a
            fold = "B"
        else:
            pc_use = pc_a
            fold = "OOF"  # out of fold — should be rare; fallback

        # Build cache: per (k, variant) basis
        per_kv_embeds = {}
        for k in k_values:
            for var in variants:
                key = (k, var)
                if var == "matched":
                    basis = pc_use[:k]
                else:
                    basis = torch.from_numpy(
                        random_k_orthonormal_for_id(tid, C.HIDDEN_SIZE, k)
                    ).float()

                def configure(p, basis=basis, mask=mask, mu=mu_t):
                    p.mode = "project_k"
                    p.basis = basis
                    p.mu = mu
                    p.alpha = 1.0
                    p.patch_mask = mask
                pe, pm = patched_embeds(pipe, neutral_prompt, configure)
                per_kv_embeds[key] = (pe, pm)

        for (k, var), (pe, pm) in per_kv_embeds.items():
            for seed in seeds:
                fname = f"{tid}_pca_k{k}_{var}_s{seed}.png"
                p_path = os.path.join(images_dir, fname)
                if p_path in seen or os.path.exists(p_path):
                    n_skipped += 1
                    continue
                img = generate(pipe, pe, pm, seed)
                img.save(p_path)
                n_done += 1
                new_rows.append({
                    "id": tid, "bias_type": bias_type,
                    "method": "pca", "hyperparam": float(k),
                    "variant": var, "seed": int(seed),
                    "image_path": p_path, "prompt_neutral": neutral_prompt,
                    "n_masked": int(n_masked), "n_recipient": int(T_recip),
                    "fold": fold, "scope": "L26_HEAD",
                    "intervention": "subtract_subspace",
                    "layer": int(LAYER),
                })

        # checkpoint per triplet
        if new_rows:
            append_manifest(manifest_path, new_rows)
            new_rows = []
        elapsed = (time.time() - t0) / 60
        rate = n_done / max(elapsed * 60, 1e-3)
        print(f"  [m1 t{tri_idx+1}/{len(sample)}] {tid[:8]} fold={fold}  "
              f"done={n_done} skipped={n_skipped} no_head={n_no_head}  "
              f"{elapsed:.1f}m  {rate:.2f}img/s")

    if new_rows:
        append_manifest(manifest_path, new_rows)
    print(f"[m1 pca] done. images_generated={n_done}  skipped_existing={n_skipped}  "
          f"no_head={n_no_head}  time={(time.time()-t0)/60:.1f}m")


# ---------------------------------------------------------------------------
# Method 3: per-category direction (additive)
# ---------------------------------------------------------------------------

def safe_cat_key(category):
    return category.replace("-", "_")


def run_category(pipe, sample, head_pos_by_row, alphas, variants, seeds):
    method = "method3_category"
    images_dir = img_dir_for(method)
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = manifest_path_for(method)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    seen = existing_image_paths(manifest_path)
    print(f"[m3 cat] seeds={list(seeds)} alphas={list(alphas)} variants={list(variants)}")
    print(f"[m3 cat] images_dir={images_dir}")
    print(f"[m3 cat] existing manifest rows: {len(seen)}")

    d = np.load(os.path.join(M3_DIR, "directions.npz"), allow_pickle=True)
    categories = list(d["categories"])
    print(f"[m3 cat] categories: {categories}")
    cat_dirs = {}
    cat_folds = {}
    for c in categories:
        sk = safe_cat_key(c)
        cat_dirs[c] = {
            "d_a": torch.from_numpy(np.asarray(d[f"d_a_{sk}"])).float(),
            "d_b": torch.from_numpy(np.asarray(d[f"d_b_{sk}"])).float(),
        }
        cat_folds[c] = {
            "fold_a_ids": set(d[f"fold_a_ids_{sk}"].tolist()),
            "fold_b_ids": set(d[f"fold_b_ids_{sk}"].tolist()),
        }

    new_rows = []
    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_no_head = 0
    n_no_cat = 0

    for tri_idx, row in sample.iterrows():
        tid = row["id"]
        bias_type = row["bias_type"]
        neutral_prompt = row["prompt_neutral"]
        ri_neutral = int(row["row_idx_neutral"])

        if bias_type not in cat_dirs:
            n_no_cat += 1
            print(f"  [skip] {tid[:8]} bias_type={bias_type}: no per-category direction")
            continue

        mask, T_recip, n_masked = head_mask_for_prompt(
            pipe, neutral_prompt, ri_neutral, head_pos_by_row
        )
        if n_masked == 0:
            n_no_head += 1
            continue

        if tid in cat_folds[bias_type]["fold_a_ids"]:
            d_matched = cat_dirs[bias_type]["d_b"]
            fold = "A"
        elif tid in cat_folds[bias_type]["fold_b_ids"]:
            d_matched = cat_dirs[bias_type]["d_a"]
            fold = "B"
        else:
            d_matched = cat_dirs[bias_type]["d_a"]
            fold = "OOF"

        d_random = torch.from_numpy(random_unit_for_id(tid, C.HIDDEN_SIZE)).float()

        # Encode once per (alpha, variant)
        per_av_embeds = {}
        for a in alphas:
            for var in variants:
                d_use = d_matched if var == "matched" else d_random
                def configure(p, d_use=d_use, a=a, mask=mask):
                    p.mode = "add"
                    p.direction = d_use
                    p.alpha = float(a)
                    p.patch_mask = mask
                pe, pm = patched_embeds(pipe, neutral_prompt, configure)
                per_av_embeds[(a, var)] = (pe, pm)

        for (a, var), (pe, pm) in per_av_embeds.items():
            for seed in seeds:
                fname = f"{tid}_category_a{float(a):g}_{var}_s{seed}.png"
                p_path = os.path.join(images_dir, fname)
                if p_path in seen or os.path.exists(p_path):
                    n_skipped += 1
                    continue
                img = generate(pipe, pe, pm, seed)
                img.save(p_path)
                n_done += 1
                new_rows.append({
                    "id": tid, "bias_type": bias_type,
                    "method": "category", "hyperparam": float(a),
                    "variant": var, "seed": int(seed),
                    "image_path": p_path, "prompt_neutral": neutral_prompt,
                    "n_masked": int(n_masked), "n_recipient": int(T_recip),
                    "fold": fold, "scope": "L26_HEAD",
                    "intervention": "additive_category",
                    "layer": int(LAYER),
                })

        if new_rows:
            append_manifest(manifest_path, new_rows)
            new_rows = []
        elapsed = (time.time() - t0) / 60
        print(f"  [m3 t{tri_idx+1}/{len(sample)}] {tid[:8]} {bias_type} fold={fold}  "
              f"done={n_done} skipped={n_skipped} no_head={n_no_head} no_cat={n_no_cat}  "
              f"{elapsed:.1f}m")

    if new_rows:
        append_manifest(manifest_path, new_rows)
    print(f"[m3 cat] done. images_generated={n_done}  skipped_existing={n_skipped}  "
          f"no_head={n_no_head}  no_cat={n_no_cat}  time={(time.time()-t0)/60:.1f}m")


# ---------------------------------------------------------------------------
# Method 9: per-prompt direction (additive)
# ---------------------------------------------------------------------------

def run_per_prompt(pipe, sample, head_pos_by_row, alphas, variants, seeds):
    method = "method9_kg"
    images_dir = img_dir_for(method)
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = manifest_path_for(method)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    seen = existing_image_paths(manifest_path)
    print(f"[m9 kg] seeds={list(seeds)} alphas={list(alphas)} variants={list(variants)}")
    print(f"[m9 kg] images_dir={images_dir}")
    print(f"[m9 kg] existing manifest rows: {len(seen)}")

    d = np.load(os.path.join(M9_DIR, "directions.npz"), allow_pickle=True)
    m9_ids = list(d["ids"])
    d_p_by_id = {tid: torch.from_numpy(d["d_p"][i]).float()
                 for i, tid in enumerate(m9_ids)}
    print(f"[m9 kg] per-prompt directions for {len(d_p_by_id)} triplets")

    new_rows = []
    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_no_head = 0
    n_no_dp = 0

    for tri_idx, row in sample.iterrows():
        tid = row["id"]
        bias_type = row["bias_type"]
        neutral_prompt = row["prompt_neutral"]
        ri_neutral = int(row["row_idx_neutral"])

        if tid not in d_p_by_id:
            n_no_dp += 1
            print(f"  [skip] {tid[:8]}: no per-prompt direction (no TAIL coverage)")
            continue

        mask, T_recip, n_masked = head_mask_for_prompt(
            pipe, neutral_prompt, ri_neutral, head_pos_by_row
        )
        if n_masked == 0:
            n_no_head += 1
            continue

        d_matched = d_p_by_id[tid]
        d_random = torch.from_numpy(random_unit_for_id(tid, C.HIDDEN_SIZE)).float()

        per_av_embeds = {}
        for a in alphas:
            for var in variants:
                d_use = d_matched if var == "matched" else d_random
                def configure(p, d_use=d_use, a=a, mask=mask):
                    p.mode = "add"
                    p.direction = d_use
                    p.alpha = float(a)
                    p.patch_mask = mask
                pe, pm = patched_embeds(pipe, neutral_prompt, configure)
                per_av_embeds[(a, var)] = (pe, pm)

        for (a, var), (pe, pm) in per_av_embeds.items():
            for seed in seeds:
                fname = f"{tid}_kg_a{float(a):g}_{var}_s{seed}.png"
                p_path = os.path.join(images_dir, fname)
                if p_path in seen or os.path.exists(p_path):
                    n_skipped += 1
                    continue
                img = generate(pipe, pe, pm, seed)
                img.save(p_path)
                n_done += 1
                new_rows.append({
                    "id": tid, "bias_type": bias_type,
                    "method": "per_prompt", "hyperparam": float(a),
                    "variant": var, "seed": int(seed),
                    "image_path": p_path, "prompt_neutral": neutral_prompt,
                    "n_masked": int(n_masked), "n_recipient": int(T_recip),
                    "fold": "PER_PROMPT", "scope": "L26_HEAD",
                    "intervention": "additive_per_prompt",
                    "layer": int(LAYER),
                })

        if new_rows:
            append_manifest(manifest_path, new_rows)
            new_rows = []
        elapsed = (time.time() - t0) / 60
        print(f"  [m9 t{tri_idx+1}/{len(sample)}] {tid[:8]}  "
              f"done={n_done} skipped={n_skipped} no_head={n_no_head} no_dp={n_no_dp}  "
              f"{elapsed:.1f}m")

    if new_rows:
        append_manifest(manifest_path, new_rows)
    print(f"[m9 kg] done. images_generated={n_done}  skipped_existing={n_skipped}  "
          f"no_head={n_no_head}  no_dp={n_no_dp}  time={(time.time()-t0)/60:.1f}m")


# ---------------------------------------------------------------------------
# SV-Tail-GT: paper-1 exact recipe (per-pair SV, encoder-output broadcast)
# ---------------------------------------------------------------------------

def encode_full(pipe, prompt):
    """Return (last_hidden_state (1, T, D), attention_mask (1, T))."""
    enc = ER.tokenize_wrapped(pipe.tokenizer, prompt, device=DEVICE)
    inner = ER._text_decoder(pipe.text_encoder)
    with torch.no_grad():
        out = inner(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            output_hidden_states=False, use_cache=False,
        )
    return out.last_hidden_state, enc["attention_mask"]


def pool_post_drop(last_hidden_state, attention_mask, drop_idx=C.PROMPT_TEMPLATE_DROP_IDX):
    """Mean-pool last_hidden_state over non-pad-non-drop positions.

    Matches paper-1's compute_tail_steering_vector: works in pipe.encode_prompt
    space which is post-drop, post-mask. Returns (1, 1, D).
    """
    bool_mask = attention_mask.bool()
    valid = last_hidden_state[0][bool_mask[0]]  # (T_valid, D)
    kept = valid[drop_idx:]                      # (T_valid - 34, D)
    if kept.shape[0] == 0:
        raise RuntimeError(
            f"pool_post_drop: no content tokens after drop_idx; T_valid={valid.shape[0]}"
        )
    pooled = kept.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, D)
    return pooled


def per_pair_sv(pipe, stereo_tail, anti_tail):
    """Paper-1 per-pair SV: pool encoder output of each tail, take diff."""
    e_s, m_s = encode_full(pipe, stereo_tail)
    e_a, m_a = encode_full(pipe, anti_tail)
    pool_s = pool_post_drop(e_s, m_s)  # (1, 1, D)
    pool_a = pool_post_drop(e_a, m_a)
    return (pool_a - pool_s).float()  # (1, 1, D)


def run_sv_tail_gt(pipe, sample, alphas, seeds):
    method = "sv_tail_gt"
    images_dir = img_dir_for(method)
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = manifest_path_for(method)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    seen = existing_image_paths(manifest_path)
    print(f"[sv] seeds={list(seeds)} alphas={list(alphas)}")
    print(f"[sv] images_dir={images_dir}")
    print(f"[sv] existing manifest rows: {len(seen)}")
    print(f"[sv] using paper-1 per-pair SV recipe at encoder output (post-drop, post-mask)")

    new_rows = []
    t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_no_tail = 0

    for tri_idx, row in sample.iterrows():
        tid = row["id"]
        bias_type = row["bias_type"]
        neutral_prompt = row["prompt_neutral"]
        stereo_tail = row.get("stereotype_tails")
        anti_tail = row.get("anti_stereotype_tails")

        if not isinstance(stereo_tail, str) or not isinstance(anti_tail, str):
            n_no_tail += 1
            print(f"  [skip] {tid[:8]}: missing stereo/anti tail strings")
            continue
        if not stereo_tail.strip() or not anti_tail.strip():
            n_no_tail += 1
            print(f"  [skip] {tid[:8]}: empty tail strings")
            continue

        sv = per_pair_sv(pipe, stereo_tail, anti_tail)  # (1, 1, D)
        pe_clean, pm_clean = clean_embeds(pipe, neutral_prompt)  # (1, T', D), (1, T')

        T_recip = pe_clean.shape[1]
        # Broadcast SV to all T_recip positions
        sv_b = sv.to(device=pe_clean.device, dtype=pe_clean.dtype).expand(-1, T_recip, -1)

        # Pre-build steered embeds per alpha
        per_a_embeds = {}
        for a in alphas:
            pe_s = pe_clean + float(a) * sv_b
            per_a_embeds[a] = (pe_s, pm_clean)

        for a, (pe, pm) in per_a_embeds.items():
            for seed in seeds:
                fname = f"{tid}_sv_a{float(a):g}_s{seed}.png"
                p_path = os.path.join(images_dir, fname)
                if p_path in seen or os.path.exists(p_path):
                    n_skipped += 1
                    continue
                img = generate(pipe, pe, pm, seed)
                img.save(p_path)
                n_done += 1
                new_rows.append({
                    "id": tid, "bias_type": bias_type,
                    "method": "sv_tail_gt", "hyperparam": float(a),
                    "variant": "sv", "seed": int(seed),
                    "image_path": p_path, "prompt_neutral": neutral_prompt,
                    "n_masked": int(T_recip), "n_recipient": int(T_recip),
                    "fold": "PER_PAIR", "scope": "encoder_out_broadcast",
                    "intervention": "additive_sv_broadcast",
                    "layer": -1,
                })

        if new_rows:
            append_manifest(manifest_path, new_rows)
            new_rows = []
        elapsed = (time.time() - t0) / 60
        print(f"  [sv t{tri_idx+1}/{len(sample)}] {tid[:8]}  "
              f"done={n_done} skipped={n_skipped} no_tail={n_no_tail}  "
              f"{elapsed:.1f}m")

    if new_rows:
        append_manifest(manifest_path, new_rows)
    print(f"[sv] done. images_generated={n_done}  skipped_existing={n_skipped}  "
          f"no_tail={n_no_tail}  time={(time.time()-t0)/60:.1f}m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=("pca", "category", "per_prompt", "sv_tail_gt"))
    ap.add_argument("--hyperparams", type=float, nargs="+", required=True,
                    help="k values for pca (cast to int); alpha values otherwise")
    ap.add_argument("--variants", nargs="+", default=["matched", "random"],
                    choices=["matched", "random", "sv"],
                    help="For pca/category/per_prompt. SV ignores this.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N triplets (smoke test)")
    args = ap.parse_args()

    print(f"[setup] method={args.method}")
    print(f"[setup] hyperparams={args.hyperparams}")
    print(f"[setup] variants={args.variants}  seeds={args.seeds}")
    print(f"[setup] LAYER={LAYER}  NUM_STEPS={NUM_STEPS}  CFG={CFG}")

    sample = pd.read_parquet(SAMPLE_PATH).reset_index(drop=True)
    if args.limit is not None:
        sample = sample.head(args.limit).reset_index(drop=True)
    print(f"[load] sample n={len(sample)}")

    head_pos_by_row = load_head_pos_by_row()

    pipe = load_pipe()

    if args.method == "pca":
        k_values = [int(round(x)) for x in args.hyperparams]
        run_pca(pipe, sample, head_pos_by_row,
                k_values, args.variants, args.seeds)
    elif args.method == "category":
        run_category(pipe, sample, head_pos_by_row,
                     args.hyperparams, args.variants, args.seeds)
    elif args.method == "per_prompt":
        run_per_prompt(pipe, sample, head_pos_by_row,
                       args.hyperparams, args.variants, args.seeds)
    elif args.method == "sv_tail_gt":
        run_sv_tail_gt(pipe, sample, args.hyperparams, args.seeds)
    else:
        raise ValueError(args.method)


if __name__ == "__main__":
    main()
