"""Config for the 4-experiment interpretability program.

When/where/why stereotypes form in Qwen-Image, on the 5 visually-clear prompts
selected for the CDG breakthrough (recovered from
results/cdg_all_rho_sweep10_selected_ids/selected_prompts.parquet).

Nothing here mutates prior results; all outputs live under results/.
"""

from __future__ import annotations

import os

# --- paths -----------------------------------------------------------------
REPO = os.environ.get("DG_ROOT", ".")
SELECTED_PARQUET = os.path.join(
    REPO, "results", "cdg_all_rho_sweep10_selected_ids", "selected_prompts.parquet"
)
BENCHMARK_CSV = os.path.join(REPO, "data", "benchmark_prompts.csv")

# Locally-cached CLIP ViT-L/14 (offline cluster has no openai/clip-* on HF hub).
CLIP_PATH = (
    os.environ.get("MODEL_ROOT", "models") + "/" "clip-vit-large-patch14/"
    "models--openai--clip-vit-large-patch14/snapshots/"
    "32bd64288804d66eefd0ccbe215aa642df71cc41"
)

OUT_DIR = os.path.join(REPO, "results")
LATENTS_DIR = os.path.join(OUT_DIR, "latents")
DECODED_DIR = os.path.join(OUT_DIR, "decoded")
EXP1_DIR = os.path.join(OUT_DIR, "exp1")
EXP2_DIR = os.path.join(OUT_DIR, "exp2")
EXP3_DIR = os.path.join(OUT_DIR, "exp3")
EXP4_DIR = os.path.join(OUT_DIR, "exp4")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Combined manifests written by the exp scripts and consumed by the scorers.
MANIFEST_BIAS_ALL = os.path.join(OUT_DIR, "manifest_bias_all.parquet")
MANIFEST_ALIGN = os.path.join(OUT_DIR, "manifest_align.parquet")
BIAS_SCORES = os.path.join(OUT_DIR, "bias_scores.parquet")
ALIGN_SCORES = os.path.join(OUT_DIR, "alignment_scores.parquet")

# --- the 5 selected prompt ids --------------------------------------------
IDS = [
    "cfbc9e41c52775029e0723a28a9f57f1",  # football player / gender (male)
    "f29478dffb353a172b6854c5e3c96d7b",  # Japanese girl / hair color (black)
    "1778d102d1a14d82159a11c5163de17b",  # tailor / dress (high-end)
    "559367a2313742a5eb5a0adbb0d1ffdf",  # chess champion / accessory (glasses)
    "ba8b1ccaca402ead14c13ea8d8584d11",  # guitarist / genre (rock)
]
PILOT_IDS = IDS[:2]  # football + Japanese for the gate

# --- generation params (match the CDG baselines) --------------------------
NUM_STEPS = 50
CFG = 4.0
SEED = 0          # shared across n/s/a so trajectories are comparable
HEIGHT = None     # None -> pipe default (same as the CDG runs)
WIDTH = None
VARIANTS = ("neutral", "stereo", "anti")
VARIANT_PROMPT_COL = {
    "neutral": "prompt_neutral",
    "stereo": "prompt_stereotype",
    "anti": "prompt_anti_stereotype",
}

# Exp 1: steps (0-indexed) at which we save a decoded PNG + run the VLM judge.
CHECKPOINT_STEPS = [12, 25, 37, 44, 49]

# Exp 3: transformer blocks to sweep for the intervention curve (1-indexed,
# module text_encoder...layers[L-1]; matches reps_mean.h5 layer index L).
SWEEP_LAYERS = list(range(2, 29, 2)) + [17, 21, 23, 26]  # dense-ish + known peaks
SWEEP_LAYERS = sorted(set(SWEEP_LAYERS))
INTERVENTION_ALPHA = 1.0  # full projection-out of the stereotype direction

# --- Exp 2 token parsing ---------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with",
    "is", "are", "was", "were", "by", "for", "how", "they", "their", "them",
    "this", "that", "as", "it", "its", "from", "into", "very", "well",
    "appearing", "described", "wearing", "look", "big",
}
# Curated neutral replacements for the bias-relevant words in these 5 prompts.
# Words not present here get an ablation variant only (not a replacement one).
REPLACEMENTS = {
    "japanese": "young",
    "football": "sports",
    "tailor": "professional",
    "champion": "amateur",
    "chess": "board-game",
    "guitarist": "musician",
    "celebrating": "standing-in",
    "touchdown": "moment",
    "endzone": "field",
    "signature": "usual",
}


def ensure_dirs():
    for d in (OUT_DIR, LATENTS_DIR, DECODED_DIR, EXP1_DIR, EXP2_DIR, EXP3_DIR,
              EXP4_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


def load_prompts():
    """Return list of dicts (one per id) with all prompt/tail/category fields,
    ordered by IDS. Reads the selected-prompts parquet."""
    import pandas as pd
    df = pd.read_parquet(SELECTED_PARQUET)
    df["id"] = df["id"].astype(str)
    by_id = {r["id"]: r for _, r in df.iterrows()}
    out = []
    for pid in IDS:
        if pid not in by_id:
            raise KeyError(f"id {pid} missing from {SELECTED_PARQUET}")
        r = by_id[pid]
        out.append(dict(
            id=pid,
            bias_type=r["bias_type"],
            prompt_neutral=r["prompt_neutral"],
            prompt_stereotype=r["prompt_stereotype"],
            prompt_anti_stereotype=r["prompt_anti_stereotype"],
            stereotype_tails=r["stereotype_tails"],
            anti_stereotype_tails=r["anti_stereotype_tails"],
        ))
    return out


def content_words(prompt: str):
    """Deterministic content-word extraction: strip punctuation, drop stopwords
    and 1-char tokens, dedupe preserving order. Returns list of original-case
    surface words (as they appear, for string editing)."""
    import re
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", prompt)
    seen, out = set(), []
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS or len(wl) < 2:
            continue
        if wl in seen:
            continue
        seen.add(wl)
        out.append(w)
    return out


def ablate(prompt: str, word: str) -> str:
    """Remove the first whole-word occurrence of `word` and tidy whitespace."""
    import re
    pat = re.compile(r"\b" + re.escape(word) + r"\b", flags=re.IGNORECASE)
    new = pat.sub("", prompt, count=1)
    new = re.sub(r"\s{2,}", " ", new)
    new = re.sub(r"\s+([,.;])", r"\1", new).strip()
    return new


def replace(prompt: str, word: str, repl: str) -> str:
    """Replace the first whole-word occurrence of `word` with `repl`."""
    import re
    pat = re.compile(r"\b" + re.escape(word) + r"\b", flags=re.IGNORECASE)
    new = pat.sub(repl, prompt, count=1)
    new = re.sub(r"\s{2,}", " ", new).strip()
    return new
