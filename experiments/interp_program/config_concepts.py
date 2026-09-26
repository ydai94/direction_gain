"""Exp 29 config — non-sensitive concept control (the 'detection != control'
generalization test).

24 concept contrasts across 4 types (object / color / attribute / relation),
each instantiated over ~10 carriers so a probe has within-class samples and the
direction d = mean(repA) - mean(repB) is stable. Mirrors the stereotype triplet
machinery with VARIANTS = ("base", "A", "B") <-> ("neutral", "stereo", "anti").

Nothing here mutates the stereotype caches; all outputs live under
results/exp29_concepts/ and a private reps.h5.
"""
from __future__ import annotations

import os

import pandas as pd

from experiments.interp_program import config_interp as IC  # paths, CLIP, SWEEP_LAYERS

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(IC.OUT_DIR, "exp29_concepts")
REPS_H5 = os.path.join(OUT_DIR, "reps.h5")          # (N, 29, 3584) fp16
CONCEPT_INDEX = os.path.join(OUT_DIR, "concept_index.parquet")
DIRECTIONS_NPZ = os.path.join(OUT_DIR, "directions.npz")
READABILITY = os.path.join(OUT_DIR, "readability_curve.parquet")
EDIT_SURVIVAL = os.path.join(OUT_DIR, "edit_survival_concepts.parquet")
CONTROL_SCORES = os.path.join(OUT_DIR, "control_scores.parquet")   # merged
BEST_LAYER_TABLE = os.path.join(OUT_DIR, "best_layer_table.csv")
CORRELATIONS = os.path.join(OUT_DIR, "correlations.csv")
DECODED_SUB = "exp29"                                # under IC.DECODED_DIR
FIG = os.path.join(IC.FIG_DIR, "exp29_probe_vs_control.png")
README = os.path.join(OUT_DIR, "README.md")


def control_scores_shard(i: int) -> str:
    return os.path.join(OUT_DIR, f"control_scores_shard{i}.parquet")


# ---------------------------------------------------------------------------
# generation / intervention params
# ---------------------------------------------------------------------------
SEED = IC.SEED                       # 0
SWEEP_LAYERS = IC.SWEEP_LAYERS       # 1-indexed encoder blocks (same as stereotype sweep)
PROJECT_ALPHA = 1.0                  # full single projection-out (matches exp3)
STEER_ALPHA = 1.5                    # add raw (meanB-meanA) * alpha; pilot-checked
# one representative carrier index per type used for the image sweep
REP_CARRIER = 0


# ---------------------------------------------------------------------------
# carriers (10 per type)
# ---------------------------------------------------------------------------
OBJECT_CTX = ["in a park", "on the grass", "in a room", "outdoors", "on the street",
              "near a house", "in the wild", "on a table", "in a garden", "in a field"]
COLOR_NOUNS = ["car", "shirt", "ball", "house", "cup", "flower", "book", "chair",
               "umbrella", "hat"]
ATTR_NOUNS = ["person", "man", "woman", "child", "dog", "house", "car", "tree",
              "worker", "building"]
REL_SUBJ = ["a man", "a woman", "a dog", "a car", "a child", "a cat", "a house",
            "a bicycle", "a tree", "a person"]

CARRIERS = {"object": OBJECT_CTX, "color": COLOR_NOUNS,
            "attribute": ATTR_NOUNS, "relation": REL_SUBJ}


# ---------------------------------------------------------------------------
# 24 contrasts (cid, ctype, value_a, value_b, neutral_filler)
#   neutral_filler is the 'base' value (a hypernym for objects, "" elsewhere).
# ---------------------------------------------------------------------------
CONTRASTS = [
    # object
    dict(cid="dog_cat",        ctype="object", a="dog",      b="cat",       neu="animal"),
    dict(cid="car_bicycle",    ctype="object", a="car",      b="bicycle",   neu="vehicle"),
    dict(cid="apple_banana",   ctype="object", a="apple",    b="banana",    neu="fruit"),
    dict(cid="chair_table",    ctype="object", a="chair",    b="table",     neu="furniture"),
    dict(cid="bird_fish",      ctype="object", a="bird",     b="fish",      neu="animal"),
    dict(cid="house_tent",     ctype="object", a="house",    b="tent",      neu="structure"),
    # color
    dict(cid="red_blue",       ctype="color", a="red",     b="blue",   neu=""),
    dict(cid="black_white",    ctype="color", a="black",   b="white",  neu=""),
    dict(cid="green_yellow",   ctype="color", a="green",   b="yellow", neu=""),
    dict(cid="orange_purple",  ctype="color", a="orange",  b="purple", neu=""),
    dict(cid="pink_brown",     ctype="color", a="pink",    b="brown",  neu=""),
    dict(cid="gold_silver",    ctype="color", a="gold",    b="silver", neu=""),
    # attribute
    dict(cid="old_young",        ctype="attribute", a="old",     b="young",    neu=""),
    dict(cid="smiling_frowning", ctype="attribute", a="smiling", b="frowning", neu=""),
    dict(cid="big_small",        ctype="attribute", a="big",     b="small",    neu=""),
    dict(cid="tall_short",       ctype="attribute", a="tall",    b="short",    neu=""),
    dict(cid="happy_sad",        ctype="attribute", a="happy",   b="sad",      neu=""),
    dict(cid="clean_dirty",      ctype="attribute", a="clean",   b="dirty",    neu=""),
    # relation (value is a trailing location phrase)
    dict(cid="street_forest",    ctype="relation", a="in the street",  b="in the forest",      neu=""),
    dict(cid="city_country",     ctype="relation", a="in the city",    b="in the countryside", neu=""),
    dict(cid="indoors_outdoors", ctype="relation", a="indoors",        b="outdoors",           neu=""),
    dict(cid="mountain_beach",   ctype="relation", a="on a mountain",  b="on a beach",         neu=""),
    dict(cid="rain_sun",         ctype="relation", a="in the rain",    b="in the sunshine",    neu=""),
    dict(cid="day_night",        ctype="relation", a="during the day", b="at night",           neu=""),
]

VARIANTS = ("base", "A", "B")

# short single-word keys for the VLM categorical judge ("is it KEY_A, KEY_B, or neither?")
KEY_OVERRIDE = {
    "in the street": "street", "in the forest": "forest", "in the city": "city",
    "in the countryside": "countryside", "on a mountain": "mountain", "on a beach": "beach",
    "in the rain": "rainy", "in the sunshine": "sunny", "during the day": "daytime",
    "at night": "nighttime", "indoors": "indoors", "outdoors": "outdoors",
}


def concept_key(value: str) -> str:
    """Short label for the VLM judge."""
    return KEY_OVERRIDE.get(value, value.split()[-1])


def _collapse(s: str) -> str:
    s = " ".join(s.split()).strip()
    # fix the leading article (all prompts start with "a "): a -> an before a vowel
    if s.startswith("a ") and len(s) > 2 and s[2].lower() in "aeiou":
        s = "an " + s[2:]
    return s


def make_prompt(ctype: str, carrier: str, value: str) -> str:
    """Place `value` into a carrier. `value` may be "" for the base variant."""
    if ctype == "object":
        # carrier is a context phrase; value is the noun (or hypernym for base)
        return _collapse(f"a {value} {carrier}")
    if ctype in ("color", "attribute"):
        # carrier is a noun; value is the adjective (or "" for base)
        return _collapse(f"a {value} {carrier}")
    if ctype == "relation":
        # carrier already includes the article+subject; value is the trailing phrase
        return _collapse(f"{carrier} {value}")
    raise ValueError(ctype)


def build_concept_table() -> pd.DataFrame:
    """One row per (concept, carrier, variant). Columns:
       concept_id, ctype, carrier_idx, variant, value, prompt, row_idx."""
    rows = []
    for c in CONTRASTS:
        carriers = CARRIERS[c["ctype"]]
        for ci, carrier in enumerate(carriers):
            for variant, val in (("base", c["neu"]), ("A", c["a"]), ("B", c["b"])):
                rows.append(dict(
                    concept_id=c["cid"], ctype=c["ctype"], carrier_idx=ci,
                    variant=variant, value=val,
                    prompt=make_prompt(c["ctype"], carrier, val),
                ))
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["row_idx"] = range(len(df))
    return df


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(IC.DECODED_DIR, DECODED_SUB), exist_ok=True)
    os.makedirs(IC.FIG_DIR, exist_ok=True)


if __name__ == "__main__":
    # dry-run: print the resolved cohort + paths
    ensure_dirs()
    t = build_concept_table()
    print(f"[config_concepts] {t['concept_id'].nunique()} contrasts, "
          f"{len(t)} prompts ({t.groupby('ctype')['concept_id'].nunique().to_dict()})")
    for cid, g in t.groupby("concept_id", sort=False):
        ex = g[g.carrier_idx == REP_CARRIER]
        a = ex[ex.variant == "A"]["prompt"].iloc[0]
        b = ex[ex.variant == "B"]["prompt"].iloc[0]
        base = ex[ex.variant == "base"]["prompt"].iloc[0]
        print(f"  {cid:18s} [{g['ctype'].iloc[0]:9s}] base='{base}'  A='{a}'  B='{b}'")
    print(f"\nreps.h5        -> {REPS_H5}")
    print(f"concept_index  -> {CONCEPT_INDEX}")
    print(f"directions     -> {DIRECTIONS_NPZ}")
    print(f"SWEEP_LAYERS   -> {SWEEP_LAYERS}")
    print(f"decoded PNGs   -> {os.path.join(IC.DECODED_DIR, DECODED_SUB)}/<concept>/<op>/L<NN>.png")
