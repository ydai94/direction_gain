"""Exp 29 Axis 2 (scoring) — Qwen3-VL categorical concept judge.

CLIP is far too insensitive to the A/B concept distinction (an obvious old->young
flip moved CLIP by 0.05). We instead ask Qwen3-VL a direct categorical question per
image: "which word best describes the main subject: {key_a}, {key_b}, or neither?".

Loads the merged generation manifest, judges each image, writes control_scores.parquet:
  concept_id, ctype, op, layer, label in {A,B,neither,unknown}, score in {0,0.5,1}.
  score = 0 if reads A, 1 if reads B (fully flipped), 0.5 if neither (concept removed).

Separate job from generation (Qwen-Image pipe NOT loaded here -> Qwen3-VL gets the
full GPU, no CPU offload). Resumable: skips (concept,op,layer) already scored.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from experiments.interp_program import config_concepts as CC

PROMPT = ("Look only at the main subject of this image. Which single word best "
          "describes it: '{a}', '{b}', or 'neither'? Reply with exactly one of: "
          "{a}, {b}, neither.")


def _merge_manifests():
    ms = sorted(glob.glob(os.path.join(CC.OUT_DIR, "manifest_shard*.parquet")))
    if not ms:
        raise FileNotFoundError("no manifest_shard*.parquet — run exp29_control first")
    df = pd.concat([pd.read_parquet(m) for m in ms], ignore_index=True)
    return df.drop_duplicates(["concept_id", "op", "layer"], keep="last")


def _ask(model, processor, image_path, a, b):
    import torch
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": PROMPT.format(a=a, b=b)}]}]
    inputs = processor.apply_chat_template(
        msg, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    txt = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()
    import gc
    del inputs, gen, trimmed
    gc.collect(); torch.cuda.empty_cache()
    return txt


def _label(txt, a, b):
    t = re.sub(r"[^a-z ]", " ", txt.lower())
    words = t.split()
    has_a = a.lower() in words or a.lower() in t
    has_b = b.lower() in words or b.lower() in t
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    if "neither" in t:
        return "neither"
    return "unknown"


SCORE = {"A": 0.0, "B": 1.0, "neither": 0.5, "unknown": 0.5}


def run():
    from experiments.causal_patching.score_images import load_qwen_vl
    man = _merge_manifests()
    prev = pd.read_parquet(CC.CONTROL_SCORES) if os.path.exists(CC.CONTROL_SCORES) else None
    done = set(zip(prev.concept_id, prev.op, prev.layer)) if prev is not None else set()

    model, processor = load_qwen_vl()
    rows = []
    for i, r in enumerate(man.itertuples(index=False)):
        if (r.concept_id, r.op, r.layer) in done:
            continue
        if not os.path.exists(r.image_path):
            continue
        txt = _ask(model, processor, r.image_path, r.key_a, r.key_b)
        lab = _label(txt, r.key_a, r.key_b)
        rows.append(dict(concept_id=r.concept_id, ctype=r.ctype, op=r.op,
                         layer=int(r.layer), key_a=r.key_a, key_b=r.key_b,
                         label=lab, score=SCORE[lab], raw=txt))
        if (i + 1) % 40 == 0 or i + 1 == len(man):
            df = pd.DataFrame(rows)
            if prev is not None:
                df = pd.concat([prev, df], ignore_index=True)
            df = df.drop_duplicates(["concept_id", "op", "layer"], keep="last")
            df.to_parquet(CC.CONTROL_SCORES, index=False)
            print(f"[vlm] scored {i+1}/{len(man)}")
    # final flush
    df = pd.DataFrame(rows)
    if prev is not None:
        df = pd.concat([prev, df], ignore_index=True)
    df = df.drop_duplicates(["concept_id", "op", "layer"], keep="last")
    df.to_parquet(CC.CONTROL_SCORES, index=False)
    print(f"[vlm] wrote {CC.CONTROL_SCORES} ({len(df)} rows)")
    # sanity: clean images should read A (score~0)
    clean = df[df.op == "clean"]
    print(f"[vlm] clean reads-A rate = {(clean.label=='A').mean():.2f} "
          f"(n={len(clean)}); clean label dist: {clean.label.value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="score", choices=["score"])
    ap.parse_args()
    run()
