"""FLUX experiment 1 — concept control by encoder layer (Exp 29 ported to FLUX.2-klein).

Cross-model replication of the Qwen-Image findings. FAIRNESS: uses the SAME prompts as the
Qwen experiments — `config_concepts.py`'s exact 24 contrasts + carriers + make_prompt.

FLUX.2-klein encoder = Qwen3 (36 layers). The DiT conditioning is the STACK of hidden states
from layers {9, 18, 27} only (joint_attention_dim 12288 = 3×4096). So:
  - intervening at a layer hook index i (on text_encoder.model.layers[i], output = hidden_states[i+1])
    affects a tap T∈{9,18,27} iff i+1 ≤ T, i.e. i ≤ T-1;
  - hook index i ≥ 27 (output hidden_states ≥28) feeds NO tap → predicted DEAD ZONE.

Per concept: build per-layer direction d[L] = mean(repA[L]) − mean(repB[L]) over carriers;
for each hook layer i, project out d[i+1] from layers[i] output, generate the A-variant,
VLM-judge if it flipped to B. Tests: (a) control concentrates ≤ tap 27 / dead >27, (b) different
concept types flip at different layers (color early / object late), (c) detection≠control (separate probe).

  python -m experiments.interp_flux.flux_concept_control --mode gen --shard i --nshards N
  python -m experiments.interp_flux.flux_concept_control --mode score
  python -m experiments.interp_flux.flux_concept_control --mode finalize
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from experiments.interp_program import config_concepts as CC

ROOT = os.path.join(os.path.dirname(__file__), "concept_control")
PNG = os.path.join(ROOT, "png")
MANIFEST = os.path.join(ROOT, "manifest.parquet")
SCORES = os.path.join(ROOT, "scores.parquet")
PROBE = os.path.join(ROOT, "probe.parquet")
MODEL = "black-forest-labs/FLUX.2-klein-9B"
CACHE = os.environ.get("MODEL_ROOT", "models") + "/"

# all 24 config_concepts contrasts (same as Qwen Exp 29) for the FLUX detection≠control test
CONCEPTS = ["dog_cat", "car_bicycle", "apple_banana", "chair_table", "bird_fish", "house_tent",
            "red_blue", "black_white", "green_yellow", "orange_purple", "pink_brown", "gold_silver",
            "old_young", "smiling_frowning", "big_small", "tall_short", "happy_sad", "clean_dirty",
            "street_forest", "city_country", "indoors_outdoors", "mountain_beach", "rain_sun", "day_night"]
# hook indices on text_encoder.model.layers[i]; taps at hidden_states 9/18/27 = hook idx 8/17/26
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
ALPHA = 2.0
STEPS = 8
SEEDS = [0, 1]
GEN_CARRIER = 0   # which carrier to render


def _load_flux():
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, cache_dir=CACHE, torch_dtype=torch.bfloat16).to("cuda")
    return pipe


@torch.no_grad()
def _layer_reps(pipe, prompts):
    """Per prompt: [37, 4096] mean-pooled (over valid tokens) hidden states, all encoder layers."""
    tok, te = pipe.tokenizer, pipe.text_encoder
    outs = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        inp = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inp = {k: v.to(te.device) for k, v in inp.items()}
        o = te(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
               output_hidden_states=True, use_cache=False)
        m = inp["attention_mask"][0].bool()
        hs = torch.stack([h[0][m].mean(0) for h in o.hidden_states])  # [37, 4096]
        outs.append(hs.float().cpu())
    return torch.stack(outs)  # [n_prompts, 37, 4096]


class _ProjHook:
    """Project out (alpha×) the unit direction d from a decoder layer's output tensor."""
    def __init__(self, d_unit, alpha):
        self.d = d_unit; self.alpha = alpha

    def __call__(self, module, inp, out):
        h = out if torch.is_tensor(out) else out[0]
        d = self.d.to(h.dtype).to(h.device)
        coef = (h * d).sum(-1, keepdim=True)
        h2 = h - self.alpha * coef * d
        return h2 if torch.is_tensor(out) else (h2,) + tuple(out[1:])


def run_probe():
    """Per-layer readability (Cohen's d on the A−B direction) per concept -> probe_best layer.
    Mirrors Qwen Exp 29's probe so the FLUX detection≠control test uses the same method."""
    pipe = _load_flux()
    idx = CC.build_concept_table()
    idx = pd.DataFrame(idx) if not isinstance(idx, pd.DataFrame) else idx
    rows = []
    for cid in CONCEPTS:
        g = idx[idx.concept_id == cid]
        a = _layer_reps(pipe, g[g.variant == "A"]["prompt"].tolist())  # [nA,37,4096]
        b = _layer_reps(pipe, g[g.variant == "B"]["prompt"].tolist())
        ds = []
        for L in range(a.shape[1]):
            dirv = a[:, L].mean(0) - b[:, L].mean(0)
            dirv = dirv / (dirv.norm() + 1e-8)
            pa = (a[:, L] * dirv).sum(-1); pb = (b[:, L] * dirv).sum(-1)
            pooled = float(torch.sqrt((pa.var() + pb.var()) / 2)) + 1e-8
            ds.append(abs(float(pa.mean() - pb.mean()) / pooled))
        rows.append(dict(concept_id=cid, ctype=g["ctype"].iloc[0],
                         probe_best=int(np.argmax(ds)), probe_d=float(max(ds))))
        print(f"[probe] {cid}: probe_best L{int(np.argmax(ds))} (d={max(ds):.1f})", flush=True)
    pd.DataFrame(rows).to_parquet(PROBE, index=False)
    print(f"[probe] -> {PROBE}")


def run_gen(shard=0, nshards=1):
    os.makedirs(PNG, exist_ok=True)
    pipe = _load_flux()
    idx = CC.build_concept_table()
    idx = pd.DataFrame(idx) if not isinstance(idx, pd.DataFrame) else idx
    rows = []
    jobs = [(c, L, s) for c in CONCEPTS for L in (["clean"] + LAYERS) for s in SEEDS]
    mine = [j for n, j in enumerate(jobs) if n % nshards == shard]
    by_concept = {}
    for c, L, s in mine:
        by_concept.setdefault(c, []).append((L, s))

    for cid, tasks in by_concept.items():
        g = idx[idx.concept_id == cid]
        a_carriers = g[g.variant == "A"]["prompt"].tolist()
        b_carriers = g[g.variant == "B"]["prompt"].tolist()
        ka = CC.concept_key(g[g.variant == "A"]["value"].iloc[0])
        kb = CC.concept_key(g[g.variant == "B"]["value"].iloc[0])
        a_gen = a_carriers[GEN_CARRIER]
        # per-layer direction d[L] = mean(repA) - mean(repB), over carriers
        repA = _layer_reps(pipe, a_carriers).mean(0)  # [37,4096]
        repB = _layer_reps(pipe, b_carriers).mean(0)
        d = repA - repB                                # [37,4096]
        d_unit = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        print(f"[gen] {cid}: A='{a_gen}' ka={ka} kb={kb}; tasks={len(tasks)}", flush=True)
        for L, s in tasks:
            tag = "clean" if L == "clean" else f"L{L:02d}"
            p = os.path.join(PNG, f"{cid}_{tag}_s{s}.png")
            if not os.path.exists(p):
                handle = None
                if L != "clean":
                    hook = _ProjHook(d_unit[L + 1], ALPHA)   # layers[L] output = hidden_states[L+1]
                    handle = pipe.text_encoder.model.layers[L].register_forward_hook(hook)
                try:
                    gen = torch.Generator(device="cuda").manual_seed(int(s))
                    img = pipe(prompt=a_gen, num_inference_steps=STEPS, generator=gen).images[0]
                    img.save(p)
                finally:
                    if handle is not None:
                        handle.remove()
            rows.append(dict(concept_id=cid, ctype=g["ctype"].iloc[0], hook_layer=str(L),
                             seed=s, image_path=p, key_a=ka, key_b=kb, a_prompt=a_gen))
    df = pd.DataFrame(rows)
    mf = MANIFEST.replace(".parquet", f"_shard{shard}.parquet")
    df.to_parquet(mf, index=False)
    print(f"[gen] shard {shard}: {len(df)} rows -> {mf}")


def _rebuild_manifest():
    """Rebuild manifest from PNG filenames + config_concepts keys (no FLUX needed)."""
    import glob
    idx = CC.build_concept_table()
    idx = pd.DataFrame(idx) if not isinstance(idx, pd.DataFrame) else idx
    rows = []
    for ip in sorted(glob.glob(os.path.join(PNG, "*.png"))):
        base = os.path.basename(ip)[:-4]
        left, s = base.rsplit("_s", 1)
        cid, tag = left.rsplit("_", 1)
        hook_layer = "clean" if tag == "clean" else str(int(tag[1:]))
        g = idx[idx.concept_id == cid]
        a = g[g.variant == "A"]; b = g[g.variant == "B"]
        rows.append(dict(concept_id=cid, ctype=g["ctype"].iloc[0], hook_layer=hook_layer,
                         seed=int(s), image_path=ip,
                         key_a=CC.concept_key(a["value"].iloc[0]),
                         key_b=CC.concept_key(b["value"].iloc[0])))
    man = pd.DataFrame(rows)
    man.to_parquet(MANIFEST, index=False)
    return man


def run_score():
    man = _rebuild_manifest()
    from experiments.causal_patching.score_images import load_qwen_vl
    from experiments.interp_program.exp29_score_vlm import _ask, _label, SCORE
    model, proc = load_qwen_vl()
    rows = []
    for _, r in man.iterrows():
        lab = _label(_ask(model, proc, r["image_path"], r["key_a"], r["key_b"]), r["key_a"], r["key_b"])
        rows.append(dict(image_path=r["image_path"], label=lab, score=SCORE[lab]))
    pd.DataFrame(rows).to_parquet(SCORES, index=False)
    print(f"[score] {len(rows)} judged -> {SCORES}")


def run_finalize():
    man = pd.read_parquet(MANIFEST)
    sc = pd.read_parquet(SCORES)
    m = man.merge(sc, on="image_path")
    # flip = no longer reads A (label != A); clean should read A (label==A)
    m["flip"] = (m["label"] != "A").astype(float)
    print("\n===== FLUX concept control by encoder layer (taps at hook idx 8/17/26; DEAD >26) =====")
    print(f"{'concept':14s} {'type':9s} clean  " + "  ".join(f"L{L:02d}" for L in LAYERS))
    for cid, g in m.groupby("concept_id"):
        ctype = g["ctype"].iloc[0]
        clean = g[g.hook_layer == "clean"]["flip"].mean()
        cells = []
        for L in LAYERS:
            v = g[g.hook_layer == str(L)]["flip"]
            cells.append(f"{v.mean():.1f}" if len(v) else " - ")
        print(f"{cid:14s} {ctype:9s} {clean:4.1f}   " + "   ".join(f"{c:>3s}" for c in cells))
    m.to_csv(os.path.join(ROOT, "flux_concept_control.csv"), index=False)
    print("\nclean≈0 (still A), flip→1 means projection flipped it. DEAD ZONE prediction: L29/32/35 ≈ 0.")

    # ---- detection≠control: probe-best vs control-best layer ----
    cb = []
    for cid, g in m.groupby("concept_id"):
        flips = {L: g[g.hook_layer == str(L)]["flip"].mean() for L in LAYERS}
        mx = max(flips.values()) if flips else 0
        best = min([L for L in LAYERS if flips.get(L, 0) == mx]) if mx > 0 else -1
        cb.append(dict(concept_id=cid, control_best=best, control_max=mx))
    cbdf = pd.DataFrame(cb)
    if os.path.exists(PROBE):
        pb = pd.read_parquet(PROBE)
        mm = pb.merge(cbdf, on="concept_id")
        order = {"color": 0, "attribute": 1, "relation": 2, "object": 3}
        mm["o"] = mm.ctype.map(order)
        print("\n===== FLUX detection≠control: probe-best vs control-best layer (encoder 36 layers) =====")
        print(f"  {'concept':16s}{'type':10s}{'probe_best':>11s}{'control_best':>13s}{'flip':>6s}")
        for _, r in mm.sort_values(["o", "concept_id"]).iterrows():
            cbs = "—" if r.control_best < 0 else f"L{int(r.control_best)}"
            print(f"  {r.concept_id:16s}{r.ctype:10s}{'L'+str(int(r.probe_best)):>11s}{cbs:>13s}{r.control_max:>6.1f}")
        fl = mm[mm.control_best >= 0]
        rho = fl["probe_best"].corr(fl["control_best"], method="spearman")
        mm.to_csv(os.path.join(ROOT, "flux_detection_vs_control.csv"), index=False)
        print(f"\n  Spearman ρ(probe_best, control_best) = {rho:.2f}  (n={len(fl)} flippable)  ≈0 => detection≠control (FLUX)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["probe", "gen", "score", "finalize"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "probe":
        run_probe()
    elif args.mode == "gen":
        run_gen(args.shard, args.nshards)
    elif args.mode == "score":
        run_score()
    else:
        run_finalize()


if __name__ == "__main__":
    main()
