"""SD3.5 experiment 1 — concept control by encoder layer (Exp 29 ported to SD3.5-medium).

Cross-model replication, THIRD encoder family. FAIRNESS: uses the SAME prompts as the Qwen /
FLUX experiments — `config_concepts.py`'s exact 24 contrasts + carriers + make_prompt.

SD3.5-medium has THREE text encoders feeding an MMDiT (24 joint blocks):
  text_encoder   CLIP-L  12 layers  (seq uses penultimate hidden, + projected pooled)
  text_encoder_2 CLIP-G  32 layers  (seq uses penultimate hidden, + projected pooled)
  text_encoder_3 T5-XXL  24 layers  (seq uses LAST hidden) <- main semantic carrier, default target
Conditioning: pooled(CLIP-L 768 + CLIP-G 1280 = 2048) -> AdaLN; sequence = [CLIP(padded 4096) ; T5 4096].
Unlike FLUX there is no built-in tap dead zone; T5's every layer feeds the final hidden -> all live.

Per concept: per-layer direction d[L] = mean(repA[L]) − mean(repB[L]) over carriers; for each hook
layer i project out d[i+1] from {encoder}.block/layers[i] output, regenerate the A-variant, VLM-judge
if it flipped to B. Tests: which encoder/layer is controllable + detection≠control (separate probe).

  python -m experiments.interp_sd3.sd3_concept_control --mode probe   --enc t5
  python -m experiments.interp_sd3.sd3_concept_control --mode gen     --enc t5 --shard i --nshards N
  python -m experiments.interp_sd3.sd3_concept_control --mode score   --enc t5
  python -m experiments.interp_sd3.sd3_concept_control --mode finalize --enc t5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from experiments.interp_program import config_concepts as CC

MODEL = os.path.join(os.environ.get("MODEL_ROOT", "models"), "stable-diffusion-3.5-medium")

CONCEPTS = ["dog_cat", "car_bicycle", "apple_banana", "chair_table", "bird_fish", "house_tent",
            "red_blue", "black_white", "green_yellow", "orange_purple", "pink_brown", "gold_silver",
            "old_young", "smiling_frowning", "big_small", "tall_short", "happy_sad", "clean_dirty",
            "street_forest", "city_country", "indoors_outdoors", "mountain_beach", "rain_sun", "day_night"]

# per-encoder config: (n_layers, sweep layers, tokenizer attr, encoder attr, block-list path, max_len)
ENC_CFG = {
    "t5":    dict(n=24, layers=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23],
                  tok="tokenizer_3", enc="text_encoder_3", blocks="encoder.block", maxlen=256),
    "clipg": dict(n=32, layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31],
                  tok="tokenizer_2", enc="text_encoder_2", blocks="text_model.encoder.layers", maxlen=77),
    "clipl": dict(n=12, layers=[1, 3, 5, 7, 9, 11],
                  tok="tokenizer", enc="text_encoder", blocks="text_model.encoder.layers", maxlen=77),
}

ALPHA = 4.0     # SD3 encoders are higher-norm; tune in pilot
STEPS = 28
GUID = 4.5
SEEDS = [0]
GEN_CARRIER = 0


def _root(enc):
    r = os.path.join(os.path.dirname(__file__), f"concept_control_{enc}")
    return r, os.path.join(r, "png"), os.path.join(r, "manifest.parquet"), \
        os.path.join(r, "scores.parquet"), os.path.join(r, "probe.parquet")


def _load_sd3():
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
    return pipe


def _get_blocks(pipe, cfg):
    mod = getattr(pipe, cfg["enc"])
    for p in cfg["blocks"].split("."):
        mod = getattr(mod, p)
    return mod  # ModuleList


@torch.no_grad()
def _layer_reps(pipe, prompts, cfg):
    """Per prompt: [n_layers+1, hidden] mean-pooled (over valid tokens) hidden states."""
    tok = getattr(pipe, cfg["tok"])
    enc = getattr(pipe, cfg["enc"])
    outs = []
    for p in prompts:
        inp = tok(p, return_tensors="pt", padding="max_length", truncation=True, max_length=cfg["maxlen"])
        ids = inp["input_ids"].to(enc.device)
        am = inp["attention_mask"].to(enc.device)
        o = enc(input_ids=ids, attention_mask=am, output_hidden_states=True)
        m = am[0].bool()
        hs = torch.stack([h[0][m].mean(0) for h in o.hidden_states])  # [n_layers+1, hidden]
        outs.append(hs.float().cpu())
    return torch.stack(outs)


class _ProjHook:
    def __init__(self, d_unit, alpha):
        self.d = d_unit; self.alpha = alpha

    def __call__(self, module, inp, out):
        h = out if torch.is_tensor(out) else out[0]
        d = self.d.to(h.dtype).to(h.device)
        coef = (h * d).sum(-1, keepdim=True)
        h2 = h - self.alpha * coef * d
        return h2 if torch.is_tensor(out) else (h2,) + tuple(out[1:])


def run_probe(enc):
    cfg = ENC_CFG[enc]
    ROOT, _, _, _, PROBE = _root(enc)
    os.makedirs(ROOT, exist_ok=True)
    pipe = _load_sd3()
    idx = pd.DataFrame(CC.build_concept_table())
    rows = []
    for cid in CONCEPTS:
        g = idx[idx.concept_id == cid]
        a = _layer_reps(pipe, g[g.variant == "A"]["prompt"].tolist(), cfg)
        b = _layer_reps(pipe, g[g.variant == "B"]["prompt"].tolist(), cfg)
        ds = []
        for L in range(a.shape[1]):
            dirv = a[:, L].mean(0) - b[:, L].mean(0)
            dirv = dirv / (dirv.norm() + 1e-8)
            pa = (a[:, L] * dirv).sum(-1); pb = (b[:, L] * dirv).sum(-1)
            pooled = float(torch.sqrt((pa.var() + pb.var()) / 2)) + 1e-8
            ds.append(abs(float(pa.mean() - pb.mean()) / pooled))
        rows.append(dict(concept_id=cid, ctype=g["ctype"].iloc[0],
                         probe_best=int(np.argmax(ds)), probe_d=float(max(ds))))
        print(f"[probe:{enc}] {cid}: L{int(np.argmax(ds))} (d={max(ds):.1f})", flush=True)
    pd.DataFrame(rows).to_parquet(PROBE, index=False)
    print(f"[probe] -> {PROBE}")


def run_gen(enc, shard=0, nshards=1):
    cfg = ENC_CFG[enc]
    ROOT, PNG, MANIFEST, _, _ = _root(enc)
    os.makedirs(PNG, exist_ok=True)
    pipe = _load_sd3()
    blocks = _get_blocks(pipe, cfg)
    idx = pd.DataFrame(CC.build_concept_table())
    jobs = [(c, L, s) for c in CONCEPTS for L in (["clean"] + cfg["layers"]) for s in SEEDS]
    mine = [j for n, j in enumerate(jobs) if n % nshards == shard]
    by_concept = {}
    for c, L, s in mine:
        by_concept.setdefault(c, []).append((L, s))
    rows = []
    for cid, tasks in by_concept.items():
        g = idx[idx.concept_id == cid]
        a_carriers = g[g.variant == "A"]["prompt"].tolist()
        b_carriers = g[g.variant == "B"]["prompt"].tolist()
        ka = CC.concept_key(g[g.variant == "A"]["value"].iloc[0])
        kb = CC.concept_key(g[g.variant == "B"]["value"].iloc[0])
        a_gen = a_carriers[GEN_CARRIER]
        repA = _layer_reps(pipe, a_carriers, cfg).mean(0)
        repB = _layer_reps(pipe, b_carriers, cfg).mean(0)
        d = repA - repB
        d_unit = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        print(f"[gen:{enc}] {cid}: A='{a_gen}' ka={ka} kb={kb}; tasks={len(tasks)}", flush=True)
        for L, s in tasks:
            tag = "clean" if L == "clean" else f"L{L:02d}"
            p = os.path.join(PNG, f"{cid}_{tag}_s{s}.png")
            if not os.path.exists(p):
                handle = None
                if L != "clean":
                    hook = _ProjHook(d_unit[L + 1], ALPHA)
                    handle = blocks[L].register_forward_hook(hook)
                try:
                    gen = torch.Generator(device="cuda").manual_seed(int(s))
                    img = pipe(prompt=a_gen, num_inference_steps=STEPS, guidance_scale=GUID,
                               generator=gen).images[0]
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


def _rebuild_manifest(enc):
    cfg = ENC_CFG[enc]
    ROOT, PNG, MANIFEST, _, _ = _root(enc)
    idx = pd.DataFrame(CC.build_concept_table())
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


def run_score(enc):
    ROOT, PNG, MANIFEST, SCORES, _ = _root(enc)
    man = _rebuild_manifest(enc)
    from experiments.causal_patching.score_images import load_qwen_vl
    from experiments.interp_program.exp29_score_vlm import _ask, _label, SCORE
    model, proc = load_qwen_vl()
    rows = []
    for _, r in man.iterrows():
        lab = _label(_ask(model, proc, r["image_path"], r["key_a"], r["key_b"]), r["key_a"], r["key_b"])
        rows.append(dict(image_path=r["image_path"], label=lab, score=SCORE[lab]))
    pd.DataFrame(rows).to_parquet(SCORES, index=False)
    print(f"[score] {len(rows)} judged -> {SCORES}")


def run_finalize(enc):
    cfg = ENC_CFG[enc]
    ROOT, PNG, MANIFEST, SCORES, PROBE = _root(enc)
    man = pd.read_parquet(MANIFEST)
    sc = pd.read_parquet(SCORES)
    m = man.merge(sc, on="image_path")
    m["flip"] = (m["label"] != "A").astype(float)
    LAYERS = cfg["layers"]
    print(f"\n===== SD3.5 concept control by {enc} layer (encoder {cfg['n']} layers) =====")
    print(f"{'concept':14s} {'type':9s} clean  " + "  ".join(f"L{L:02d}" for L in LAYERS))
    for cid, g in m.groupby("concept_id"):
        ctype = g["ctype"].iloc[0]
        clean = g[g.hook_layer == "clean"]["flip"].mean()
        cells = [f"{g[g.hook_layer == str(L)]['flip'].mean():.1f}"
                 if len(g[g.hook_layer == str(L)]) else " - " for L in LAYERS]
        print(f"{cid:14s} {ctype:9s} {clean:4.1f}   " + "   ".join(f"{c:>3s}" for c in cells))
    m.to_csv(os.path.join(ROOT, f"sd3_concept_control_{enc}.csv"), index=False)

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
        print(f"\n===== SD3.5 detection≠control ({enc}): probe-best vs control-best layer =====")
        print(f"  {'concept':16s}{'type':10s}{'probe_best':>11s}{'control_best':>13s}{'flip':>6s}")
        for _, r in mm.sort_values(["o", "concept_id"]).iterrows():
            cbs = "—" if r.control_best < 0 else f"L{int(r.control_best)}"
            print(f"  {r.concept_id:16s}{r.ctype:10s}{'L'+str(int(r.probe_best)):>11s}{cbs:>13s}{r.control_max:>6.1f}")
        fl = mm[mm.control_best >= 0]
        rho = fl["probe_best"].corr(fl["control_best"], method="spearman") if len(fl) > 2 else float("nan")
        mm.to_csv(os.path.join(ROOT, f"sd3_detection_vs_control_{enc}.csv"), index=False)
        print(f"\n  Spearman ρ(probe_best, control_best) = {rho:.2f}  (n={len(fl)})  ≈0 => detection≠control (SD3.5/{enc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["probe", "gen", "score", "finalize"])
    ap.add_argument("--enc", default="t5", choices=["t5", "clipg", "clipl"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "probe":
        run_probe(args.enc)
    elif args.mode == "gen":
        run_gen(args.enc, args.shard, args.nshards)
    elif args.mode == "score":
        run_score(args.enc)
    else:
        run_finalize(args.enc)


if __name__ == "__main__":
    main()
