"""
Local-model alignment evaluation (no OpenRouter).

Loads Qwen3-VL-30B-A3B-Instruct from local weights and asks, for each
generated image, whether it depicts its neutral prompt. Returns a binary
aligned=true/false with a short justification — same schema as
`evaluate_alignment.py` — written to a separate "_local" JSONL/CSV so the
API-based results are not clobbered.

Splits work across shards for multi-GPU parallelism (one process per GPU).

Usage (two GPUs, one shard each):
    python evaluate_alignment_local.py --gpu 0 --shard 0 --num-shards 2 &
    python evaluate_alignment_local.py --gpu 1 --shard 1 --num-shards 2 &
    wait
    python evaluate_alignment_local.py --merge-only --exp-id 12

Single-GPU / smoke test:
    python evaluate_alignment_local.py --gpu 0 --sample 4

Defaults to exp 12 (GPT-Image 2). Pass `--exp-id N` for other experiments.
"""

import argparse
import gc
import glob
import json
import os
import re
import sys


# Must set CUDA_VISIBLE_DEVICES before importing torch.
def _parse_gpu():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"


os.environ["CUDA_VISIBLE_DEVICES"] = _parse_gpu()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from config import (  # noqa: E402
    DATA_CSV, EVAL_RESULTS_DIR, EXPERIMENT_NAMES, exp_output_dir,
)

DEFAULT_EXP_ID = 12
MODEL_PATH = (
    "/path/to/data/models/"
    "models--Qwen--Qwen3-VL-30B-A3B-Instruct/"
    "snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
)

ALIGN_PROMPT = """You are checking whether a generated image correctly depicts a text prompt.

## Prompt
"{prompt_neutral}"

## Task
Decide whether the image correctly aligns with the prompt:
- The main subject(s) of the prompt appear in the image.
- The action, setting, and attributes described are visibly depicted.
- Ignore artistic style, photorealism, and minor visual flaws.

Respond with a JSON object in this exact format:
{{"aligned": <true|false>, "justification": "<1-2 sentence explanation>"}}"""


def shard_jsonl_path(exp_id, shard, tag="alignment_local"):
    return os.path.join(
        EVAL_RESULTS_DIR,
        f"exp_{exp_id:02d}_{tag}_shard_{shard}.jsonl",
    )


def merged_paths(exp_id, tag="alignment_local"):
    base = os.path.join(EVAL_RESULTS_DIR, f"exp_{exp_id:02d}_{tag}")
    return base + ".jsonl", base + ".csv"


def load_model():
    from transformers import (
        AutoProcessor,
        Qwen3VLMoeForConditionalGeneration,
    )
    print(f"Loading Qwen3-VL-30B-A3B from {MODEL_PATH} ...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"Model loaded. Device map: {getattr(model, 'hf_device_map', '?')}")
    return model, processor


def parse_aligned(text):
    """Return (aligned: bool|None, justification: str)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            aligned = obj.get("aligned")
            if isinstance(aligned, bool):
                return aligned, obj.get("justification", "")
            if isinstance(aligned, str):
                v = aligned.strip().lower()
                if v in ("true", "yes"):
                    return True, obj.get("justification", "")
                if v in ("false", "no"):
                    return False, obj.get("justification", "")
        except json.JSONDecodeError:
            pass
    m = re.search(r'"aligned"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", "(fallback parse)"
    return None, text[:300]


def judge_single(model, processor, image_path, prompt_neutral):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": ALIGN_PROMPT.format(
            prompt_neutral=prompt_neutral)},
    ]}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
        )
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated)
    ]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated, trimmed
    gc.collect()
    torch.cuda.empty_cache()

    return parse_aligned(text)


def merge_shards(exp_id, tag="alignment_local"):
    shards = sorted(glob.glob(os.path.join(
        EVAL_RESULTS_DIR,
        f"exp_{exp_id:02d}_{tag}_shard_*.jsonl",
    )))
    out_jsonl, out_csv = merged_paths(exp_id, tag)
    rows = []
    with open(out_jsonl, "w") as fout:
        for sp in shards:
            with open(sp) as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows.append(obj)
                    fout.write(line)
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Merged {len(shards)} shard(s): {len(rows)} entries "
          f"-> {out_csv}")


def load_processed_keys(jsonl_path):
    keys = set()
    if not os.path.exists(jsonl_path):
        return keys
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aligned") is None:
                continue
            alpha = r.get("alpha")
            keys.add((
                r["case_id"],
                str(alpha) if alpha is not None else "",
                str(r["seed"]),
            ))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0",
                    help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--exp-id", type=int, default=DEFAULT_EXP_ID,
                    choices=list(EXPERIMENT_NAMES.keys()))
    ap.add_argument("--manifest", default=None,
                    help="Manifest filename within the exp dir "
                         "(default: manifest.csv). Use manifest_alpha1.csv "
                         "for the stable alpha=1.0 snapshot.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Restrict to manifest rows with this alpha value")
    ap.add_argument("--sample", type=int, default=0,
                    help="Only first N manifest rows (after sharding)")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip inference, just merge existing shard files")
    ap.add_argument("--name-tag", default="alignment_local",
                    help="Output file basename token "
                         "(produces exp_NN_<tag>{,_shard_K}.{jsonl,csv}). "
                         "Default 'alignment_local' for back-compat; pass "
                         "'alpha1_alignment' for alpha=1 results.")
    args = ap.parse_args()

    if args.merge_only:
        merge_shards(args.exp_id, args.name_tag)
        return

    exp_id = args.exp_id
    out_dir = exp_output_dir(exp_id)
    manifest_name = args.manifest or "manifest.csv"
    manifest_path = os.path.join(out_dir, manifest_name)
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Manifest: {len(manifest)} entries (exp {exp_id})")

    if args.alpha is not None:
        manifest = manifest[manifest["alpha"] == args.alpha].reset_index(
            drop=True)
        print(f"Filtered to alpha={args.alpha}: {len(manifest)} rows")

    if args.num_shards > 1:
        manifest = manifest.iloc[args.shard::args.num_shards].reset_index(
            drop=True)
        print(f"Shard {args.shard}/{args.num_shards}: {len(manifest)} rows")
    if args.sample > 0:
        manifest = manifest.head(args.sample)
        print(f"Sample: {len(manifest)} rows")

    cases_df = pd.read_csv(DATA_CSV)
    case_lookup = cases_df.set_index("id").to_dict("index")

    out_path = shard_jsonl_path(exp_id, args.shard, args.name_tag)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    processed = load_processed_keys(out_path)
    if processed:
        print(f"Resuming: skipping {len(processed)} already-scored rows")

    to_process = []
    for _, row in manifest.iterrows():
        alpha = row.get("alpha")
        key = (
            row["case_id"],
            str(alpha) if pd.notna(alpha) else "",
            str(row["seed"]),
        )
        if key in processed:
            continue
        if not os.path.exists(row["image_path"]):
            continue
        to_process.append(row)

    print(f"To process: {len(to_process)} images")
    if not to_process:
        return

    model, processor = load_model()

    fout = open(out_path, "a")
    ok = miss = 0
    pbar = tqdm(to_process, desc=f"align shard {args.shard}")
    for row in pbar:
        case_info = case_lookup.get(row["case_id"], {})
        prompt_neutral = case_info.get("prompt_neutral", "")
        if not prompt_neutral or pd.isna(prompt_neutral):
            continue

        aligned, justif = judge_single(
            model, processor, row["image_path"], prompt_neutral)

        alpha = row.get("alpha")
        rec = {
            "case_id": row["case_id"],
            "experiment_id": exp_id,
            "target": case_info.get("target", ""),
            "bias_type": case_info.get("bias_type", ""),
            "alpha": alpha if pd.notna(alpha) else None,
            "seed": int(row["seed"]),
            "image_path": row["image_path"],
            "prompt_neutral": prompt_neutral,
            "aligned": aligned,
            "justification": justif,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

        if aligned is None:
            miss += 1
        else:
            ok += 1
        pbar.set_postfix(ok=ok, unparsed=miss)

    fout.close()
    pbar.close()
    print(f"Shard {args.shard} done: {ok} scored, {miss} unparsed "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
