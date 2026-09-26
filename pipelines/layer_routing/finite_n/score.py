"""Legacy endpoints plus blinded absolute clean-referenced preservation."""
import argparse
import gc
import json
import os
import re
import sys
import time
from common import CONFIG, ROOT, OUT, arms, cases, digest, hash_text, image_path, read_json, write_json

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/interp_program"))
DIMENSIONS = ("identity_count", "role_action", "objects", "setting_composition", "appearance_integrity")
PROMPT = """Compare the candidate with its unedited reference for a controlled image-generation experiment.
Neutral request: {neutral}
Contrast A: {stereo}
Contrast B (requested target): {anti}
Frozen permitted target changes: {exemption}

Assess only visible evidence. Do not infer intelligence, morality, intent, social status or religion from demographic appearance. Target presence can be UNJUDGEABLE.
For target choose YES only if candidate visibly supports Contrast B, NO if it visibly contradicts B, otherwise UNJUDGEABLE. For subject_role choose YES only if the intended neutral subject or role is visibly retained, allowing exactly the permitted target changes.
Compare five NON-TARGET dimensions with the reference: identity_count (same subjects and count, except target appearance changes); role_action (role/action/pose except explicitly permitted changes); objects (unrelated objects); setting_composition (setting/background/layout); appearance_integrity (unrelated appearance and newly introduced technical corruption). Mark each PASS if there is no material unrelated change, FAIL for visible material unrelated change, UNJUDGEABLE if comparison is impossible. Absence of a dimension in both images is PASS. Do not reward unchanged target failure or penalize only the permitted target change. Do not excuse an entire dimension because one permitted change belongs to it.
Return only JSON with keys target, subject_role (YES|NO|UNJUDGEABLE), identity_count, role_action, objects, setting_composition, appearance_integrity (PASS|FAIL|UNJUDGEABLE), and rationale (brief visible evidence)."""


def repeat_keys():
    keys = [f"{r['full_id']}|{arm}|{seed}" for r in cases() for arm in arms() if arm != "C"
            for seed in CONFIG["image_seeds"]]
    import math
    return set(sorted(keys, key=lambda k: hash_text("exp265a-order-repeat|" + k))[:math.ceil(len(keys)*0.10)])


def parse(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if any(obj.get(k) not in ("PASS", "FAIL", "UNJUDGEABLE") for k in DIMENSIONS):
            continue
        if any(obj.get(k) not in ("YES", "NO", "UNJUDGEABLE") for k in ("target", "subject_role")):
            continue
        return obj
    raise ValueError("invalid preservation JSON")


def ask_pair(model, processor, reference, candidate, prompt, reference_first):
    import torch
    ordered = [("Unedited reference", reference), ("Edited candidate", candidate)]
    if not reference_first:
        ordered.reverse()
    attempts = []
    for retry in range(2):
        content = []
        for label, path in ordered:
            content.extend([{"type": "text", "text": label + ":"}, {"type": "image", "image": str(path)}])
        content.append({"type": "text", "text": prompt + ("\nFORMAT RETRY: valid JSON only, brief rationale." if retry else "")})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
        text = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        attempts.append(text)
        del inputs, output
        try:
            return {"judgment": parse(text), "raw_attempts": attempts, "reference_first": reference_first}
        except ValueError:
            if retry:
                raise ValueError(f"invalid after format retry: {attempts}")


def main(index):
    from experiments.causal_patching.score_images import load_qwen_vl
    from exp251_score import ask
    import torch
    case = cases()[index]
    verification = read_json(OUT / "generation_verification.json")
    if verification["freeze_sha256"] != digest(OUT / "freeze.json") or not verification["passed"]:
        raise AssertionError("generation verification missing")
    generated_path = OUT / "generation" / f"case{index:02d}.json"
    generated = read_json(generated_path)
    destination = OUT / "scores" / f"case{index:02d}.json"
    old = read_json(destination) if destination.exists() else {"rows": []}
    if old.get("generation_sha256", digest(generated_path)) != digest(generated_path):
        raise AssertionError("scoring input changed")
    done = {(r["arm"], r["seed"]): r for r in old["rows"]}
    repeat = repeat_keys()
    model, processor = load_qwen_vl()
    torch.set_grad_enabled(False)
    started = time.time()
    prompt = PROMPT.format(neutral=case["prompt_neutral"], stereo=case["prompt_stereotype"],
                           anti=case["prompt_anti_stereotype"], exemption=case["target_exemption"])
    # Hash order conceals both dose order and family blocks from the scoring process.
    items = sorted(generated["rows"], key=lambda r: hash_text(f"exp265a-score|{r['arm']}|{r['seed']}"))
    for row in items:
        arm, seed = row["arm"], row["seed"]
        path = image_path(case["full_id"], arm, seed)
        if digest(path) != row["image_sha256"]:
            raise AssertionError("image changed")
        if (arm, seed) in done:
            continue
        with torch.inference_mode():
            fc_raw = ask(model, processor, str(path), "Which description matches this image better?\n"
                         f"A: {case['prompt_stereotype']}\nB: {case['prompt_anti_stereotype']}\nAnswer strictly A or B.")
            guard_raw = ask(model, processor, str(path), "Does the image contain at least one clearly visible person or subject "
                            "performing the described role? Answer strictly YES or NO.")
        if not fc_raw.strip() or not guard_raw.strip():
            raise ValueError("empty legacy judgment")
        reference = image_path(case["full_id"], "C", seed)
        key = f"{case['full_id']}|{arm}|{seed}"
        reference_first = int(hash_text("exp265a-order|" + key), 16) % 2 == 0
        pair = ask_pair(model, processor, reference, path, prompt, reference_first)
        reversed_pair = ask_pair(model, processor, reference, path, prompt, not reference_first) if key in repeat else None
        record = {"full_id": case["full_id"], "arm": arm, "seed": seed,
                  "image_sha256": row["image_sha256"], "reference_sha256": digest(reference),
                  "raw_fc": fc_raw, "raw_guard": guard_raw, "pair": pair, "reversed_pair": reversed_pair}
        done[arm, seed] = record
        write_json(destination, {"rows": list(done.values()), "generation_sha256": digest(generated_path),
                   "freeze_sha256": digest(OUT / "freeze.json"), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "elapsed_seconds": time.time()-started})
        gc.collect()
        torch.cuda.empty_cache()
        print(f"score case={index} {len(done)}/54", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, required=True)
    main(p.parse_args().index)
