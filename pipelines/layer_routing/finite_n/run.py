"""Checkpointed Exp265A GPU stages: smoke, generation, and finite N."""
import argparse
import os
import time
from pathlib import Path
from common import CONFIG, OUT, arms, cases, digest, image_path, read_json, write_json


def hardware():
    import torch
    import subprocess
    return {"gpu": torch.cuda.get_device_name(0), "total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_bytes": torch.cuda.max_memory_allocated(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}


def check_smoke():
    payload = read_json(OUT / "smoke" / "result.json")
    if not payload["passed"] or payload["freeze_sha256"] != digest(OUT / "freeze.json"):
        raise AssertionError("passing smoke for this freeze is required")


def probe_case(pipe, case, edits, tensors, seeds):
    import torch
    import exp256_gradient_screen_qwen as core
    core.TIMESTEPS = tuple(CONFIG["timesteps"])
    core.NUM_STEPS = CONFIG["steps"]
    mask, stereo, smask, anti, amask = tensors
    empty, emask = core.encode(pipe, " ")
    records = []
    for seed in seeds:
        keep, shapes = core.trajectory(pipe, edits["C"], mask, empty, emask, seed)
        for step in CONFIG["timesteps"]:
            latent, timestep = keep[step]
            def velocity(embedding, attention_mask, label):
                return core.velocity(pipe.transformer, latent, timestep, embedding, attention_mask,
                                     shapes, context=f"exp265_{label}").float().flatten()
            baseline = velocity(edits["C"], mask, "neutral")
            target = velocity(anti, amask, "anti") - velocity(stereo, smask, "stereo")
            magnitude = float(target.norm())
            if not torch.isfinite(target).all() or magnitude <= 1e-8:
                raise AssertionError("invalid finite-N reference")
            for arm in arms():
                response = velocity(edits[arm], mask, arm) - baseline if arm != "C" else torch.zeros_like(baseline)
                gain = float(torch.dot(response, target / magnitude))
                response_norm = float(response.norm())
                if not torch.isfinite(response).all():
                    raise AssertionError("nonfinite response")
                records.append({"full_id": case["full_id"], "arm": arm, "probe_seed": seed,
                                "step": step, "N": gain, "response_norm": response_norm,
                                "cosine": gain / max(response_norm, 1e-8), "target_norm": magnitude})
    return records


def main(stage, index):
    import torch
    from model import encode_case, load_pipe, render
    torch.set_grad_enabled(False)
    # Validate all bound files even for nonbenchmark smoke; do not encode benchmark prompts.
    frozen = cases()
    started = time.time()
    if stage != "smoke":
        check_smoke()
        case = frozen[index]
    else:
        case = {"full_id": "nonbenchmark_owl", "output_rows": [2],
                "prompt_neutral": "A ceramic owl on a wooden shelf beside a tiny clock.",
                "prompt_stereotype": "A red ceramic owl on a wooden shelf beside a tiny clock.",
                "prompt_anti_stereotype": "A blue ceramic owl on a wooden shelf beside a tiny clock."}
    pipe = load_pipe()
    edits, metadata, tensors = encode_case(pipe, case)
    if set(edits) != set(arms()):
        raise AssertionError("operator coverage")
    if stage == "smoke":
        from exp264b_generate import full_output_edit
        from exp264b_common import native_add_delta, bool_mask, output_rows_for_pole, unit_direction
        mask, stereo, _sm, anti, _am = tensors
        old_f, _ = full_output_edit(edits["C"], stereo, anti)
        sr, ar = output_rows_for_pole(pipe.tokenizer, case["prompt_stereotype"], case["prompt_anti_stereotype"])
        unit, _ = unit_direction(stereo, sr, anti, ar)
        old_a, _ = native_add_delta(edits["C"], unit, bool_mask(edits["C"].shape[1], case["output_rows"], device="cuda"))
        if not torch.equal(old_f, edits["F2"]) or not torch.equal(old_a, edits["A2"]):
            raise AssertionError("legacy operator parity")
        from model import match_budget, positional
        zero, _ = match_budget(edits["C"], positional(edits["C"], stereo, anti), 0)
        if not torch.equal(zero, edits["C"]):
            raise AssertionError("zero-dose identity")
        root = OUT / "smoke"
        root.mkdir(parents=True, exist_ok=True)
        for arm in ("C", "F2", "M_d3", "A_d3"):
            image = render(pipe, edits[arm], mask, 265)
            if image.size != (1024, 1024):
                raise AssertionError("image dimensions")
            image.save(root / f"{arm}.png")
        cells = probe_case(pipe, case, edits, tensors, [6])
        write_json(root / "probe.json", cells)
        write_json(root / "operators.json", metadata)
        absent = all(p.grad is None for m in (pipe.transformer, pipe.text_encoder) for p in m.parameters())
        if not absent:
            raise AssertionError("model parameter gradients")
        write_json(root / "result.json", {"passed": True, "freeze_sha256": digest(OUT / "freeze.json"),
                   "checks": ["legacy_F2_parity", "legacy_A2_parity", "zero_identity", "norm_matching",
                              "anchor_support", "18_arms", "36_finite_probe_cells", "four_valid_images", "no_parameter_grads"],
                   "elapsed_seconds": time.time()-started, **hardware()})
    elif stage == "generate":
        destination = OUT / "generation" / f"case{index:02d}.json"
        old = read_json(destination) if destination.exists() else {"rows": []}
        if old.get("freeze_sha256", digest(OUT / "freeze.json")) != digest(OUT / "freeze.json"):
            raise AssertionError("checkpoint freeze changed")
        done = {(r["arm"], r["seed"]): r for r in old["rows"]}
        rendered = {(r["conditioning_sha256"],r["seed"]):r for r in old["rows"]}
        for arm in arms():
            for seed in CONFIG["image_seeds"]:
                path = image_path(case["full_id"], arm, seed)
                if (arm, seed) in done:
                    if digest(path) != done[arm, seed]["image_sha256"] or metadata[arm]["conditioning_sha256"] != done[arm, seed]["conditioning_sha256"]:
                        raise AssertionError("generation checkpoint changed")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                identity_key = (metadata[arm]["conditioning_sha256"],seed)
                source = rendered.get(identity_key)
                if source:
                    import shutil
                    if digest(source["image_path"]) != source["image_sha256"]:
                        raise AssertionError("identity reuse source changed")
                    shutil.copyfile(source["image_path"],path)
                else:
                    image = render(pipe, edits[arm], tensors[0], seed)
                    if image.size != (1024, 1024):
                        raise AssertionError("image size")
                    image.save(path)
                done[arm, seed] = {"full_id": case["full_id"], "arm": arm, "seed": seed,
                                   "image_path": str(path), "image_sha256": digest(path),
                                   "identity_reuse_arm": source["arm"] if source else None, **metadata[arm]}
                rendered[identity_key] = done[arm,seed]
                write_json(destination, {"freeze_sha256": digest(OUT / "freeze.json"), "rows": list(done.values()),
                           "elapsed_seconds": time.time()-started, **hardware()})
                print(f"generate case={index} arm={arm} seed={seed} {len(done)}/54", flush=True)
    elif stage == "probe":
        destination = OUT / "probes" / f"case{index:02d}.json"
        if destination.exists():
            old = read_json(destination)
            if old["freeze_sha256"] != digest(OUT / "freeze.json") or old["operators"] != metadata:
                raise AssertionError("probe checkpoint changed")
            if len(old["rows"]) != 108:
                raise AssertionError("probe checkpoint incomplete")
            return
        cells = probe_case(pipe, case, edits, tensors, CONFIG["probe_seeds"])
        write_json(destination, {"rows": cells, "operators": metadata,
                   "freeze_sha256": digest(OUT / "freeze.json"), "elapsed_seconds": time.time()-started, **hardware()})
    print(f"{stage} complete {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "generate", "probe"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    main(args.stage, args.index)
