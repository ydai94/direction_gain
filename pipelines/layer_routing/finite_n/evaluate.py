"""Frozen calibration selection and exploratory paired evaluation."""
import argparse
import json
import numpy as np
import pandas as pd
from common import CONFIG, OUT, arms, cases, digest, read_json, write_json
from score import DIMENSIONS, repeat_keys

np.random.seed(CONFIG["bootstrap_seed"])
METRICS = ("Y", "H", "J", "P", "Q", "T", "Q_visible")


def metrics(row):
    judgment = row["pair"]["judgment"]
    y = int(row["raw_fc"].strip().upper()[:1] == "B")
    h = int(row["raw_guard"].strip().upper()[:3] == "YES")
    p = int(all(judgment[k] == "PASS" for k in DIMENSIONS))
    t = int(judgment["target"] == "YES")
    return {"Y": y, "H": h, "J": y*h, "P": p, "Q": y*h*p, "T": t,
            "Q_visible": t*int(judgment["subject_role"] == "YES")*p,
            "preservation_unjudgeable": int(any(judgment[k] == "UNJUDGEABLE" for k in DIMENSIONS)),
            "target_unjudgeable": int(judgment["target"] == "UNJUDGEABLE")}


def bootstrap(values, seed=CONFIG["bootstrap_seed"]):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise AssertionError("invalid bootstrap input")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(CONFIG["bootstrap_draws"], len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(), "n_cases": len(values)}


def select(calibration):
    means = calibration.groupby("arm")[[*METRICS, "actual_norm"]].mean()
    chosen = {}
    for family in ("F", "M", "A"):
        eligible = [a for a in arms() if a.startswith(family + "_") or a == family + "2"]
        chosen[family] = min(eligible, key=lambda a: (-means.loc[a, "Q"], -means.loc[a, "J"],
                            -means.loc[a, "P"], means.loc[a, "actual_norm"], a))
    return chosen


def analyze():
    data = []
    repeats = []
    generated_all = []
    probes = []
    source_hashes = {}
    expected_repeat = repeat_keys()
    observed_repeat = set()
    for case in cases():
        index = case["index"]
        genpath = OUT / "generation" / f"case{index:02d}.json"
        scorepath = OUT / "scores" / f"case{index:02d}.json"
        probepath = OUT / "probes" / f"case{index:02d}.json"
        gen = read_json(genpath)
        score = read_json(scorepath)
        probe = read_json(probepath)
        for payload in (gen, score, probe):
            if payload["freeze_sha256"] != digest(OUT / "freeze.json"):
                raise AssertionError("input freeze mismatch")
        if score["generation_sha256"] != digest(genpath):
            raise AssertionError("score source mismatch")
        for path in (genpath, scorepath, probepath):
            source_hashes[str(path.relative_to(OUT))] = digest(path)
        generated = {(r["arm"], r["seed"]): r for r in gen["rows"]}
        expected = {(a, s) for a in arms() for s in CONFIG["image_seeds"]}
        if len(score["rows"]) != 54 or {(r["arm"], r["seed"]) for r in score["rows"]} != expected or set(generated) != expected:
            raise AssertionError("endpoint rectangle")
        probe_keys = {(r["arm"], r["probe_seed"], r["step"]) for r in probe["rows"]}
        if len(probe["rows"]) != 108 or probe_keys != {(a,s,t) for a in arms() for s in CONFIG["probe_seeds"] for t in CONFIG["timesteps"]}:
            raise AssertionError("probe rectangle")
        for arm in arms():
            if probe["operators"][arm]["conditioning_sha256"] != generated[arm, CONFIG["image_seeds"][0]]["conditioning_sha256"]:
                raise AssertionError("probe/generation conditioning mismatch")
        for row in score["rows"]:
            g = generated[row["arm"], row["seed"]]
            if row["image_sha256"] != g["image_sha256"] or row["full_id"] != case["full_id"]:
                raise AssertionError("scored image mismatch")
            data.append({"full_id": case["full_id"], "triplet_key": case["triplet_key"], "split": case["split"],
                         "arm": row["arm"], "seed": row["seed"], "actual_norm": g["actual_norm"],
                         "structural_zero": g.get("structural_zero",False), **metrics(row)})
            if row["reversed_pair"] is not None:
                key = f"{case['full_id']}|{row['arm']}|{row['seed']}"
                observed_repeat.add(key)
                first = all(row["pair"]["judgment"][k] == "PASS" for k in DIMENSIONS)
                second = all(row["reversed_pair"]["judgment"][k] == "PASS" for k in DIMENSIONS)
                repeats.append(first == second)
        generated_all.extend(gen["rows"])
        probes.extend(probe["rows"])
    if observed_repeat != expected_repeat:
        raise AssertionError("repeat-order coverage")
    frame = pd.DataFrame(data)
    frame.to_parquet(OUT / "endpoints.parquet", index=False)
    pd.DataFrame(probes).to_parquet(OUT / "probe_cells.parquet", index=False)
    per_case = frame.groupby(["full_id", "triplet_key", "split", "arm"], as_index=False)[[*METRICS, "actual_norm"]].mean()
    per_case.to_parquet(OUT / "case_metrics.parquet", index=False)
    # Only calibration enters selection. Persist selected arms before evaluation contrasts.
    chosen = select(per_case[per_case.split.eq("calibration")])
    write_json(OUT / "selected_doses.json", {"selected": chosen, "rule": "Q,J,P,-norm,arm", "source_hashes": source_hashes})
    evaluation = per_case[per_case.split.eq("evaluation")]
    baseline = evaluation[evaluation.arm.eq(chosen["F"])].set_index("full_id")
    clean = evaluation[evaluation.arm.eq("C")].set_index("full_id")
    contrasts = {}
    for family in ("M", "A"):
        candidate = evaluation[evaluation.arm.eq(chosen[family])].set_index("full_id").loc[baseline.index]
        values = {m: bootstrap(candidate[m].to_numpy()-baseline[m].to_numpy()) for m in METRICS}
        values["Y_minus_clean"] = bootstrap(candidate.Y.to_numpy()-clean.loc[baseline.index].Y.to_numpy())
        values["screen_pass"] = bool(values["J"]["mean"] >= -0.05 and values["P"]["mean"] >= 0.10
                                      and values["Q"]["mean"] >= 0.05 and values["Y_minus_clean"]["mean"] >= 0.10)
        contrasts[family] = values
    quality = {"clean_P": float(frame.loc[frame.arm.eq("C"), "P"].mean()),
               "unjudgeable_fraction": float(frame.preservation_unjudgeable.mean()),
               "target_unjudgeable_fraction": float(frame.target_unjudgeable.mean()),
               "repeat_order_P_agreement": float(np.mean(repeats)), "n_repeats": len(repeats)}
    quality["masked_structural_zero_cases"] = sorted(frame.loc[frame.structural_zero,"full_id"].unique().tolist())
    quality["passed"] = bool(quality["clean_P"] >= .90 and quality["unjudgeable_fraction"] <= .10 and quality["repeat_order_P_agreement"] >= .80)
    decision = ("STOP_PRESERVATION_MEASUREMENT" if not quality["passed"] else
                "ADVANCE_LOCAL_OUTPUT_DIAGNOSTIC" if any(x["screen_pass"] for x in contrasts.values()) else
                "STOP_NO_LOCAL_OUTPUT_ADVANTAGE")
    summary, seed_values = {}, {}
    for arm in arms():
        values = frame[(frame.arm.eq(arm)) & frame.split.eq("evaluation")].groupby("seed")[list(METRICS)].mean()
        seed_values[arm] = values.to_dict("index")
        summary[arm] = {m: {"mean": float(values[m].mean()), "std": float(values[m].std(ddof=1)), "n_seeds": 3} for m in METRICS}
    curves = []
    for split in ("calibration", "evaluation"):
        for arm in arms():
            subset = per_case[(per_case.split.eq(split)) & per_case.arm.eq(arm)]
            curves.append({"split": split, "arm": arm, **{m: bootstrap(subset[m]) for m in METRICS},
                           "mean_actual_norm": float(subset.actual_norm.mean())})
    results = {"task": "output dose and non-target preservation", "dataset": "project repository frozen 24 anchors",
               "metrics": list(METRICS), "seeds": CONFIG["image_seeds"], "simulated": False,
               "provenance": {"mode": "measured", "source": "Exp265A Slurm generation/probe/scoring", "dataset_contract": "data_contract.md",
                              "freeze_sha256": digest(OUT / "freeze.json"), "source_hashes": source_hashes},
               "summary": summary, "per_seed": seed_values, "selected": chosen,
               "evaluation_contrasts": contrasts, "quality": quality, "curves": curves, "decision": decision,
               "notes": "Exploratory development screen. Pointwise descriptive case-bootstrap intervals; no confirmatory inference or causal semantic attribution."}
    write_json(OUT / "results.json", results)
    print(json.dumps({"decision": decision, "selected": chosen, "quality": quality, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    analyze()
