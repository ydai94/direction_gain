"""Independent coverage, hash, norm and endpoint-mean checks."""
import argparse
import itertools
import math
from collections import defaultdict
from common import CONFIG, OUT, arms, cases, digest, image_path, read_json, write_json


def verify_images():
    from PIL import Image
    total = 0
    for case in cases():
        data = read_json(OUT / "generation" / f"case{case['index']:02d}.json")
        expected = set(itertools.product(arms(), CONFIG["image_seeds"]))
        if len(data["rows"]) != len(expected) or {(r["arm"],r["seed"]) for r in data["rows"]} != expected:
            raise AssertionError("generation coverage")
        for r in data["rows"]:
            path = image_path(case["full_id"], r["arm"], r["seed"])
            if r["full_id"] != case["full_id"] or digest(path) != r["image_sha256"]:
                raise AssertionError("image provenance")
            with Image.open(path) as im:
                if im.size != (1024, 1024) or im.format != "PNG":
                    raise AssertionError("invalid image")
                im.verify()
            if r.get("structural_zero", False) and (not r["arm"].startswith("M_") or r["actual_norm"] != 0):
                raise AssertionError("invalid structural-zero operator")
            if r["requested_norm"] is not None and not r.get("structural_zero",False) and abs(r["actual_norm"] / r["requested_norm"] - 1) > CONFIG["norm_tolerance"]:
                raise AssertionError("norm not matched")
            if r["arm"].startswith(("M", "A")) and not set(r["active_rows"]).issubset(case["output_rows"]):
                raise AssertionError("mask support")
            if r["arm"] == "C" and r["actual_norm"] != 0:
                raise AssertionError("clean changed")
            total += 1
    result = {"passed": True, "n_images": total, "freeze_sha256": digest(OUT / "freeze.json")}
    if total != 1296:
        raise AssertionError("total image count")
    write_json(OUT / "generation_verification.json", result)
    return result


def verify_results():
    import numpy as np
    result = read_json(OUT / "results.json")
    # Reconstruct endpoints without using evaluate.metrics or its groupby code.
    fields = ("identity_count", "role_action", "objects", "setting_composition", "appearance_integrity")
    metric_rows = defaultdict(list)
    case_rows = defaultdict(list)
    for case in cases():
        payload = read_json(OUT / "scores" / f"case{case['index']:02d}.json")
        for row in payload["rows"]:
            jdg = row["pair"]["judgment"]
            y = float(row["raw_fc"].strip().upper().startswith("B"))
            h = float(row["raw_guard"].strip().upper().startswith("YES"))
            p = float(all(jdg[k] == "PASS" for k in fields))
            t = float(jdg["target"] == "YES")
            metrics = {"Y": y, "H": h, "J": y*h, "P": p, "Q": y*h*p, "T": t,
                       "Q_visible": t*float(jdg["subject_role"] == "YES")*p}
            if case["split"] == "evaluation":
                for m, value in metrics.items():
                    metric_rows[row["arm"],m].append(value)
                    case_rows[row["arm"],m,case["full_id"]].append(value)
    for (arm,m), values in metric_rows.items():
        if not math.isclose(sum(values)/len(values), result["summary"][arm][m]["mean"], abs_tol=1e-12):
            raise AssertionError("summary reconstruction")
    eval_ids = sorted(c["full_id"] for c in cases() if c["split"] == "evaluation")
    for family in ("M","A"):
        a,b = result["selected"][family], result["selected"]["F"]
        for m in result["metrics"]:
            differences = np.array([sum(case_rows[a,m,c])/3-sum(case_rows[b,m,c])/3 for c in eval_ids])
            rng = np.random.default_rng(CONFIG["bootstrap_seed"])
            means = np.mean(differences[rng.integers(0,12,(10000,12))],axis=1)
            expected = result["evaluation_contrasts"][family][m]
            if not np.allclose(np.quantile(means,[.025,.975]), expected["ci95"], atol=1e-12) or not math.isclose(float(differences.mean()),expected["mean"],abs_tol=1e-12):
                raise AssertionError("bootstrap reconstruction")
    for name, sha in result["provenance"]["source_hashes"].items():
        if digest(OUT/name) != sha:
            raise AssertionError("analysis input changed")
    payload = {"passed": True, "checks": ["image_hashes", "1296_images", "norms", "support", "clean_identity",
               "all_metric_means", "14_paired_bootstraps", "input_hashes"], "results_sha256": digest(OUT / "results.json")}
    write_json(OUT / "verification.json", payload)
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["images", "results"])
    args=p.parse_args()
    print(verify_images())
    if args.stage == "results":
        print(verify_results())
