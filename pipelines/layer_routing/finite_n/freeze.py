"""Bind text-reviewed cases and code before benchmark model execution."""
from common import ROOT, HERE, OUT, CONFIG, digest, read_json, write_json

def freeze():
    destination = OUT / "freeze.json"
    if destination.exists():
        raise RuntimeError("already frozen; preserve original before amending")
    rows = read_json(OUT / "cases_text_review.json")
    exemptions = read_json(OUT / "target_exemptions.json")
    if set(exemptions) != {r["full_id"] for r in rows}:
        raise AssertionError("target-exemption coverage")
    for r in rows:
        r["target_exemption"] = exemptions[r["full_id"]]
    write_json(OUT / "cases.json", rows)
    paths = sorted(p for p in HERE.iterdir() if p.suffix in (".py", ".json", ".slurm", ".md", ".txt"))
    paths += [OUT / name for name in ("cases.json", "data_contract.md", "experiment_design.md", "target_exemptions.json")]
    paths += [ROOT / CONFIG["source"]]
    write_json(destination, {"experiment": "Exp265A", "registration_date": "2026-09-06",
               "inputs": {str(p.relative_to(ROOT)): digest(p) for p in paths},
               "n_cases": len(rows), "n_calibration": sum(r["split"] == "calibration" for r in rows),
               "n_images": 1296, "n_probe_cells": 2592,
               "prior_outcomes": "Historical development exposure; no new Exp265A outcome yet."})
    print(digest(destination))

if __name__ == "__main__":
    freeze()
