"""Frozen Exp265A data binding and stdlib provenance utilities."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DG_ROOT", str(HERE.parents[2])))
CONFIG = json.loads((HERE / "config.json").read_text())
OUT = ROOT / CONFIG["output"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def arms():
    return ["C", "F2", "A2"] + [f"{m}_d{i}" for m in ("F", "M", "A") for i in range(5)]


def cases():
    freeze = read_json(OUT / "freeze.json")
    for name, sha in freeze["inputs"].items():
        path = ROOT / name
        if digest(path) != sha:
            raise AssertionError(f"frozen input changed: {name}")
    data = read_json(OUT / "cases.json")
    if len(data) != CONFIG["n_cases"] or len({r["triplet_key"] for r in data}) != len(data):
        raise AssertionError("invalid case coverage")
    return data


def image_path(case_id, arm, seed):
    return OUT / "images" / case_id / arm / f"seed{seed}.png"


def prepare():
    import pandas as pd
    source = ROOT / CONFIG["source"]
    if digest(source) != CONFIG["source_sha256"]:
        raise AssertionError("source changed")
    columns = ["full_id", "triplet_key", "source", "bias_type", "prompt_neutral",
               "prompt_stereotype", "prompt_anti_stereotype", "output_rows", "anchor_available"]
    frame = pd.read_parquet(source, columns=columns)
    rows = frame.loc[frame.anchor_available].to_dict("records")
    if len(rows) != 45:
        raise AssertionError("source reliable-anchor count changed")
    groups = {}
    for row in rows:
        groups.setdefault((row["source"], row["bias_type"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: hash_text(CONFIG["cohort_salt"] + "|" + r["triplet_key"]))
    selected = []
    while len(selected) < CONFIG["n_cases"]:
        for key in sorted(groups):
            if groups[key] and len(selected) < CONFIG["n_cases"]:
                selected.append(groups[key].pop(0))
    calibration = {r["full_id"] for r in sorted(selected, key=lambda r: hash_text(
        CONFIG["split_salt"] + "|" + r["triplet_key"]))[:CONFIG["n_calibration"]]}
    for i, row in enumerate(selected):
        row["index"] = i
        row["split"] = "calibration" if row["full_id"] in calibration else "evaluation"
        row["output_rows"] = json.loads(row["output_rows"])
        del row["anchor_available"]
    if (OUT / "freeze.json").exists():
        raise AssertionError("cannot rewrite frozen cohort")
    write_json(OUT / "cases_text_review.json", selected)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    prepare()
