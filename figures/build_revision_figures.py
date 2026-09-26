"""Lay out archived selection and policy estimates at manuscript size.

No fitting, resampling, or new outcomes. Input estimates and intervals are read
from the existing plotted_values.json and Exp278 exports. Policy component
changes are already means over the full test set; selection gains are means
over selected prompts. Their denominators must remain distinct.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
plt.style.use(HERE / "paper.mplstyle")
# Preserve the model colors already used in the manuscript.
MODELS = ("sd3", "flux", "qwen")
NAMES = {"sd3": "SD3.5", "flux": "FLUX", "qwen": "Qwen-Image"}
COLORS = {"sd3": "#3B6BA5", "flux": "#D9822B", "qwen": "#3E9651"}
MARKERS = {"sd3": "o", "flux": "s", "qwen": "^"}
INK = "#263241"
GREY = "#888888"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.titleweight": "normal", "axes.titlelocation": "left",
    "legend.fontsize": 7.5, "legend.frameon": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
})


def save(fig, name, dpi=240):
    # Fixed canvas, not a tight crop: the manuscript places these at 1:1.
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=dpi, bbox_inches=None)
    plt.close(fig)


def selection(data):
    fig, ax = plt.subplots(figsize=(2.4, 1.9))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.25, top=0.97)
    for model in MODELS:
        records = data[model]["curves"]
        assert len(records) == 5
        x = [100 * row["fraction"] for row in records]
        for score, ls, fill in (("G", "-", COLORS[model]), ("M", "--", "white")):
            points = [row["strategies"][score] for row in records]
            y = [point["mean_gain"] for point in points]
            lo = [point["ci95"][0] for point in points]
            hi = [point["ci95"][1] for point in points]
            assert all(a <= v <= b for a, v, b in zip(lo, y, hi))
            ax.errorbar(x, y,
                        yerr=[[v-a for v, a in zip(y, lo)], [b-v for v, b in zip(y, hi)]],
                        ls=ls, marker=MARKERS[model], ms=6, mfc=fill,
                        mec=COLORS[model], mew=0.7, color=COLORS[model],
                        lw=0.9, elinewidth=0.65, capsize=1.8, capthick=0.65)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 25, 50, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlim=(9, 110), ylim=(-10, 64), yticks=[0, 20, 40],
           xlabel="Prompts selected (%)", ylabel="Improvement (points)")
    ax.axhline(0, color="#d9dde2", lw=0.6, zorder=0)
    blank = Line2D([], [], ls="none", marker="none", label=" ")
    ax.legend(handles=[
        Line2D([], [], color=INK, ls="-", marker="o", ms=6, lw=0.9, label="by $G$"),
        Line2D([], [], color=INK, ls="--", marker="o", mfc="white", ms=6, lw=0.9, label="by $M$"),
        blank,
    ] + [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls="none",
                ms=6, label=NAMES[m]) for m in MODELS],
              ncol=2, loc="upper right", handlelength=1.7, handletextpad=0.5,
              columnspacing=0.9, borderaxespad=0.2, labelspacing=0.35)
    save(fig, "selection")


def frontier(curves, results):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.65))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.27, top=0.76, wspace=0.36)
    for ax, model in zip(axes, MODELS):
        full = results[model]["qwen3vl"]["frozen_dev_threshold"]["steer_all"]
        end_x, end_y = -full["alignment_change_pp"], full["bias_reduction"]
        for score, ls, color, width in (("M", "--", COLORS[model], 1.0),
                                        ("B", ":", GREY, 0.85),
                                        ("G", "-", COLORS[model], 1.25)):
            rows = curves[(model, score)]
            # Endpoints come from the stored full-set result, not the final
            # subsampled CSV row (whose coverage is slightly less than one).
            ax.plot([0] + [r["align_loss_pp"] for r in rows] + [end_x],
                    [0] + [r["bias_red"] for r in rows] + [end_y],
                    ls=ls, color=color, lw=width)
        ax.plot([0, end_x], [0, end_y], color=GREY, lw=0.8)
        ax.plot(end_x, end_y, marker="o", ms=6, mfc="white", mec=INK, mew=0.7)
        ax.set_title(NAMES[model], pad=5)
        ax.set_xlabel("Prompt-matching loss (pp)", fontsize=7.5, labelpad=4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xticks({"sd3": [0, 1, 2, 3], "flux": [0, 2, 4, 6, 8],
                       "qwen": [0, 5, 10, 15, 20]}[model])
        ax.tick_params(pad=2)
    axes[0].set_ylabel("Stereotype-rating\nreduction", fontsize=8, labelpad=3)
    handles = [Line2D([], [], color=INK, lw=1.25, label="$G$"),
               Line2D([], [], color=INK, ls="--", lw=1, label="$M$"),
               Line2D([], [], color=GREY, lw=0.8, label="Random selection"),
               Line2D([], [], color=GREY, ls=":", lw=0.85, label="Clean-image $B$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.0),
               ncol=4, handlelength=2.1, columnspacing=1.5, labelspacing=0.45)
    save(fig, "policy_frontier")


def risk(curves, results):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.02), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.24, top=0.77, wspace=0.24)
    for ax, model in zip(axes, MODELS):
        all_risk = results[model]["qwen3vl"]["P_worse"]
        ax.axhline(all_risk, color=GREY, lw=0.8)
        for score, ls in (("M", "--"), ("G", "-")):
            rows = [r for r in curves[(model, score)] if r["cov"] >= 0.02]
            ax.plot([100*r["cov"] for r in rows], [r["risk"] for r in rows],
                    ls=ls, color=COLORS[model], lw=1.1)
        ax.set(title=NAMES[model], xlim=(0, 102), ylim=(0, 0.5),
               xticks=[0, 25, 50, 75, 100], xlabel="Prompts steered (%)")
        ax.tick_params(pad=2)
    axes[0].set_ylabel("Share that got worse")
    handles = [Line2D([], [], color=INK, lw=1.1, label="Ranked by $G$"),
               Line2D([], [], color=INK, ls="--", lw=1.1, label="Ranked by response size"),
               Line2D([], [], color=GREY, lw=0.8, label="Steer every prompt")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.02),
               ncol=3, handlelength=1.8, columnspacing=1.0)
    save(fig, "policy")


def qualitative():
    # Use original archived experiment PNGs, not the old 80 x 80 PDF thumbnails.
    # interpolation="none" preserves all source pixels in the vector PDF.
    examples = [
        ("...rifle in a forest", "man / woman", "8.1", "4", "0"),
        ("...guitarist performing", "rock / classical", "7.6", "4", "0"),
        ("...nurse in the patient's room", "female / male", "7.5", "5", "0"),
        ("...outside a school gate", "dropout / educated", "-5.6", "2", "2"),
        ("...Bangladesh, at a workplace", "factory / office", "-2.5", "4", "4"),
        ("...Brahmin sought for advice", "Vedas / rock", "-2.5", "4", "4"),
    ]
    fig = plt.figure(figsize=(5.06, 3.35))
    for row, (heading_y, label_y, image_y, excerpt_y, pair_y, score_y) in enumerate([
            (0.974, 0.909, 0.658, 0.634, 0.589, 0.545),
            (0.475, 0.410, 0.159, 0.135, 0.090, 0.046)]):
        band = "top" if row == 0 else "bottom"
        fig.text(0.012, heading_y, f"({'a' if row == 0 else 'b'}) Prompts from the {band} 5% of $G$",
                 ha="left", va="top", fontsize=8.5, fontweight="bold")
        for col in range(3):
            center = (col + 0.5) / 3
            i = row * 3 + col
            for side in range(2):
                x = center - 0.15 + 0.152 * side
                # 0.146 of a 5.06-in canvas is approximately 53 pt, matching
                # the displayed image size in the previous manuscript.
                ax = fig.add_axes([x, image_y, 0.146, 0.2205])
                pixels = plt.imread(HERE / "qualitative_assets" / f"image-{2*i+side:03d}.png")
                assert pixels.shape[:2] == (1024, 1024)
                ax.imshow(pixels, interpolation="none")
                ax.set_axis_off()
                fig.text(x + 0.073, label_y, "clean" if side == 0 else "steer",
                         ha="center", va="top", fontsize=7.5)
            excerpt, pair, gain, before, after = examples[i]
            fig.text(center, excerpt_y, '"' + excerpt + '"', ha="center", va="top", fontsize=7.5,
                     fontfamily="DejaVu Serif")
            fig.text(center, pair_y, pair, ha="center", va="top", fontsize=7.5)
            fig.text(center, score_y, f"$G={gain}$; rating ${before}\\to{after}$",
                     ha="center", va="top", fontsize=7.5)
    save(fig, "qualitative", dpi=600)


def main():
    paths = [INPUTS / name for name in ("plotted_values.json", "exp278_curves.csv", "exp278_results.json")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data = json.loads(paths[0].read_text())["models"]
    results = json.loads(paths[2].read_text())["models"]
    curves = {(m, s): [] for m in MODELS for s in ("G", "M", "B")}
    # Stream the CSV and retain only the primary judge and plotted methods.
    with paths[1].open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["model"], row["gate"])
            if row["judge"] == "qwen3vl" and key in curves:
                curves[key].append({name: float(row[name]) for name in
                                    ("cov", "risk", "bias_red", "align_loss_pp")})
    for rows in curves.values():
        assert len(rows) == 324
        assert all(a["cov"] < b["cov"] for a, b in zip(rows, rows[1:]))
    selection(data)
    frontier(curves, results)
    risk(curves, results)
    qualitative()
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print("Rendered selection and policy curves from unchanged estimates, and relaid out the original qualitative image pixels.")
    print(json.dumps({"input_sha256": hashes, "curve_rows_per_model_and_method": 324,
                      "minimum_text_pt": 7.5}, indent=2))


if __name__ == "__main__":
    main()
