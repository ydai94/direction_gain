"""Build figures/policy.pdf: how often steering makes an image worse, as more
prompts are steered.

Three lines per model: ranking by G, ranking by response size M, and steering
every prompt (the horizontal reference, whose height is the share of all prompts
that get worse). The clean-image baseline and the best-possible ranking are in
the appendix table instead.

Inputs in figures/inputs/: exp278_curves.csv, exp278_results.json.
Run: python3 figures/build_policy.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
IN = HERE / "inputs"

MODELS = ["sd3", "flux", "qwen"]
NICE = {"sd3": "SD3.5", "flux": "FLUX", "qwen": "Qwen-Image"}
COLOR = {"sd3": "#3B6BA5", "flux": "#D9822B", "qwen": "#3E9651"}
GREY = "#888888"
SIZES = (8, 7, 6.5)
JUDGE = "qwen3vl"


def style():
    base, secondary, tick = SIZES
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": base,
        "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": secondary, "legend.frameon": False,
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def main():
    style()
    curves = pd.read_csv(IN / "exp278_curves.csv")
    results = json.loads((IN / "exp278_results.json").read_text())
    curves = curves[(curves.judge == JUDGE) & (curves["cov"] >= 0.02)]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85), sharey=True)
    for ax, m in zip(axes, MODELS):
        d = curves[curves.model == m]
        share_all = results["models"][m][JUDGE]["P_worse"]
        ax.axhline(share_all, color=GREY, lw=0.9, zorder=1, label="steer every prompt")
        for gate, ls, lw, label in [("M", "--", 1.1, "ranked by response size"),
                                    ("G", "-", 1.5, "ranked by $G$")]:
            s = d[d.gate == gate].sort_values("cov")
            ax.plot(100 * s["cov"], s["risk"], ls=ls, lw=lw, color=COLOR[m],
                    zorder=3 if gate == "G" else 2, label=label)
        ax.set_title(NICE[m])
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 0.5)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("prompts steered (%)")
    axes[0].set_ylabel("share that got worse")

    # Proxy handles, so the legend can be neutral grey without recolouring the data.
    proxies = [plt.Line2D([], [], color="0.35", ls="-", lw=1.5, label="ranked by $G$"),
               plt.Line2D([], [], color="0.35", ls="--", lw=1.1, label="ranked by response size"),
               plt.Line2D([], [], color=GREY, ls="-", lw=0.9, label="steer every prompt")]
    fig.legend(handles=proxies, loc="upper center", ncol=3, fontsize=SIZES[2],
               handlelength=2.0, bbox_to_anchor=(0.5, 1.10), columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "policy.pdf")
    fig.savefig(HERE / "policy.png")


if __name__ == "__main__":
    main()
