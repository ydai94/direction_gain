"""Build figures/predicting.pdf (Section 4.2) and figures/cost.pdf (Section 4.5).

Both are 5.5 in wide, the manuscript \textwidth, so \includegraphics[width=\linewidth]
places them at 1:1 and the 8/7/6.5 pt ladder survives.

Inputs live in figures/inputs/ (copies of the archived analysis outputs):
  a1_a3_results.json     rho and intervals for M, text features, G, and the image pair
  a1_partial_corr.csv    partial rho of G given the image-pair improvement
  a1_probe_ablation.csv  retained rho and forward calls for each probe budget
  a1_cost_table.csv      forward calls for one clean-steered image pair
  exp276_seed_curve.csv  rho of G against the mean of 1, 2, 4, and 8 outcome seeds
  plotted_values.json    interval for rho(M, R)

Run: python3 figures/build_predicting_cost.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).parent
IN = HERE / "inputs"

MODELS = ["sd3", "flux", "qwen"]
NICE = {"sd3": "SD3.5", "flux": "FLUX", "qwen": "Qwen"}
COLOR = {"sd3": "#3B6BA5", "flux": "#D9822B", "qwen": "#3E9651"}
MARKER = {"sd3": "o", "flux": "s", "qwen": "^"}
GREY = "#888888"
SIZES = (8, 7, 6.5)


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
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def panel_letter(ax, letter):
    ax.text(-0.02, 1.12, letter, transform=ax.transAxes, fontsize=SIZES[0] + 1,
            fontweight="bold", ha="right", va="top")


def dot_rows(ax, rows, labels, xlabel, title):
    """One row per quantity, one dot and interval per model."""
    offsets = [0.22, 0.0, -0.22]
    for r, row in enumerate(rows):
        for j, m in enumerate(MODELS):
            value, lo, hi = row(m)
            y = len(rows) - 1 - r + offsets[j]
            ax.plot([lo, hi], [y, y], color=COLOR[m], lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot([value], [y], marker=MARKER[m], ms=4, color=COLOR[m],
                    mec=COLOR[m], mew=0.9, zorder=3)
    ax.axvline(0, color="0.55", lw=0.7, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def model_handles(extra=None):
    handles = [plt.Line2D([], [], marker=MARKER[m], color=COLOR[m], mec=COLOR[m],
                          ms=4, lw=0, label=NICE[m]) for m in MODELS]
    return handles + (extra or [])


def main():
    style()
    res = json.loads((IN / "a1_a3_results.json").read_text())["models"]
    plotted = json.loads((IN / "plotted_values.json").read_text())["models"]
    curve = pd.read_csv(IN / "exp276_seed_curve.csv")
    ablation = pd.read_csv(IN / "a1_probe_ablation.csv")
    cost = pd.read_csv(IN / "a1_cost_table.csv").set_index("model")
    partial = pd.read_csv(IN / "a1_partial_corr.csv").set_index("model")

    # ---- Figure: predicting which prompts improve -------------------------
    # Point estimates of rho for M, text features, and G are in the main-text
    # results table; this figure carries the intervals the table cannot show.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.68),
                             gridspec_kw=dict(width_ratios=[1.1, 1.0]))
    dot_rows(
        axes[0],
        [lambda m: (res[m]["published_check"]["rho_G"] - res[m]["published_check"]["rho_M"],
                    *res[m]["published_check"]["rho_G_minus_M_ci"]),
         lambda m: (res[m]["A3a"]["rho_G_minus_B_text"], *res[m]["A3a"]["rho_G_minus_B_text_ci"])],
        ["$G$ over response size $M$", "$G$ over text features"],
        r"difference in $\rho$ with improvement", "Paired advantage of $G$")
    axes[0].set_xlim(-0.06, 0.30)
    axes[0].set_xticks([0, 0.1, 0.2, 0.3])
    axes[0].legend(handles=model_handles(), loc="lower right", handletextpad=0.4,
                   borderaxespad=0.2, labelspacing=0.2)

    ax = axes[1]
    for m in MODELS:
        s = curve[curve.model == m].sort_values("k_seeds")
        ax.plot(s.k_seeds, s.rho_G, "-", marker=MARKER[m], color=COLOR[m], ms=3.6,
                lw=1.2, mec=COLOR[m], zorder=3)
        ax.vlines(s.k_seeds, s.rho_G_min, s.rho_G_max, color=COLOR[m], lw=0.8,
                  alpha=0.5, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_ylim(0.03, 0.30)
    ax.set_xlim(0.82, 9.8)
    ax.set_xlabel("seeds averaged")
    ax.set_ylabel(r"$\rho(G,$ improvement$)$")
    ax.set_title("Outcome noise")

    for letter, a in zip("ab", axes):
        panel_letter(a, letter)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(HERE / "predicting.pdf")
    fig.savefig(HERE / "predicting.png")

    # ---- Figure: cost and comparison with a generated image pair ---------
    fig2, axes2 = plt.subplots(1, 2, figsize=(5.5, 1.68),
                               gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    ax = axes2[0]
    for m in MODELS:
        s = ablation[ablation.model == m].sort_values("forwards")
        ax.plot(s.forwards, s.retained_pct, "-", marker=MARKER[m], color=COLOR[m],
                ms=3.6, lw=1.2, mec=COLOR[m])
        ax.axvline(cost.loc[m, "one_seed_baseline_forwards"], color=COLOR[m], lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
    ax.set_xticklabels(["10", "30", "100", "300"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(8, 360)
    ax.set_ylim(86, 102)
    ax.set_xlabel("forward calls per prompt")
    ax.set_ylabel(r"% of six-probe $\rho$")
    ax.set_title("Probe budget")
    ax.annotate("dotted: one image pair", (0.97, 0.04), xycoords="axes fraction",
                fontsize=SIZES[2], color=GREY, ha="right", va="bottom")
    # Name the two budgets quoted in the text, on the Qwen curve; the same
    # configurations occur in the same order on each curve.
    qwen = ablation[ablation.model == "qwen"]
    for (seeds, steps), label, xytext, va in [((1, 1), "1 seed, 1 step", (9, -2), "center"),
                                              ((2, 2), "2 seeds, 2 steps", (7, -9), "top")]:
        row = qwen[(qwen.n_seeds == seeds) & (qwen.n_steps == steps)].iloc[0]
        ax.annotate(label, (row.forwards, row.retained_pct), textcoords="offset points",
                    xytext=xytext, fontsize=SIZES[2], color=GREY, ha="left", va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY,
                                    shrinkA=0.5, shrinkB=2.5))

    dot_rows(
        axes2[1],
        [lambda m: (res[m]["A3b"]["rho_G_minus_one_seed"], *res[m]["A3b"]["rho_G_minus_one_seed_ci"]),
         lambda m: (partial.loc[m, "partial_G_R4_given_R3"], partial.loc[m, "ci_lo"],
                    partial.loc[m, "ci_hi"])],
        ["$G$ minus image pair", "$G$ given image pair"],
        r"$\rho$ on one held-out seed", "Against an image pair")
    axes2[1].set_xlim(-0.28, 0.26)
    axes2[1].set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    axes2[1].legend(handles=model_handles(), loc="lower left", handletextpad=0.5,
                    handlelength=1.4, borderaxespad=0.2, labelspacing=0.2)

    for letter, a in zip("ab", axes2):
        panel_letter(a, letter)
    fig2.tight_layout(w_pad=1.0)
    fig2.savefig(HERE / "cost.pdf")
    fig2.savefig(HERE / "cost.png")


if __name__ == "__main__":
    main()
