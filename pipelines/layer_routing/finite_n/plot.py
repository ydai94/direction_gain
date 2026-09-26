"""Three static measured-evidence figures; no plot before verification."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CONFIG, OUT, read_json, write_json, digest

COLORS = {"F":"#2a78d6", "M":"#1baf7a", "A":"#eda100"}
LABELS = {"F":"Full positional", "M":"Masked positional", "A":"Pooled anchor"}
plt.rcParams.update({"font.size":8, "axes.labelsize":8, "axes.titlesize":9,
    "legend.fontsize":8, "xtick.labelsize":8, "ytick.labelsize":8,
    "axes.spines.top":False, "axes.spines.right":False, "lines.linewidth":1.2,
    "lines.markersize":6, "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none"})

def save(fig, name):
    root = OUT/"figures"
    root.mkdir(exist_ok=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(root/f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

def main():
    r=read_json(OUT/"results.json")
    v=read_json(OUT/"verification.json")
    if not v["passed"] or v["results_sha256"] != digest(OUT/"results.json"):
        raise AssertionError("results must pass verification")
    curves={(c["split"],c["arm"]):c for c in r["curves"]}
    fig,axs=plt.subplots(2,3,figsize=(7.4,4.8),constrained_layout=True)
    for row,split in enumerate(("calibration","evaluation")):
        for col,m in enumerate(("J","P","Q")):
            ax=axs[row,col]
            for fam in COLORS:
                cells=[curves[split,f"{fam}_d{i}"][m] for i in range(5)]
                mean=np.array([x["mean"] for x in cells]); low=np.array([x["ci95"][0] for x in cells]); high=np.array([x["ci95"][1] for x in cells])
                ax.errorbar(CONFIG["doses"],mean,yerr=[mean-low,high-mean],color=COLORS[fam],label=LABELS[fam],marker="o",capsize=2)
            ax.set_xscale("log",base=2); ax.set_ylim(-.04,1.04)
            ax.set_title(f"{split.title()}: {m}"); ax.set_xlabel("Budget / raw anchor norm")
            ax.set_ylabel("Case-mean rate"); ax.grid(axis="y",alpha=.15)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3)
    save(fig,"fig1_dose_curves")
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.3),constrained_layout=True)
    for ax,split in zip(axs,("calibration","evaluation")):
        for fam in COLORS:
            cells=[curves[split,f"{fam}_d{i}"] for i in range(5)]
            ax.plot([x["J"]["mean"] for x in cells],[x["P"]["mean"] for x in cells],"o-",color=COLORS[fam],label=LABELS[fam])
        for arm,marker,color in (("F2","X",COLORS["F"]),("A2","s",COLORS["A"])):
            c=curves[split,arm]; ax.scatter(c["J"]["mean"],c["P"]["mean"],marker=marker,color=color,s=65,label=arm)
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel="Target + guard success (J)",ylabel="Non-target preservation (P)",title=split.title())
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside upper center",ncol=3)
    save(fig,"fig2_efficacy_preservation")
    n=pd.read_parquet(OUT/"probe_cells.parquet").groupby(["full_id","arm"],as_index=False).N.mean()
    cases=pd.read_parquet(OUT/"case_metrics.parquet")
    joined=cases.merge(n,on=["full_id","arm"],validate="one_to_one")
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.3),constrained_layout=True)
    for ax,split in zip(axs,("calibration","evaluation")):
        for fam in COLORS:
            cell=joined[(joined.split.eq(split)) & joined.arm.str.startswith(fam+"_")].groupby("arm")[["actual_norm","N"]].median()
            ax.plot(cell.actual_norm,cell.N,"o-",color=COLORS[fam],label=LABELS[fam])
        ax.axhline(0,color="0.5",linewidth=.6); ax.set_xscale("log")
        ax.set(xlabel="Median executed tensor L2 norm",ylabel="Median case-mean finite N",title=split.title())
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside upper center",ncol=3)
    save(fig,"fig3_norm_response")
    write_json(OUT/"figures/manifest.json",[
      {"id":n,"path_pdf":n+".pdf","path_png":n+".png","path_svg":n+".svg","caption":caption,"section":"results"}
      for n,caption in [("fig1_dose_curves","Measured J/P/Q curves, 12 cases per split and three seeds; pointwise case-bootstrap 95% intervals."),
        ("fig2_efficacy_preservation","Measured efficacy-preservation points at fixed doses; lines connect doses and do not imply a continuous Pareto frontier."),
        ("fig3_norm_response","Median measured tensor norm versus finite N; N is a frozen-latent proxy, not a semantic preservation measure.")]])

if __name__ == "__main__":
    main()
