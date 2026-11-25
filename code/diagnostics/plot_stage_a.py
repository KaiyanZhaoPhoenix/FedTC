"""Produce the Stage A curves and action/weight diagnostics from held logs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics.stage_a_world import GRID, H, WINDOWS
from utils.experiment import write_json

METHODS=("static","bandit","rl")
LABELS=("Static Best","CI-Bandit (γ=0)","CI-RL (γ=0.9)")


def report(folder,balanced=False,prototype=False):
    panels=[("switch","global_acc","MNIST · regime switch"),("longtail","global_acc","CIFAR-100-LT · global"),
            ("longtail","tail20","CIFAR-100-LT · tail 20 classes")]
    if balanced:
        panels=[("longtail_balanced","global_acc","Prior-adjusted head · global"),
                ("longtail_balanced","tail20","Prior-adjusted head · tail 20 classes")]
    if prototype:
        panels=[("longtail_prototype","global_acc","Prototype initialization · global"),
                ("longtail_prototype","tail20","Prototype initialization · tail 20 classes")]
    fig,axes=plt.subplots(1,len(panels),figsize=(4*len(panels),3.8),layout="constrained")
    data={k:json.loads((folder/k/"held.json").read_text()) for k in {p[0] for p in panels}}
    for ax,(scenario,metric,title) in zip(axes,panels):
        for m,label,color,mark in zip(METHODS,LABELS,("#53585e","#187d83","#aa4229"),("o","s","^")):
            curves=np.array([[point[metric]*100 for point in row["curve"]] for row in data[scenario] if row["method"]==m])
            x=np.arange(1,WINDOWS+1)*H
            y=curves.mean(axis=0)
            if prototype:
                x=np.r_[0,x]
                y=np.r_[100*np.mean([r["initial_test"][metric] for r in data[scenario] if r["method"]==m]),y]
            ax.plot(x,y,marker=mark,markersize=4,label=label,color=color)
        ax.set(xlabel="Communication round",title=title,xticks=[5,10,20,30,40])
        ax.grid(axis="y",alpha=.23)
        if scenario=="switch":
            ax.axvline(20.5,ls="--",lw=1,color="#777777")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(fontsize=8)
    suffix="_prototype" if prototype else "_balanced" if balanced else ""
    for fmt in ("pdf","png"):
        fig.savefig(folder/f"stage_a_curves{suffix}.{fmt}",dpi=180)
    plt.close(fig)
    stats={}
    for scenario,records in data.items():
        methods={}
        for m in METHODS:
            records_m=[r for r in records if r["method"]==m]
            windows=[]
            for t in range(WINDOWS):
                points=[r["log"][t] for r in records_m]
                actions=[p["a"] for p in points]
                windows.append(dict(round_start=1+t*H,lambda_zero=sum(GRID[a][0]==0 for a in actions)/len(actions),
                    tau_mean=np.mean([GRID[a][1] for a in actions]),action_counts=dict(Counter(actions)),
                    corrupt_weight=np.mean([p["corrupted_weight"] for p in points]),
                    observed_corrupt_uploads=sum(p["corrupted_uploads"] for p in points),
                    tail_mass=np.mean([p["tail_mass"] for p in points]),available_tail=np.mean([p["available_tail"] for p in points]),
                    interval_width=np.mean([p["high"]-p["low"] for p in points])))
            methods[m]=windows
        stats[scenario]=methods
    write_json(folder/f"action_audit{suffix}.json",stats)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--balanced",action="store_true")
    modes.add_argument("--prototype",action="store_true")
    args=p.parse_args()
    report(args.folder,args.balanced,args.prototype)
