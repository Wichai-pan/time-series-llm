#!/usr/bin/env python3
"""Generate meeting-ready figures from this week's verified results (English labels)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("reports/meeting_figures_20260629"); OUT.mkdir(parents=True, exist_ok=True)
C = {"gpt2": "#e8551f", "qwen": "#23b5d3", "base": "#9aa0a6", "hi": "#d96cfa"}


def fig_explosion():
    fig, ax = plt.subplots(figsize=(5, 3.6))
    vals = [3761, 2.98]
    bars = ax.bar(["gpt2\nraw-unified", "Qwen\nlabel-only"], vals, color=[C["gpt2"], C["qwen"]])
    ax.set_yscale("log"); ax.set_ylabel("generated window abs max (log)")
    ax.axhline(4.5, color="k", ls="--", lw=1); ax.text(1.35, 5.2, "real ~4.5", fontsize=8)
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v*1.25, str(v), ha="center", fontsize=9)
    ax.set_title("Value explosion: gpt2 explodes (3761), Qwen stable (2.98)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/"fig1_explosion.png", dpi=130); plt.close(fig)


def fig_best_acd():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    cfgs = ["clean\nbaseline", "gpt2\n+repair", "Qwen\n+repair"]
    walk = [0.165, 0.740, 0.728]; run = [0.498, 0.738, 0.893]
    x = np.arange(len(cfgs)); w = 0.36
    ax.bar(x-w/2, walk, w, label="walking", color=C["qwen"])
    ax.bar(x+w/2, run, w, label="running", color=C["hi"])
    ax.set_xticks(x); ax.set_xticklabels(cfgs); ax.set_ylabel("ACD (lower = better)")
    for i,(a,b) in enumerate(zip(walk,run)):
        ax.text(i-w/2,a+.02,f"{a:.2f}",ha="center",fontsize=8); ax.text(i+w/2,b+.02,f"{b:.2f}",ha="center",fontsize=8)
    ax.legend(); ax.set_title("Best config (train ref): Qwen+repair matches gpt2", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/"fig2_best_acd.png", dpi=130); plt.close(fig)


def fig_controllability():
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    stats = ["max\n(peak amp)", "std\n(spread)", "range", "period\n(frequency)"]
    adh = [0.399, 0.176, 0.115, -0.017]; err = [0.067, 0.087, 0.077, 0.025]
    cols = [C["qwen"], C["qwen"], C["qwen"], C["gpt2"]]
    ax.bar(stats, adh, yerr=err, color=cols, capsize=4)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("adherence (Spearman)")
    for i,v in enumerate(adh): ax.text(i, v+0.03 if v>0 else v-0.06, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("Prompt controllability: amplitude YES, frequency NO", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/"fig3_controllability.png", dpi=130); plt.close(fig)


def fig_adherence_scatter(csv):
    if not Path(csv).exists(): print("  scatter skipped (no csv)"); return
    d = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for lab, col in [("walking", C["qwen"]), ("running", C["hi"])]:
        s = d[d.label == lab]
        ax.scatter(s.requested_level, s.realized_max, s=14, alpha=0.5, color=col, label=lab)
    ax.set_xlabel("requested max level"); ax.set_ylabel("realized max")
    ax.legend(); ax.set_title("Amplitude knob works: requested up -> realized up (max)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/"fig4_adherence_scatter.png", dpi=130); plt.close(fig)


def fig_heldout():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    grp = ["gpt2 walk","gpt2 run","Qwen walk","Qwen run"]
    train = [0.74,0.74,0.73,0.89]; held = [1.40,1.74,0.98,1.46]
    x = np.arange(len(grp)); w=0.36
    ax.bar(x-w/2, train, w, label="train", color=C["base"])
    ax.bar(x+w/2, held, w, label="held-out 106/108", color=C["qwen"])
    ax.set_xticks(x); ax.set_xticklabels(grp, fontsize=8); ax.set_ylabel("ACD")
    for i,(a,b) in enumerate(zip(train,held)):
        ax.text(i-w/2,a+.02,f"{a:.2f}",ha="center",fontsize=7); ax.text(i+w/2,b+.02,f"{b:.2f}",ha="center",fontsize=7)
    ax.legend(); ax.set_title("Generalization: Qwen degrades less cross-subject (esp. running)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"fig5_heldout.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adherence-csv", default="output/time_series/qwen_s101_max_seed42_20260625/stat_adherence.csv")
    a = p.parse_args()
    fig_explosion(); fig_best_acd(); fig_controllability(); fig_adherence_scatter(a.adherence_csv); fig_heldout()
    print("figures ->", OUT)
    for f in sorted(OUT.glob("*.png")): print("  ", f.name)


if __name__ == "__main__":
    main()
