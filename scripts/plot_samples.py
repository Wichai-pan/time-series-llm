#!/usr/bin/env python3
"""Per-sample sanity plots (advisor's ask: one sample per panel, NOT all overlaid).

Picks a few individual synthetic windows and draws each in its own subplot with a real
window overlaid for reference. Separates 'representative' samples (closest to the real
distribution) from 'worst' samples (farthest / most out-of-band), so good and bad cases
are shown side by side instead of an unreadable spaghetti overlay.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def real_windows(parquet, ch, tl):
    df = pd.read_parquet(parquet)
    s, _o, _ss = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                       augmentation_strategy="univariate", min_windows_length=300,
                                       min_windows_number=30, train_splitting="minimize-overlap")
    return np.asarray(s[0], dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-jsonl", type=Path, required=True)
    p.add_argument("--real-parquet", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--label", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=4, help="panels per row")
    a = p.parse_args()
    real = real_windows(a.real_parquet, a.channel, a.train_length)
    syn = np.array([json.loads(l)["generated_time_series"][a.channel] for l in open(a.synthetic_jsonl) if l.strip()], dtype=np.float64)
    # rank synthetic by nearest-neighbour distance to the real set (proxy for "looks real")
    nn = np.array([np.sqrt(((real - s) ** 2).sum(1)).min() for s in syn])
    order = np.argsort(nn)
    best = order[:a.n]          # most real-looking
    worst = order[-a.n:][::-1]  # least real-looking
    rmean = real.mean(0)
    fig, axes = plt.subplots(2, a.n, figsize=(3.2 * a.n, 5.2), sharex=True)
    for col, idx in enumerate(best):
        ax = axes[0, col]
        ax.plot(real[np.argmin(np.sqrt(((real - syn[idx]) ** 2).sum(1)))], color="#23b5d3", lw=1, alpha=0.7, label="real(nn)")
        ax.plot(syn[idx], color="#d96cfa", lw=1.2, label="synth")
        ax.set_title("good #%d (nn=%.1f)" % (col + 1, nn[idx]), fontsize=9)
        if col == 0: ax.set_ylabel("representative")
    for col, idx in enumerate(worst):
        ax = axes[1, col]
        ax.plot(rmean, color="#23b5d3", lw=1, alpha=0.5, label="real mean")
        ax.plot(syn[idx], color="#e8551f", lw=1.2, label="synth")
        ax.set_title("worst #%d (nn=%.1f)" % (col + 1, nn[idx]), fontsize=9)
        if col == 0: ax.set_ylabel("worst")
    axes[0, -1].legend(fontsize=7); axes[1, -1].legend(fontsize=7)
    fig.suptitle("Per-sample synthetic vs real — %s (n_syn=%d)" % (a.label, len(syn)), fontsize=11)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=120)
    print("  saved %s  (good nn med=%.1f / worst nn med=%.1f)" % (a.output.name, np.median(nn[best]), np.median(nn[worst])))


if __name__ == "__main__":
    main()
