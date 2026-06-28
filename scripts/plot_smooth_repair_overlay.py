#!/usr/bin/env python3
"""Overlay clip vs shrink repair on the SAME generated windows (local, standardized space).

Shows that latent global shrink keeps a smoother, shape-preserving waveform compared to
per-dimension hard clipping, for the windows that were corrected the most.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/smooth-repair-20260615/jsonl")
CHANNEL = "hand_acc16_x"
VARIANTS = {"clip_p05_p95": "#d62728", "shrink_minmax": "#1f77b4", "shrink_p05_p95": "#2ca02c"}


def load(variant: str, label: str) -> np.ndarray:
    rows = []
    for line in (ROOT / variant / f"{label}_final_data.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(np.asarray(json.loads(line)["generated_time_series"][CHANNEL], dtype=float))
    return np.asarray(rows)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True)
    for r, label in enumerate(["walking", "running"]):
        data = {v: load(v, label) for v in VARIANTS}
        clip = data["clip_p05_p95"]
        # windows where clip's amplitude is largest = where the raw latent was most out-of-range
        idx = np.argsort(np.abs(clip).max(axis=1))[::-1][:2]
        for c, i in enumerate(idx):
            ax = axes[r, c]
            for v, color in VARIANTS.items():
                ax.plot(data[v][i], color=color, lw=1.4,
                        label=v if (r == 0 and c == 0) else None)
            ax.set_title(f"{label} — generated window #{i}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylabel("value (std)")
    axes[1, 0].set_xlabel("time index")
    axes[1, 1].set_xlabel("time index")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Clip vs latent-shrink repair on the same generated windows", y=1.08, fontsize=12)
    fig.tight_layout()
    out = Path("outputs/smooth-repair-20260615/clip_vs_shrink_overlay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
