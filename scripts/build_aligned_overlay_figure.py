import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "presentation-figures-20260525"
SRC = OUT_DIR / "source-data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL = "hand_acc16_x"
WINDOW_LENGTH = 300


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def load_jsonl_windows(path: Path) -> np.ndarray:
    windows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            windows.append(np.asarray(rec["generated_time_series"][CHANNEL], dtype=np.float64))
    return np.vstack(windows)


def best_shift_and_corr(reference: np.ndarray, candidate: np.ndarray):
    ref = zscore(reference)
    cand = zscore(candidate)
    best = None
    for shift in range(len(ref)):
        rolled = np.roll(cand, shift)
        corr = float(np.corrcoef(ref, rolled)[0, 1])
        if best is None or corr > best[0]:
            best = (corr, shift, rolled)
    return best


def align_one(reference: np.ndarray, window: np.ndarray, idx: int):
    corr, shift, aligned = best_shift_and_corr(reference, window)
    return {
        "idx": idx,
        "corr": corr,
        "shift": shift,
        "aligned": aligned,
        "raw": window,
        "abs_max": float(np.max(np.abs(window))),
    }


def pick_best_aligned(reference: np.ndarray, windows: np.ndarray):
    best = None
    for idx, window in enumerate(windows):
        item = align_one(reference, window, idx)
        if best is None or item["corr"] > best["corr"]:
            best = item
    return best


real_df = pd.read_parquet(SRC / "pamap2_subject101_running_hand_acc16_x.parquet")
real_values = real_df[CHANNEL].to_numpy(dtype=np.float64)

# Use a stable periodic window after the transient start. This is only a
# qualitative reference; synthetic samples are not paired with this window.
real_start = 1100
real_window = real_values[real_start : real_start + WINDOW_LENGTH]
real_z = zscore(real_window)

settings = {
    "clean baseline": SRC / "running_baseline_final_data.jsonl",
    "raw unified": SRC / "running_raw_unified_final_data.jsonl",
    "clip_p05_p95": ROOT / "outputs/latent-constraints-20260523/clip_p05_p95/running_final_data.jsonl",
}

selected = {}
for name, path in settings.items():
    windows = load_jsonl_windows(path)
    if name == "raw unified":
        # Use the previously identified worst raw-unified running sample to
        # make the failure mode visible in the overlay. The other settings use
        # best-aligned examples.
        selected[name] = align_one(real_window, windows[73], 73)
    else:
        selected[name] = pick_best_aligned(real_window, windows)

colors = {
    "real": "#1f77b4",
    "clean baseline": "#7f7f7f",
    "raw unified": "#d62728",
    "clip_p05_p95": "#2ca02c",
}

fig = plt.figure(figsize=(13.5, 7.9), dpi=180)
gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.05], hspace=0.48)

ax = fig.add_subplot(gs[0])
t = np.arange(WINDOW_LENGTH)
ax.plot(t, real_z, color=colors["real"], lw=2.6, label=f"real window (start={real_start})")
for name, item in selected.items():
    ax.plot(t, item["aligned"], color=colors[name], lw=2.0, alpha=0.88, label=f"{name} (sample #{item['idx']}, shift={item['shift']})")

ax.set_title("Aligned representative running windows (shape-normalized)", fontsize=17, weight="bold")
ax.set_xlabel("time index within 300-sample window")
ax.set_ylabel("z-scored value")
ax.grid(True, alpha=0.22)
ax.legend(loc="upper right", frameon=True, fontsize=9)
ax.text(
    0.0,
    -0.17,
    "All curves are individually z-scored and circularly shifted by max correlation. Raw unified uses a known worst sample to expose the failure mode.",
    transform=ax.transAxes,
    fontsize=9.5,
    color="#475569",
)

ax2 = fig.add_subplot(gs[1])
labels = ["real"] + list(selected.keys())
abs_vals = [float(np.max(np.abs(real_z)))] + [selected[k]["abs_max"] for k in selected]
corr_vals = [1.0] + [selected[k]["corr"] for k in selected]
bar_colors = [colors["real"]] + [colors[k] for k in selected.keys()]

x = np.arange(len(labels))
bars = ax2.bar(x, abs_vals, color=bar_colors, alpha=0.78)
ax2.set_yscale("symlog", linthresh=5)
ax2.set_xticks(x, labels, rotation=0)
ax2.set_ylabel("abs max\n(symlog)")
ax2.set_title("Amplitude sanity check for the selected curves", fontsize=12, weight="bold", pad=8)
ax2.grid(True, axis="y", alpha=0.2)
for i, (bar, abs_v, corr_v) in enumerate(zip(bars, abs_vals, corr_vals)):
    label = f"abs={abs_v:.2f}\ncorr={corr_v:.2f}"
    ax2.text(bar.get_x() + bar.get_width() / 2, abs_v if abs_v > 0 else 0.1, label, ha="center", va="bottom", fontsize=8)

fig.suptitle("Real vs synthetic running overlay across settings", fontsize=20, weight="bold", y=0.98)
fig.text(0.5, 0.015, "Qualitative comparison only: generated windows are not paired predictions of the real window.", ha="center", fontsize=10, color="#334155")

out_png = OUT_DIR / "running_aligned_overlay_real_baseline_raw_constrained.png"
out_pdf = OUT_DIR / "running_aligned_overlay_real_baseline_raw_constrained.pdf"
fig.savefig(out_png, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

rows = []
for name, item in selected.items():
    rows.append(
        {
            "setting": name,
            "sample_index": item["idx"],
            "circular_shift": item["shift"],
            "shape_corr_after_alignment": item["corr"],
            "raw_abs_max": item["abs_max"],
        }
    )
pd.DataFrame(rows).to_csv(OUT_DIR / "running_aligned_overlay_selected_samples.csv", index=False)
print(out_png)
