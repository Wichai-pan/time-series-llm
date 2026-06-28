#!/usr/bin/env python3
"""Evaluate normalization ablation outputs in each run's saved model space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODES = [
    "current_activity_window_zscore",
    "joint_window_zscore",
    "global_series_zscore",
    "activity_series_zscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-root", type=Path, required=True)
    parser.add_argument("--local-output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--max-acf-lag", type=int, default=150)
    return parser.parse_args()


def acf_1d(signal: np.ndarray, max_lag: int) -> np.ndarray:
    max_lag = min(max_lag, signal.size - 1)
    centered = signal - np.mean(signal)
    denom = np.dot(centered, centered)
    if np.isclose(denom, 0.0):
        result = np.zeros(max_lag + 1, dtype=np.float64)
        result[0] = 1.0
        return result
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 : corr.size // 2 + max_lag + 1]
    counts = np.arange(signal.size, signal.size - max_lag - 1, -1, dtype=np.float64)
    return (corr / counts) / (denom / signal.size)


def acf_matrix(windows: np.ndarray, max_lag: int) -> np.ndarray:
    return np.stack([acf_1d(window, max_lag) for window in windows], axis=0)


def psd_matrix(windows: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    powers = []
    freqs = None
    for window in windows:
        f, p = welch(
            window - np.mean(window),
            fs=sampling_rate,
            nperseg=min(window.size, 300),
            scaling="density",
        )
        freqs = f
        powers.append(p)
    assert freqs is not None
    return freqs, np.stack(powers, axis=0)


def dominant_peak(values: np.ndarray, x_values: np.ndarray, min_index: int = 1) -> tuple[float, float]:
    candidates, _ = find_peaks(values[min_index:])
    candidates = candidates + min_index
    if candidates.size == 0:
        idx = int(np.argmax(values[min_index:]) + min_index)
    else:
        idx = int(candidates[np.argmax(values[candidates])])
    return float(x_values[idx]), float(values[idx])


def flat_xy(walking: np.ndarray, running: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([walking, running], axis=0)
    y = np.concatenate([
        np.zeros(walking.shape[0], dtype=np.int64),
        np.ones(running.shape[0], dtype=np.int64),
    ])
    return x.reshape(x.shape[0], -1), y


def label_controllability(
    walking_real: np.ndarray,
    running_real: np.ndarray,
    walking_syn: np.ndarray,
    running_syn: np.ndarray,
) -> dict[str, object]:
    x_train, y_train = flat_xy(walking_real, running_real)
    x_syn, y_requested = flat_xy(walking_syn, running_syn)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_syn)
    out = {
        "label_accuracy": float(accuracy_score(y_requested, pred)),
        "label_balanced_accuracy": float(balanced_accuracy_score(y_requested, pred)),
        "label_macro_f1": float(f1_score(y_requested, pred, average="macro")),
        "label_confusion_matrix": confusion_matrix(y_requested, pred, labels=[0, 1]).tolist(),
    }
    for name, label_id in [("walking", 0), ("running", 1)]:
        mask = y_requested == label_id
        out[f"{name}_requested_accuracy"] = float((pred[mask] == label_id).mean())
    return out


def rhythm_summary(activity: str, real: np.ndarray, synth: np.ndarray, sampling_rate: float, max_acf_lag: int) -> dict[str, object]:
    real_acf = acf_matrix(real, max_acf_lag)
    synth_acf = acf_matrix(synth, max_acf_lag)
    lags = np.arange(real_acf.shape[1])
    real_acf_lag, real_acf_score = dominant_peak(real_acf.mean(axis=0), lags, min_index=2)
    synth_acf_lag, synth_acf_score = dominant_peak(synth_acf.mean(axis=0), lags, min_index=2)
    freqs, real_psd = psd_matrix(real, sampling_rate)
    _freqs2, synth_psd = psd_matrix(synth, sampling_rate)
    freq_mask = freqs > 0
    real_freq, real_power = dominant_peak(real_psd.mean(axis=0)[freq_mask], freqs[freq_mask], min_index=0)
    synth_freq, synth_power = dominant_peak(synth_psd.mean(axis=0)[freq_mask], freqs[freq_mask], min_index=0)
    return {
        "activity": activity,
        "real_windows": int(real.shape[0]),
        "synthetic_windows": int(synth.shape[0]),
        "real_abs_max": float(np.max(np.abs(real))),
        "synthetic_abs_max": float(np.max(np.abs(synth))),
        "real_std_mean": float(np.std(real, axis=1).mean()),
        "synthetic_std_mean": float(np.std(synth, axis=1).mean()),
        "amplitude_ratio": float(np.max(np.abs(synth)) / max(np.max(np.abs(real)), 1e-12)),
        "real_acf_lag": real_acf_lag,
        "synthetic_acf_lag": synth_acf_lag,
        "acf_lag_diff": abs(real_acf_lag - synth_acf_lag),
        "real_acf_score": real_acf_score,
        "synthetic_acf_score": synth_acf_score,
        "real_psd_hz": real_freq,
        "synthetic_psd_hz": synth_freq,
        "psd_hz_diff": abs(real_freq - synth_freq),
        "real_psd_peak_power": real_power,
        "synthetic_psd_peak_power": synth_power,
    }


def plot_overlay(out_dir: Path, mode: str, activity: str, real: np.ndarray, synth: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    real_mean = real.mean(axis=0)
    idx = int(np.argmin(np.linalg.norm(synth - real_mean[None, :], axis=1)))
    synth_one = synth[idx]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(real_mean, color="#0072B2", linewidth=2, label="real mean window")
    ax.plot(synth_one, color="#CC79A7", linewidth=1.8, label=f"nearest synthetic #{idx}")
    ax.set_title(f"{mode} - {activity}")
    ax.set_xlabel("time index")
    ax.set_ylabel("model-space value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{mode}_{activity}_overlay.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.local_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for mode in MODES:
        run_dir = args.base_output_root / f"pamap2_subject101_norm_ablation_{mode}_{args.channel}"
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            rows.append({"mode": mode, "activity": "missing", "status": "missing_metadata"})
            continue
        walking_real = np.load(run_dir / "walking_real_windows_model_space.npy")
        running_real = np.load(run_dir / "running_real_windows_model_space.npy")
        walking_syn = np.load(run_dir / "walking_synthetic_windows_model_space.npy")
        running_syn = np.load(run_dir / "running_synthetic_windows_model_space.npy")

        metadata = json.loads(metadata_path.read_text())
        for activity, real, synth in [
            ("walking", walking_real, walking_syn),
            ("running", running_real, running_syn),
        ]:
            row = rhythm_summary(activity, real, synth, args.sampling_rate, args.max_acf_lag)
            row["mode"] = mode
            row["embedding_dims"] = metadata.get("embedding_dims")
            row["status"] = "ok"
            rows.append(row)
            plot_overlay(args.local_output_dir / "overlays", mode, activity, real, synth)

        label_row = {"mode": mode}
        label_row.update(label_controllability(walking_real, running_real, walking_syn, running_syn))
        label_rows.append(label_row)

    summary_df = pd.DataFrame(rows)
    label_df = pd.DataFrame(label_rows)
    summary_df.to_csv(args.local_output_dir / "normalization_value_rhythm_summary.csv", index=False)
    label_df.to_csv(args.local_output_dir / "normalization_label_controllability.csv", index=False)

    report = [
        "# Normalization Ablation Evaluation",
        "",
        "## Value / rhythm summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Label controllability",
        "",
        label_df.to_markdown(index=False),
        "",
    ]
    (args.local_output_dir / "normalization_ablation_evaluation.md").write_text("\n".join(report), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(label_df.to_string(index=False))


if __name__ == "__main__":
    main()
