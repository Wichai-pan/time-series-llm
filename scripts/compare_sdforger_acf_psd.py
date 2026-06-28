#!/usr/bin/env python3
"""Compare ACF/PSD structure between real SDForger windows and generated windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-parquet", type=Path, required=True)
    parser.add_argument("--synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--activity", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--augmentation-strategy", default="univariate")
    parser.add_argument("--min-windows-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--max-acf-lag", type=int, default=150)
    parser.add_argument("--max-plot-frequency", type=float, default=10.0)
    parser.add_argument("--expected-period", type=int, default=None)
    return parser.parse_args()


def load_real_windows(args: argparse.Namespace) -> np.ndarray:
    df = pd.read_parquet(args.real_parquet)
    if args.channel not in df.columns:
        raise KeyError(f"{args.channel} is missing from {args.real_parquet}")
    scaled, _original, _ = preprocess_train_data(
        df,
        train_channels=[args.channel],
        train_length=args.train_length,
        train_samples=args.train_samples,
        augmentation_strategy=args.augmentation_strategy,
        min_windows_length=args.min_windows_length,
        min_windows_number=args.min_windows_number,
        train_splitting=args.train_splitting,
    )
    return np.asarray(scaled[0], dtype=np.float64)


def load_synthetic_windows(path: Path, channel: str) -> np.ndarray:
    windows = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            generated = record.get("generated_time_series", {})
            if channel not in generated:
                raise KeyError(f"{channel} missing from generated_time_series")
            windows.append(np.asarray(generated[channel], dtype=np.float64))
    if not windows:
        raise ValueError(f"No synthetic windows found in {path}")
    lengths = {window.shape[0] for window in windows}
    if len(lengths) != 1:
        raise ValueError(f"Synthetic windows have inconsistent lengths: {sorted(lengths)}")
    return np.stack(windows, axis=0)


def acf_1d(signal: np.ndarray, max_lag: int) -> np.ndarray:
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
    max_lag = min(max_lag, windows.shape[1] - 1)
    return np.stack([acf_1d(window, max_lag) for window in windows], axis=0)


def psd_matrix(windows: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = None
    powers = []
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


def summarize(
    args: argparse.Namespace,
    real: np.ndarray,
    synth: np.ndarray,
    real_acf: np.ndarray,
    synth_acf: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
    synth_psd: np.ndarray,
) -> dict[str, object]:
    lags = np.arange(real_acf.shape[1])
    real_acf_mean = real_acf.mean(axis=0)
    synth_acf_mean = synth_acf.mean(axis=0)
    real_psd_mean = real_psd.mean(axis=0)
    synth_psd_mean = synth_psd.mean(axis=0)

    real_acf_lag, real_acf_score = dominant_peak(real_acf_mean, lags, min_index=2)
    synth_acf_lag, synth_acf_score = dominant_peak(synth_acf_mean, lags, min_index=2)
    freq_mask = freqs > 0
    real_freq, real_power = dominant_peak(real_psd_mean[freq_mask], freqs[freq_mask], min_index=0)
    synth_freq, synth_power = dominant_peak(synth_psd_mean[freq_mask], freqs[freq_mask], min_index=0)

    return {
        "activity": args.activity,
        "channel": args.channel,
        "real_windows": int(real.shape[0]),
        "synthetic_windows": int(synth.shape[0]),
        "window_length": int(real.shape[1]),
        "sampling_rate_hz": args.sampling_rate,
        "expected_period": args.expected_period,
        "real_mean_acf_peak_lag": real_acf_lag,
        "real_mean_acf_peak_score": real_acf_score,
        "synthetic_mean_acf_peak_lag": synth_acf_lag,
        "synthetic_mean_acf_peak_score": synth_acf_score,
        "acf_peak_lag_abs_diff": abs(real_acf_lag - synth_acf_lag),
        "real_mean_psd_peak_hz": real_freq,
        "real_mean_psd_peak_power": real_power,
        "synthetic_mean_psd_peak_hz": synth_freq,
        "synthetic_mean_psd_peak_power": synth_power,
        "psd_peak_hz_abs_diff": abs(real_freq - synth_freq),
        "real_window_std_mean": float(np.std(real, axis=1).mean()),
        "synthetic_window_std_mean": float(np.std(synth, axis=1).mean()),
        "synthetic_abs_max": float(np.max(np.abs(synth))),
        "real_abs_max": float(np.max(np.abs(real))),
    }


def plot_comparison(
    args: argparse.Namespace,
    summary: dict[str, object],
    real_acf: np.ndarray,
    synth_acf: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
    synth_psd: np.ndarray,
) -> None:
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    lags = np.arange(real_acf.shape[1])

    real_acf_mean = real_acf.mean(axis=0)
    synth_acf_mean = synth_acf.mean(axis=0)
    real_acf_std = real_acf.std(axis=0)
    synth_acf_std = synth_acf.std(axis=0)
    real_psd_mean = real_psd.mean(axis=0)
    synth_psd_mean = synth_psd.mean(axis=0)
    real_psd_std = real_psd.std(axis=0)
    synth_psd_std = synth_psd.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    axes[0].plot(lags, real_acf_mean, label="real mean ACF", color="#0072B2", linewidth=2)
    axes[0].fill_between(
        lags,
        real_acf_mean - real_acf_std,
        real_acf_mean + real_acf_std,
        color="#0072B2",
        alpha=0.15,
    )
    axes[0].plot(lags, synth_acf_mean, label="synthetic mean ACF", color="#CC79A7", linewidth=2)
    axes[0].fill_between(
        lags,
        synth_acf_mean - synth_acf_std,
        synth_acf_mean + synth_acf_std,
        color="#CC79A7",
        alpha=0.15,
    )
    if args.expected_period is not None:
        axes[0].axvline(args.expected_period, color="black", linestyle=":", linewidth=1, label="expected period")
    axes[0].axvline(summary["real_mean_acf_peak_lag"], color="#0072B2", linestyle="--", linewidth=1)
    axes[0].axvline(summary["synthetic_mean_acf_peak_lag"], color="#CC79A7", linestyle="--", linewidth=1)
    axes[0].set_title(f"{args.activity} ACF")
    axes[0].set_xlabel("lag")
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(fontsize=8)

    freq_mask = freqs <= args.max_plot_frequency
    axes[1].plot(freqs[freq_mask], real_psd_mean[freq_mask], label="real mean PSD", color="#0072B2", linewidth=2)
    axes[1].fill_between(
        freqs[freq_mask],
        np.maximum(real_psd_mean[freq_mask] - real_psd_std[freq_mask], 0),
        real_psd_mean[freq_mask] + real_psd_std[freq_mask],
        color="#0072B2",
        alpha=0.15,
    )
    axes[1].plot(freqs[freq_mask], synth_psd_mean[freq_mask], label="synthetic mean PSD", color="#CC79A7", linewidth=2)
    axes[1].fill_between(
        freqs[freq_mask],
        np.maximum(synth_psd_mean[freq_mask] - synth_psd_std[freq_mask], 0),
        synth_psd_mean[freq_mask] + synth_psd_std[freq_mask],
        color="#CC79A7",
        alpha=0.15,
    )
    axes[1].axvline(summary["real_mean_psd_peak_hz"], color="#0072B2", linestyle="--", linewidth=1)
    axes[1].axvline(summary["synthetic_mean_psd_peak_hz"], color="#CC79A7", linestyle="--", linewidth=1)
    axes[1].set_title(f"{args.activity} PSD")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("power")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Real vs Synthetic ACF/PSD - {args.activity} {args.channel}")
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(args: argparse.Namespace, summary: dict[str, object]) -> None:
    lines = [
        f"# ACF / PSD Comparison - {args.activity} {args.channel}",
        "",
        f"- Real parquet: `{args.real_parquet}`",
        f"- Synthetic JSONL: `{args.synthetic_jsonl}`",
        f"- Plot PNG: `{args.output_prefix.with_suffix('.png')}`",
        f"- Plot PDF: `{args.output_prefix.with_suffix('.pdf')}`",
        f"- Value space: standardized SDForger window space",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| `{key}` | `{value:.6f}` |")
        else:
            lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    real = load_real_windows(args)
    synth = load_synthetic_windows(args.synthetic_jsonl, args.channel)
    if real.shape[1] != synth.shape[1]:
        raise ValueError(f"Window length mismatch: real {real.shape}, synthetic {synth.shape}")

    real_acf = acf_matrix(real, args.max_acf_lag)
    synth_acf = acf_matrix(synth, args.max_acf_lag)
    freqs, real_psd = psd_matrix(real, args.sampling_rate)
    synth_freqs, synth_psd = psd_matrix(synth, args.sampling_rate)
    if not np.allclose(freqs, synth_freqs):
        raise ValueError("Real and synthetic PSD frequency grids differ.")

    summary = summarize(args, real, synth, real_acf, synth_acf, freqs, real_psd, synth_psd)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args, summary)
    plot_comparison(args, summary, real_acf, synth_acf, freqs, real_psd, synth_psd)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
