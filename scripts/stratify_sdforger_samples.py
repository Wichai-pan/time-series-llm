#!/usr/bin/env python3
"""Stratify SDForger synthetic windows into good/borderline/bad diagnostic groups."""

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
    parser.add_argument("--activity", required=True)
    parser.add_argument("--real-parquet", type=Path, required=True)
    parser.add_argument("--synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-period", type=int, required=True)
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--min-windows-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--max-acf-lag", type=int, default=150)
    parser.add_argument("--good-count", type=int, default=6)
    parser.add_argument("--bad-count", type=int, default=6)
    return parser.parse_args()


def load_real_windows(args: argparse.Namespace) -> np.ndarray:
    df = pd.read_parquet(args.real_parquet)
    scaled, _original, _ = preprocess_train_data(
        df,
        train_channels=[args.channel],
        train_length=args.train_length,
        train_samples=args.train_samples,
        augmentation_strategy="univariate",
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
            windows.append(np.asarray(record["generated_time_series"][channel], dtype=np.float64))
    return np.stack(windows, axis=0)


def acf_1d(signal: np.ndarray, max_lag: int) -> np.ndarray:
    centered = signal - np.mean(signal)
    denom = np.dot(centered, centered)
    if np.isclose(denom, 0.0):
        values = np.zeros(max_lag + 1, dtype=np.float64)
        values[0] = 1.0
        return values
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 : corr.size // 2 + max_lag + 1]
    counts = np.arange(signal.size, signal.size - max_lag - 1, -1, dtype=np.float64)
    return (corr / counts) / (denom / signal.size)


def peak_x(values: np.ndarray, xs: np.ndarray, min_index: int = 1) -> tuple[float, float]:
    peaks, _ = find_peaks(values[min_index:])
    peaks = peaks + min_index
    if peaks.size == 0:
        idx = int(np.argmax(values[min_index:]) + min_index)
    else:
        idx = int(peaks[np.argmax(values[peaks])])
    return float(xs[idx]), float(values[idx])


def window_features(window: np.ndarray, sampling_rate: float, max_acf_lag: int) -> dict[str, float]:
    max_acf_lag = min(max_acf_lag, window.size - 1)
    acf = acf_1d(window, max_acf_lag)
    lags = np.arange(acf.size)
    acf_lag, acf_score = peak_x(acf, lags, min_index=2)

    freqs, power = welch(
        window - np.mean(window),
        fs=sampling_rate,
        nperseg=min(window.size, 300),
        scaling="density",
    )
    mask = freqs > 0
    psd_hz, psd_power = peak_x(power[mask], freqs[mask], min_index=0)

    return {
        "acf_peak_lag": acf_lag,
        "acf_peak_score": acf_score,
        "psd_peak_hz": psd_hz,
        "psd_peak_power": psd_power,
        "mean": float(np.mean(window)),
        "std": float(np.std(window)),
        "abs_max": float(np.max(np.abs(window))),
    }


def classify_rows(real: np.ndarray, synth: np.ndarray, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    real_features = pd.DataFrame(
        [window_features(window, args.sampling_rate, args.max_acf_lag) for window in real]
    )
    real_mean_acf_lag = float(real_features["acf_peak_lag"].median())
    real_mean_psd_hz = float(real_features["psd_peak_hz"].median())
    real_std_median = float(real_features["std"].median())
    real_abs_max_q95 = float(real_features["abs_max"].quantile(0.95))

    rows = []
    for idx, window in enumerate(synth):
        feat = window_features(window, args.sampling_rate, args.max_acf_lag)
        acf_dist_real = abs(feat["acf_peak_lag"] - real_mean_acf_lag)
        acf_dist_expected = abs(feat["acf_peak_lag"] - args.expected_period)
        acf_dist = min(acf_dist_real, acf_dist_expected)
        psd_dist = abs(feat["psd_peak_hz"] - real_mean_psd_hz)
        std_ratio = feat["std"] / real_std_median if real_std_median > 0 else np.inf
        amp_ratio = feat["abs_max"] / real_abs_max_q95 if real_abs_max_q95 > 0 else np.inf

        acf_ok = acf_dist <= max(8.0, 0.12 * args.expected_period)
        psd_ok = psd_dist <= 0.5
        std_ok = 0.55 <= std_ratio <= 1.55
        amp_ok = amp_ratio <= 1.35
        score = (
            min(acf_dist / max(args.expected_period, 1), 2.0)
            + min(psd_dist / 1.0, 2.0)
            + abs(np.log(max(std_ratio, 1e-6)))
            + max(0.0, amp_ratio - 1.0)
        )
        label = "good" if acf_ok and psd_ok and std_ok and amp_ok else "bad"
        if label == "bad" and psd_ok and (acf_ok or std_ok) and amp_ratio <= 1.8:
            label = "borderline"

        rows.append(
            {
                "sample_index": idx,
                **feat,
                "real_reference_acf_lag_median": real_mean_acf_lag,
                "expected_period": args.expected_period,
                "acf_lag_distance": acf_dist,
                "psd_peak_hz_distance": psd_dist,
                "std_ratio_to_real_median": std_ratio,
                "amp_ratio_to_real_absmax_q95": amp_ratio,
                "quality_score": score,
                "quality_label": label,
            }
        )

    reference = {
        "real_acf_lag_median": real_mean_acf_lag,
        "real_psd_hz_median": real_mean_psd_hz,
        "real_std_median": real_std_median,
        "real_abs_max_q95": real_abs_max_q95,
    }
    return pd.DataFrame(rows).sort_values("quality_score"), reference


def plot_examples(
    synth: np.ndarray,
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    count: int,
) -> None:
    subset = rows.head(count) if "best" in title.lower() else rows.tail(count).iloc[::-1]
    fig, axes = plt.subplots(len(subset), 1, figsize=(11, max(2.0 * len(subset), 4)), sharex=True)
    if len(subset) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, subset.iterrows()):
        idx = int(row["sample_index"])
        ax.plot(synth[idx], color="#CC79A7", linewidth=1.2)
        ax.set_ylabel(f"#{idx}")
        ax.set_title(
            f"{row['quality_label']} score={row['quality_score']:.3f}, "
            f"ACF lag={row['acf_peak_lag']:.0f}, PSD={row['psd_peak_hz']:.2f}Hz, "
            f"amp ratio={row['amp_ratio_to_real_absmax_q95']:.2f}",
            fontsize=9,
        )
    axes[-1].set_xlabel("time index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(args: argparse.Namespace, rows: pd.DataFrame, reference: dict[str, float]) -> None:
    counts = rows["quality_label"].value_counts().to_dict()
    lines = [
        f"# Synthetic Sample Stratification - {args.activity} {args.channel}",
        "",
        f"- Real parquet: `{args.real_parquet}`",
        f"- Synthetic JSONL: `{args.synthetic_jsonl}`",
        f"- Expected period: `{args.expected_period}`",
        f"- Samples: `{len(rows)}`",
        "",
        "## Reference",
        "",
        "| Reference metric | Value |",
        "|---|---:|",
    ]
    for key, value in reference.items():
        lines.append(f"| `{key}` | `{value:.6f}` |")
    lines += [
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label in ["good", "borderline", "bad"]:
        lines.append(f"| `{label}` | `{counts.get(label, 0)}` |")
    lines += [
        "",
        "## Best Samples",
        "",
        rows.head(10).to_markdown(index=False),
        "",
        "## Worst Samples",
        "",
        rows.tail(10).iloc[::-1].to_markdown(index=False),
        "",
    ]
    (args.output_dir / f"{args.activity}_{args.channel}_sample_stratification.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = load_real_windows(args)
    synth = load_synthetic_windows(args.synthetic_jsonl, args.channel)
    rows, reference = classify_rows(real, synth, args)

    csv_path = args.output_dir / f"{args.activity}_{args.channel}_sample_stratification.csv"
    json_path = args.output_dir / f"{args.activity}_{args.channel}_sample_stratification.json"
    rows.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "activity": args.activity,
                "channel": args.channel,
                "reference": reference,
                "label_counts": rows["quality_label"].value_counts().to_dict(),
                "samples": rows.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(args, rows, reference)
    plot_examples(
        synth,
        rows,
        args.output_dir / f"{args.activity}_{args.channel}_best_samples.png",
        f"{args.activity} best synthetic samples",
        args.good_count,
    )
    plot_examples(
        synth,
        rows,
        args.output_dir / f"{args.activity}_{args.channel}_worst_samples.png",
        f"{args.activity} worst synthetic samples",
        args.bad_count,
    )
    print(json.dumps({"activity": args.activity, "label_counts": rows["quality_label"].value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
