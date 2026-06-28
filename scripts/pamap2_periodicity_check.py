#!/usr/bin/env python3
"""PAMAP2 activity-level periodicity check for baseline verification."""

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
from statsmodels.tsa.stattools import acf


PAMAP2_COLUMNS = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temp",
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_acc6_x",
    "hand_acc6_y",
    "hand_acc6_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_0",
    "hand_orient_1",
    "hand_orient_2",
    "hand_orient_3",
    "chest_temp",
    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_acc6_x",
    "chest_acc6_y",
    "chest_acc6_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_0",
    "chest_orient_1",
    "chest_orient_2",
    "chest_orient_3",
    "ankle_temp",
    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
    "ankle_acc6_x",
    "ankle_acc6_y",
    "ankle_acc6_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_0",
    "ankle_orient_1",
    "ankle_orient_2",
    "ankle_orient_3",
]

ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", default=["hand_acc16_x", "ankle_acc16_x"])
    parser.add_argument("--activities", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--segment-length", type=int, default=5000)
    parser.add_argument("--max-acf-lag", type=int, default=1000)
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    return parser.parse_args()


def load_subject(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        engine="python",
    )
    return df


def clean_series(values: pd.Series, max_len: int) -> np.ndarray:
    series = pd.to_numeric(values, errors="coerce").interpolate(
        method="linear", limit_direction="both"
    )
    arr = series.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr[:max_len]


def dominant_acf(signal: np.ndarray, max_lag: int) -> tuple[int | None, float | None]:
    if signal.size < 10:
        return None, None
    max_lag = min(max_lag, signal.size // 2)
    values = acf(signal, nlags=max_lag, fft=True)
    peaks, _ = find_peaks(values[1:])
    peaks = peaks + 1
    if peaks.size == 0:
        return None, None
    best = int(peaks[np.argmax(values[peaks])])
    return best, float(values[best])


def dominant_psd(signal: np.ndarray, sampling_rate: float) -> tuple[float | None, float | None]:
    if signal.size < 32:
        return None, None
    centered = signal - np.mean(signal)
    freqs, power = welch(centered, fs=sampling_rate, nperseg=min(1024, signal.size))
    if freqs.size <= 1:
        return None, None
    idx = int(np.argmax(power[1:]) + 1)
    return float(freqs[idx]), float(power[idx])


def periodicity_label(acf_lag: int | None, acf_score: float | None, dom_freq: float | None) -> str:
    if acf_lag is None or acf_score is None or dom_freq is None:
        return "unclear"
    if acf_score >= 0.35 and dom_freq > 0:
        return "strong"
    if acf_score >= 0.2 and dom_freq > 0:
        return "moderate"
    return "weak"


def plot_triplet(
    signal: np.ndarray,
    output_path: Path,
    title: str,
    sampling_rate: float,
    max_acf_lag: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_acf_lag = min(max_acf_lag, signal.size // 2)
    acf_values = acf(signal, nlags=max_acf_lag, fft=True)
    freqs, power = welch(signal - np.mean(signal), fs=sampling_rate, nperseg=min(1024, signal.size))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    axes[0].plot(signal, linewidth=0.8)
    axes[0].set_title("Raw signal segment")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("value")

    axes[1].plot(np.arange(len(acf_values)), acf_values, linewidth=0.8)
    axes[1].set_title("Autocorrelation")
    axes[1].set_xlabel("lag")
    axes[1].set_ylabel("ACF")

    axes[2].plot(freqs, power, linewidth=0.8)
    axes[2].set_title("PSD / FFT-style frequency view")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("power")
    axes[2].set_xlim(0, min(10, freqs.max() if freqs.size else 10))

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_subject(args.input)

    rows = []
    for activity_id in args.activities:
        activity_df = df[df["activity_id"] == activity_id]
        activity_name = ACTIVITY_NAMES.get(activity_id, f"activity_{activity_id}")
        for channel in args.channels:
            signal = clean_series(activity_df[channel], args.segment_length)
            acf_lag, acf_score = dominant_acf(signal, args.max_acf_lag)
            dom_freq, dom_power = dominant_psd(signal, args.sampling_rate)
            period_samples = None if dom_freq in (None, 0) else args.sampling_rate / dom_freq
            label = periodicity_label(acf_lag, acf_score, dom_freq)

            plot_name = f"subject101_{activity_name}_{channel}_periodicity.png"
            plot_triplet(
                signal,
                args.output_dir / plot_name,
                f"subject101 {activity_name} - {channel}",
                args.sampling_rate,
                args.max_acf_lag,
            )

            rows.append(
                {
                    "subject": "101",
                    "activity_id": activity_id,
                    "activity": activity_name,
                    "channel": channel,
                    "available_rows": int(activity_df.shape[0]),
                    "segment_rows_used": int(signal.size),
                    "acf_peak_lag": acf_lag,
                    "acf_peak_score": acf_score,
                    "dominant_frequency_hz": dom_freq,
                    "dominant_frequency_power": dom_power,
                    "dominant_period_samples": period_samples,
                    "periodicity_label": label,
                    "plot": plot_name,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "periodicity_summary.csv", index=False)
    (args.output_dir / "periodicity_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    lines = [
        "# PAMAP2 Subject101 Periodicity Check",
        "",
        f"- Input: `{args.input}`",
        f"- Channels: `{', '.join(args.channels)}`",
        f"- Activities: `{', '.join(str(a) for a in args.activities)}`",
        f"- Segment length: `{args.segment_length}` samples",
        f"- Sampling rate assumption: `{args.sampling_rate}` Hz",
        "",
        "| activity | channel | rows | ACF peak lag | ACF score | dominant freq Hz | period samples | label | plot |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {activity} | {channel} | {rows} | {lag} | {score} | {freq} | {period} | {label} | `{plot}` |".format(
                activity=row["activity"],
                channel=row["channel"],
                rows=row["available_rows"],
                lag="N/A" if row["acf_peak_lag"] is None else row["acf_peak_lag"],
                score="N/A"
                if row["acf_peak_score"] is None
                else f"{row['acf_peak_score']:.3f}",
                freq="N/A"
                if row["dominant_frequency_hz"] is None
                else f"{row['dominant_frequency_hz']:.3f}",
                period="N/A"
                if row["dominant_period_samples"] is None
                else f"{row['dominant_period_samples']:.1f}",
                label=row["periodicity_label"],
                plot=row["plot"],
            )
        )
    (args.output_dir / "periodicity_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
