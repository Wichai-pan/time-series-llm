#!/usr/bin/env python3
"""Evaluate all PAMAP2 settings in one canonical SDForger scaled space."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data
from evaluate_normalization_tsgbench_style import (
    calculate_acd,
    calculate_dtw,
    calculate_ed,
    calculate_kd,
    calculate_mdd,
    calculate_sd,
)


ACTIVITIES = ("walking", "running")
CHANNEL = "hand_acc16_x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-parquet", type=Path, required=True)
    parser.add_argument("--running-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--subset-mode", choices=["first", "random"], default="first")
    parser.add_argument("--subset-seed", type=int, default=42)
    return parser.parse_args()


def load_baseline_windows(
    parquet: Path,
    train_length: int,
    window_length: int,
    min_windows_number: int,
    train_splitting: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet)
    scaled, original, _ = preprocess_train_data(
        df,
        train_channels=[CHANNEL],
        train_length=train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=window_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    scaled_windows = np.asarray(scaled[0], dtype=np.float64)
    raw_windows = np.asarray(original[0], dtype=np.float64)
    mean = raw_windows.mean(axis=0)
    std = raw_windows.std(axis=0)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return scaled_windows, raw_windows, mean, std


def load_jsonl(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append(np.asarray(record["generated_time_series"][CHANNEL], dtype=np.float64))
    if not rows:
        raise ValueError(f"No generated windows in {path}")
    return np.stack(rows, axis=0)


def subset_synthetic(synthetic: np.ndarray, target_count: int, mode: str, seed: int) -> np.ndarray:
    if synthetic.shape[0] < target_count:
        raise ValueError(f"Need {target_count} synthetic windows, got {synthetic.shape[0]}")
    if synthetic.shape[0] == target_count:
        return synthetic
    if mode == "first":
        return synthetic[:target_count]
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(synthetic.shape[0]), target_count))
    return synthetic[idx]


def to_3d(windows: np.ndarray) -> np.ndarray:
    return windows[:, :, None].astype(np.float32)


def tsg_metrics(real_scaled: np.ndarray, synthetic_scaled: np.ndarray, subset_mode: str, subset_seed: int) -> dict[str, float]:
    synthetic_pair = subset_synthetic(synthetic_scaled, real_scaled.shape[0], subset_mode, subset_seed)
    real_3d = to_3d(real_scaled)
    synth_3d = to_3d(synthetic_pair)
    return {
        "MDD": calculate_mdd(real_3d, to_3d(synthetic_scaled)),
        "ACD": calculate_acd(real_3d, to_3d(synthetic_scaled)),
        "SD": calculate_sd(real_3d, to_3d(synthetic_scaled)),
        "KD": calculate_kd(real_3d, to_3d(synthetic_scaled)),
        "ED": calculate_ed(real_3d, synth_3d),
        "DTW": calculate_dtw(real_3d, synth_3d),
    }


def old_specs(root: Path) -> list[dict[str, object]]:
    return [
        ("baseline", "walking", root / "pamap2_subject101_walking_hand_acc16_x_univariate_baseline" / "final_data.jsonl"),
        ("baseline", "running", root / "pamap2_subject101_running_hand_acc16_x_univariate_baseline" / "final_data.jsonl"),
        ("label-v1", "walking", root / "pamap2_subject101_walking_hand_acc16_x_label_conditioned_univariate" / "final_data.jsonl"),
        ("label-v1", "running", root / "pamap2_subject101_running_hand_acc16_x_label_conditioned_univariate" / "final_data.jsonl"),
        ("raw-unified", "walking", root / "pamap2_subject101_unified_label_conditioned_hand_acc16_x" / "walking_final_data.jsonl"),
        ("raw-unified", "running", root / "pamap2_subject101_unified_label_conditioned_hand_acc16_x" / "running_final_data.jsonl"),
        ("clip_p05_p95", "walking", root / "pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints" / "clip_p05_p95" / "walking_final_data.jsonl"),
        ("clip_p05_p95", "running", root / "pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints" / "clip_p05_p95" / "running_final_data.jsonl"),
    ]


def norm_run_dir(root: Path, mode: str) -> Path:
    return root / f"pamap2_subject101_norm_ablation_{mode}_hand_acc16_x"


def canonicalize_norm_synthetic(
    mode: str,
    activity: str,
    model_windows: np.ndarray,
    metadata: dict[str, object],
    baseline_cache: dict[str, dict[str, np.ndarray]],
    joint_scaler: StandardScaler,
) -> np.ndarray:
    if mode == "current_activity_window_zscore":
        return model_windows

    if mode == "global_series_zscore":
        norm = metadata["normalization"]
        raw = model_windows * float(norm["global_std"]) + float(norm["global_mean"])
    elif mode == "activity_series_zscore":
        norm = metadata["normalization"]
        raw = model_windows * float(norm[f"{activity}_std"]) + float(norm[f"{activity}_mean"])
    elif mode == "joint_window_zscore":
        raw = model_windows * joint_scaler.scale_ + joint_scaler.mean_
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    mean = baseline_cache[activity]["mean"]
    std = baseline_cache[activity]["std"]
    return (raw - mean) / std


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    baseline_cache: dict[str, dict[str, np.ndarray]] = {}
    for activity, parquet in [("walking", args.walking_parquet), ("running", args.running_parquet)]:
        scaled, raw, mean, std = load_baseline_windows(
            parquet,
            args.train_length,
            args.window_length,
            args.min_windows_number,
            args.train_splitting,
        )
        baseline_cache[activity] = {"scaled": scaled, "raw": raw, "mean": mean, "std": std}

    joint_scaler = StandardScaler().fit(np.concatenate([baseline_cache["walking"]["raw"], baseline_cache["running"]["raw"]], axis=0))
    rows: list[dict[str, object]] = []

    for setting, activity, path in old_specs(args.output_root):
        synth = load_jsonl(path)
        metrics = tsg_metrics(baseline_cache[activity]["scaled"], synth, args.subset_mode, args.subset_seed)
        rows.append({
            "group": "main",
            "setting": setting,
            "activity": activity,
            "n": int(synth.shape[0]),
            "canonical_space": "activity_sdforger_scaled",
            **metrics,
        })

    for mode in [
        "current_activity_window_zscore",
        "joint_window_zscore",
        "global_series_zscore",
        "activity_series_zscore",
    ]:
        run_dir = norm_run_dir(args.output_root, mode)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        for activity in ACTIVITIES:
            model_windows = load_jsonl(run_dir / f"{activity}_final_data.jsonl")
            synth = canonicalize_norm_synthetic(mode, activity, model_windows, metadata, baseline_cache, joint_scaler)
            metrics = tsg_metrics(baseline_cache[activity]["scaled"], synth, args.subset_mode, args.subset_seed)
            rows.append({
                "group": "normalization",
                "setting": mode,
                "activity": activity,
                "n": int(synth.shape[0]),
                "canonical_space": "activity_sdforger_scaled",
                **metrics,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.report_dir / "canonical_scaled_tsgbench_summary.csv", index=False)
    (args.report_dir / "canonical_scaled_tsgbench_summary.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
