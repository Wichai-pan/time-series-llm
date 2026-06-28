#!/usr/bin/env python3
"""Minimal HAR utility smoke for SDForger PAMAP2 synthetic windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-real-parquet", type=Path, required=True)
    parser.add_argument("--running-real-parquet", type=Path, required=True)
    parser.add_argument("--walking-synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--running-synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--walking-stratification-csv", type=Path, required=True)
    parser.add_argument("--running-stratification-csv", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    return parser.parse_args()


def load_real_train_windows(
    parquet: Path,
    channel: str,
    train_length: int,
    window_length: int,
    min_windows_number: int,
    train_splitting: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet)
    scaled, original, _ = preprocess_train_data(
        df,
        train_channels=[channel],
        train_length=train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=window_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    train_original = np.asarray(original[0], dtype=np.float64)
    mean = np.mean(train_original, axis=0)
    std = np.std(train_original, axis=0)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return np.asarray(scaled[0], dtype=np.float64), train_original, mean, std


def load_real_test_windows(parquet: Path, channel: str, train_length: int, window_length: int) -> np.ndarray:
    df = pd.read_parquet(parquet)
    values = pd.to_numeric(df[channel], errors="raise").to_numpy(dtype=np.float64)
    heldout = values[train_length:]
    n_windows = heldout.size // window_length
    if n_windows <= 0:
        raise ValueError(f"No held-out windows available in {parquet}")
    return heldout[: n_windows * window_length].reshape(n_windows, window_length)


def load_synthetic_scaled(path: Path, channel: str) -> np.ndarray:
    windows = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            windows.append(np.asarray(record["generated_time_series"][channel], dtype=np.float64))
    return np.stack(windows, axis=0)


def inverse_transform_synthetic(scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return scaled * std + mean


def select_good(windows: np.ndarray, stratification_csv: Path) -> np.ndarray:
    rows = pd.read_csv(stratification_csv)
    good_indices = rows.loc[rows["quality_label"] == "good", "sample_index"].astype(int).to_numpy()
    if good_indices.size == 0:
        raise ValueError(f"No good samples found in {stratification_csv}")
    return windows[good_indices]


def flatten_xy(walking: np.ndarray, running: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([walking, running], axis=0)
    y = np.concatenate(
        [
            np.zeros(walking.shape[0], dtype=np.int64),
            np.ones(running.shape[0], dtype=np.int64),
        ]
    )
    return x.reshape(x.shape[0], -1), y


def evaluate_condition(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    dummy: bool = False,
) -> dict[str, object]:
    if dummy:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    labels = [0, 1]
    return {
        "condition": name,
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }


def write_report(args: argparse.Namespace, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_dir / "har_utility_smoke_results.csv", index=False)
    (args.output_dir / "har_utility_smoke_results.json").write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# HAR Utility Smoke Results",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for key, value in metadata.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Results",
        "",
        "| Condition | Train samples | Accuracy | Balanced accuracy | Macro F1 | Confusion matrix [[walk, run], ...] |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| `{condition}` | {train_samples} | {accuracy:.4f} | {balanced_accuracy:.4f} | {macro_f1:.4f} | `{confusion_matrix}` |".format(
                **row
            )
        )
    lines.append("")
    (args.output_dir / "har_utility_smoke_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _walking_scaled, walking_real_train, walking_mean, walking_std = load_real_train_windows(
        args.walking_real_parquet,
        args.channel,
        args.train_length,
        args.window_length,
        args.min_windows_number,
        args.train_splitting,
    )
    _running_scaled, running_real_train, running_mean, running_std = load_real_train_windows(
        args.running_real_parquet,
        args.channel,
        args.train_length,
        args.window_length,
        args.min_windows_number,
        args.train_splitting,
    )
    walking_test = load_real_test_windows(args.walking_real_parquet, args.channel, args.train_length, args.window_length)
    running_test = load_real_test_windows(args.running_real_parquet, args.channel, args.train_length, args.window_length)

    walking_synth = inverse_transform_synthetic(
        load_synthetic_scaled(args.walking_synthetic_jsonl, args.channel),
        walking_mean,
        walking_std,
    )
    running_synth = inverse_transform_synthetic(
        load_synthetic_scaled(args.running_synthetic_jsonl, args.channel),
        running_mean,
        running_std,
    )
    walking_synth_good = select_good(walking_synth, args.walking_stratification_csv)
    running_synth_good = select_good(running_synth, args.running_stratification_csv)

    x_test, y_test = flatten_xy(walking_test, running_test)
    x_real_train, y_real_train = flatten_xy(walking_real_train, running_real_train)
    x_synth_all, y_synth_all = flatten_xy(walking_synth, running_synth)
    x_synth_good, y_synth_good = flatten_xy(walking_synth_good, running_synth_good)
    x_real_synth_all = np.concatenate([x_real_train, x_synth_all], axis=0)
    y_real_synth_all = np.concatenate([y_real_train, y_synth_all], axis=0)
    x_real_synth_good = np.concatenate([x_real_train, x_synth_good], axis=0)
    y_real_synth_good = np.concatenate([y_real_train, y_synth_good], axis=0)

    rows = [
        evaluate_condition("majority", x_real_train, y_real_train, x_test, y_test, dummy=True),
        evaluate_condition("real-only", x_real_train, y_real_train, x_test, y_test),
        evaluate_condition("synthetic-only-all", x_synth_all, y_synth_all, x_test, y_test),
        evaluate_condition("real+synthetic-all", x_real_synth_all, y_real_synth_all, x_test, y_test),
        evaluate_condition("synthetic-only-good", x_synth_good, y_synth_good, x_test, y_test),
        evaluate_condition("real+synthetic-good", x_real_synth_good, y_real_synth_good, x_test, y_test),
    ]

    metadata = {
        "channel": args.channel,
        "train_length": args.train_length,
        "window_length": args.window_length,
        "walking_real_train_windows": int(walking_real_train.shape[0]),
        "running_real_train_windows": int(running_real_train.shape[0]),
        "walking_real_test_windows": int(walking_test.shape[0]),
        "running_real_test_windows": int(running_test.shape[0]),
        "walking_synthetic_all": int(walking_synth.shape[0]),
        "running_synthetic_all": int(running_synth.shape[0]),
        "walking_synthetic_good": int(walking_synth_good.shape[0]),
        "running_synthetic_good": int(running_synth_good.shape[0]),
        "synthetic_space_handling": "inverse_transformed_from_sdforger_scaled_space_using_activity_train_window_mean_std",
    }
    write_report(args, rows, metadata)
    print(json.dumps({"metadata": metadata, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
