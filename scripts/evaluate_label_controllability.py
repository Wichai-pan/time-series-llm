#!/usr/bin/env python3
"""Evaluate whether requested-label synthetic windows are classified as that label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    return parser.parse_args()


def load_real_train(parquet: Path, channel: str, train_length: int, window_length: int, min_windows_number: int, train_splitting: str):
    df = pd.read_parquet(parquet)
    _scaled, original, _ = preprocess_train_data(
        df,
        train_channels=[channel],
        train_length=train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=window_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    original = np.asarray(original[0], dtype=np.float64)
    mean = np.mean(original, axis=0)
    std = np.std(original, axis=0)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return original, mean, std


def load_synthetic(path: Path, channel: str) -> np.ndarray:
    windows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            windows.append(np.asarray(record["generated_time_series"][channel], dtype=np.float64))
    return np.stack(windows, axis=0)


def flatten_xy(walking: np.ndarray, running: np.ndarray):
    x = np.concatenate([walking, running], axis=0).reshape(walking.shape[0] + running.shape[0], -1)
    y = np.concatenate([
        np.zeros(walking.shape[0], dtype=np.int64),
        np.ones(running.shape[0], dtype=np.int64),
    ])
    return x, y


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walking_real, walking_mean, walking_std = load_real_train(
        args.walking_real_parquet, args.channel, args.train_length, args.window_length, args.min_windows_number, args.train_splitting
    )
    running_real, running_mean, running_std = load_real_train(
        args.running_real_parquet, args.channel, args.train_length, args.window_length, args.min_windows_number, args.train_splitting
    )
    x_train, y_train = flatten_xy(walking_real, running_real)

    walking_syn_scaled = load_synthetic(args.walking_synthetic_jsonl, args.channel)
    running_syn_scaled = load_synthetic(args.running_synthetic_jsonl, args.channel)
    walking_syn = walking_syn_scaled * walking_std + walking_mean
    running_syn = running_syn_scaled * running_std + running_mean
    x_syn, y_requested = flatten_xy(walking_syn, running_syn)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_syn)

    rows = []
    for label_name, label_id in [("walking", 0), ("running", 1)]:
        mask = y_requested == label_id
        rows.append(
            {
                "requested_label": label_name,
                "samples": int(mask.sum()),
                "predicted_walking": int((pred[mask] == 0).sum()),
                "predicted_running": int((pred[mask] == 1).sum()),
                "requested_label_accuracy": float((pred[mask] == label_id).mean()),
            }
        )

    summary = {
        "metadata": {
            "channel": args.channel,
            "train_length": args.train_length,
            "window_length": args.window_length,
            "walking_real_train_windows": int(walking_real.shape[0]),
            "running_real_train_windows": int(running_real.shape[0]),
            "walking_synthetic": int(walking_syn.shape[0]),
            "running_synthetic": int(running_syn.shape[0]),
            "synthetic_space_handling": "inverse_transformed_from_sdforger_scaled_space_using_requested_activity_train_window_mean_std",
        },
        "overall": {
            "accuracy": float(accuracy_score(y_requested, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_requested, pred)),
            "macro_f1": float(f1_score(y_requested, pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_requested, pred, labels=[0, 1]).tolist(),
        },
        "per_requested_label": rows,
    }
    (args.output_dir / "label_controllability.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.output_dir / "label_controllability_by_label.csv", index=False)

    lines = [
        "# Label Controllability Results",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| accuracy | {summary['overall']['accuracy']:.4f} |",
        f"| balanced_accuracy | {summary['overall']['balanced_accuracy']:.4f} |",
        f"| macro_f1 | {summary['overall']['macro_f1']:.4f} |",
        f"| confusion_matrix | `{summary['overall']['confusion_matrix']}` |",
        "",
        "## Per Requested Label",
        "",
        "| Requested label | Samples | Predicted walking | Predicted running | Requested-label accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['requested_label']} | {row['samples']} | {row['predicted_walking']} | {row['predicted_running']} | {row['requested_label_accuracy']:.4f} |"
        )
    lines.append("")
    (args.output_dir / "label_controllability.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
