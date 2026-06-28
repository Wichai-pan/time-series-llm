#!/usr/bin/env python3
"""Held-out-SUBJECT HAR utility: train on 101/102/105 (+synthetic), test on 106/108.

Unlike evaluate_har_utility_smoke.py (test = held-out segment of the SAME subject),
this uses genuinely unseen SUBJECTS for the test set — the meaningful generalization
claim for synthetic-data augmentation. Conditions: majority / real-only /
synthetic-only / real+synthetic, walking-vs-running, single channel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walking-train-parquet", type=Path, required=True)
    p.add_argument("--running-train-parquet", type=Path, required=True)
    p.add_argument("--walking-test-parquet", type=Path, required=True)
    p.add_argument("--running-test-parquet", type=Path, required=True)
    p.add_argument("--walking-synthetic-jsonl", type=Path, required=True)
    p.add_argument("--running-synthetic-jsonl", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    return p.parse_args()


def real_train_windows(parquet, channel, tl, wl, mwn, ts):
    df = pd.read_parquet(parquet)
    scaled, original, _ = preprocess_train_data(df, train_channels=[channel], train_length=tl, train_samples=1,
                                                augmentation_strategy="univariate", min_windows_length=wl,
                                                min_windows_number=mwn, train_splitting=ts)
    raw = np.asarray(original[0], dtype=np.float64)
    mean = raw.mean(axis=0); std = np.where(np.isclose(raw.std(axis=0), 0.0), 1.0, raw.std(axis=0))
    return raw, mean, std


def test_windows(parquet, channel, wl):
    df = pd.read_parquet(parquet)
    v = pd.to_numeric(df[channel], errors="raise").to_numpy(dtype=np.float64)
    v = v[~np.isnan(v)]
    n = v.size // wl
    if n <= 0:
        raise ValueError(f"no test windows in {parquet}")
    return v[: n * wl].reshape(n, wl)


def synth_windows(path, channel, mean, std):
    w = [np.asarray(json.loads(l)["generated_time_series"][channel], dtype=np.float64)
         for l in open(path) if l.strip()]
    return np.stack(w, axis=0) * std + mean  # inverse-transform scaled -> raw-like


def xy(walk, run):
    x = np.concatenate([walk, run], axis=0)
    y = np.concatenate([np.zeros(len(walk), int), np.ones(len(run), int)])
    return x.reshape(x.shape[0], -1), y


def evaluate(name, xtr, ytr, xte, yte, dummy=False):
    clf = DummyClassifier(strategy="most_frequent") if dummy else make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    clf.fit(xtr, ytr); pred = clf.predict(xte)
    return {"condition": name, "train_samples": int(xtr.shape[0]), "test_samples": int(xte.shape[0]),
            "accuracy": float(accuracy_score(yte, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
            "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist()}


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    walk_tr, w_mean, w_std = real_train_windows(a.walking_train_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
    run_tr, r_mean, r_std = real_train_windows(a.running_train_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
    walk_te = test_windows(a.walking_test_parquet, a.channel, a.window_length)
    run_te = test_windows(a.running_test_parquet, a.channel, a.window_length)
    walk_sy = synth_windows(a.walking_synthetic_jsonl, a.channel, w_mean, w_std)
    run_sy = synth_windows(a.running_synthetic_jsonl, a.channel, r_mean, r_std)

    x_te, y_te = xy(walk_te, run_te)
    x_re, y_re = xy(walk_tr, run_tr)
    x_sy, y_sy = xy(walk_sy, run_sy)
    x_rs, y_rs = np.concatenate([x_re, x_sy]), np.concatenate([y_re, y_sy])

    rows = [
        evaluate("majority", x_re, y_re, x_te, y_te, dummy=True),
        evaluate("real-only", x_re, y_re, x_te, y_te),
        evaluate("synthetic-only", x_sy, y_sy, x_te, y_te),
        evaluate("real+synthetic", x_rs, y_rs, x_te, y_te),
    ]
    meta = {"channel": a.channel, "test_subjects": "106/108 (held-out)", "train_subjects": "101/102/105",
            "walking_train": int(len(walk_tr)), "running_train": int(len(run_tr)),
            "walking_test": int(len(walk_te)), "running_test": int(len(run_te)),
            "walking_synth": int(len(walk_sy)), "running_synth": int(len(run_sy))}
    pd.DataFrame(rows).to_csv(a.output_dir / "har_utility_heldout_results.csv", index=False)
    (a.output_dir / "har_utility_heldout_results.json").write_text(json.dumps({"metadata": meta, "results": rows}, indent=2) + "\n")
    print(json.dumps({"metadata": meta, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
