#!/usr/bin/env python3
"""Make a scaled-window final_data.jsonl from a real parquet, optionally jittered.

Used to build comparison references for the canonical TSG/HAR evaluators:
- jitter > 0  -> naive Gaussian-jitter augmentation baseline (a trivial 'generator')
- jitter = 0  -> the real windows themselves (e.g. held-out subjects -> real-vs-real floor)
Output windows are in the SDForger standardized window space (--synthetic-space scaled).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-length", type=int, default=5000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    p.add_argument("--jitter", type=float, default=0.0, help="Gaussian noise std added to scaled windows")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    df = pd.read_parquet(a.parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[a.channel], train_length=a.train_length, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=a.window_length,
                                           min_windows_number=a.min_windows_number, train_splitting=a.train_splitting)
    w = np.asarray(scaled[0], dtype=np.float64)
    if a.jitter > 0:
        rng = np.random.RandomState(a.seed)
        w = w + rng.normal(0.0, a.jitter, size=w.shape)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as h:
        for win in w:
            h.write(json.dumps({"task_name": "ts/reference", "is_seed": False, "requested_label": a.label,
                                "generated_time_series": {a.channel: [float(v) for v in win]}}) + "\n")
    print("wrote %d windows -> %s (jitter=%.2f)" % (len(w), a.output, a.jitter))


if __name__ == "__main__":
    main()
