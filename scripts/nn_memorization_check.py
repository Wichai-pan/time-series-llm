#!/usr/bin/env python3
"""Memorization check: nearest-neighbour distance of synthetic windows to the TRAIN set.

If synthetic windows sit much closer to the nearest train window than real held-out
windows do, the generator is copying/lightly perturbing training data rather than
modelling the distribution. Reference = real-held-out-to-train NN distance.
All windows are compared in the SDForger standardized window space.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def scaled_windows(parquet, ch, tl, wl=300, mwn=30, ts="minimize-overlap"):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def load_jsonl(path, ch):
    return np.array([json.loads(l)["generated_time_series"][ch] for l in open(path) if l.strip()], dtype=np.float64)


def nn_dist(query, ref):
    # min Euclidean distance from each query window to any ref window
    out = []
    for q in query:
        out.append(float(np.sqrt(((ref - q) ** 2).sum(axis=1)).min()))
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", type=Path, required=True)
    p.add_argument("--heldout-parquet", type=Path, required=True)
    p.add_argument("--synthetic-jsonl", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--heldout-length", type=int, default=10000)
    p.add_argument("--label", default="")
    a = p.parse_args()
    train = scaled_windows(a.train_parquet, a.channel, a.train_length)
    held = scaled_windows(a.heldout_parquet, a.channel, a.heldout_length)
    syn = load_jsonl(a.synthetic_jsonl, a.channel)
    d_syn = nn_dist(syn, train)
    d_held = nn_dist(held, train)
    print("  %-8s synth-NN-to-train 中位=%.2f  | real-heldout-NN-to-train 中位=%.2f  | 比值=%.2f (>~1 不像记忆化)"
          % (a.label, np.median(d_syn), np.median(d_held), np.median(d_syn) / max(np.median(d_held), 1e-9)))


if __name__ == "__main__":
    main()
