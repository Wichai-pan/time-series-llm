#!/usr/bin/env python3
"""Within-subject real-vs-real TSG floor.

The combined-best synthetic is generated from the SAME subjects it is scored against
(within-subject). The correct irreducible floor is therefore two random halves of the
SAME subjects' real windows -- NOT held-out subjects (which add a subject-shift term).
This is the honest yardstick for "how close to real is good" for our synthetic.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data
from evaluate_sdforger_paper_metrics import calculate_mdd, calculate_acd, calculate_dtw


def windows(parquet, ch, tl, wl=300, mwn=30, ts="minimize-overlap"):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--label", default="")
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--n-splits", type=int, default=5)
    a = p.parse_args()
    w = windows(a.parquet, a.channel, a.train_length)
    n = len(w)
    mdds, acds, dtws = [], [], []
    for s in range(a.n_splits):
        rng = np.random.RandomState(s)
        idx = rng.permutation(n)
        h = n // 2
        A = w[idx[:h]][:, :, None]
        B = w[idx[h:2 * h]][:, :, None]
        mdds.append(calculate_mdd(A, B)); acds.append(calculate_acd(A, B))
        dtws.append(calculate_dtw(A, B))
    f = lambda x: (float(np.mean(x)), float(np.std(x)))
    mm, ms = f(mdds); am, as_ = f(acds); dm, ds = f(dtws)
    print("  %-8s (n=%d, %d splits)  MDD=%.3f±%.3f  ACD=%.3f±%.3f  DTW=%.1f±%.1f"
          % (a.label, n, a.n_splits, mm, ms, am, as_, dm, ds))


if __name__ == "__main__":
    main()
