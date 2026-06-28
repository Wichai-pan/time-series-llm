#!/usr/bin/env python3
"""Controllability baseline: per-bin conditional Gaussian over FICA coefficients.

The LLM's only remaining claimed advantage is CONTROLLABILITY (steer generation to a
target window-max; Qwen real-value Spearman ~0.36). This asks the adversarial question:
does a trivial conditional model achieve the same or better adherence? Bin train windows
by their max into n_levels, fit a Gaussian on the FICA coefficients WITHIN each bin, then
to "request level L" sample from bin L's Gaussian and decode. Measure Spearman(requested
level, realized max) on the SAME 4-level x 24 protocol as run_stat_prompt_hf. If this
beats 0.36, the LLM has no demonstrated advantage even on controllability.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scipy.stats import spearmanr
from fms_dgt.public.databuilders.time_series.utils import (
    fica_embed_data, fica_transform_to_original_feature_space, preprocess_train_data)


def load_windows(parquet, ch, tl, wl=300, mwn=30):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting="minimize-overlap")
    return np.asarray(scaled[0], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walking-parquet", type=Path, required=True)
    p.add_argument("--running-parquet", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    p.add_argument("--n-levels", type=int, default=4)
    p.add_argument("--per-level", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = np.random.RandomState(a.seed)

    walking = load_windows(a.walking_parquet, a.channel, a.train_length)
    running = load_windows(a.running_parquet, a.channel, a.train_length)
    combined = np.concatenate([walking, running], axis=0); combined_pre = combined[None, :, :]
    embedded, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, a.embedding_dim, a.variance_explained)
    cols = list(embedded.columns)
    coeffs = embedded[cols].to_numpy(np.float64)
    labels = np.array(["walking"] * len(walking) + ["running"] * len(running))

    rows = []
    for label in ["walking", "running"]:
        sub = coeffs[labels == label]
        win = walking if label == "walking" else running
        mx = win.max(axis=1)
        # n_levels quantile bins (matches run_stat_prompt_hf level definition)
        edges = np.quantile(mx, np.linspace(0, 1, a.n_levels + 1)[1:-1])
        binid = np.digitize(mx, edges)
        for L in range(a.n_levels):
            members = sub[binid == L]
            if len(members) < 2:
                continue
            mu = members.mean(axis=0)
            cov = np.cov(members, rowvar=False) + 1e-6 * np.eye(sub.shape[1])
            samples = rng.multivariate_normal(mu, cov, size=a.per_level)
            dec = fica_transform_to_original_feature_space(samples, combined_pre, data_embedded, mixing, mean)[0]
            for w in dec:
                rows.append({"label": label, "requested_level": L, "realized_max": float(w.max())})

    out = pd.DataFrame(rows)
    for label in ["walking", "running", "ALL"]:
        d = out if label == "ALL" else out[out.label == label]
        if len(d) > 5 and d.requested_level.nunique() > 1:
            rho = spearmanr(d.requested_level, d.realized_max).correlation
            med = d.groupby("requested_level").realized_max.median().round(3).to_dict()
            print("  cond-Gaussian %-8s n=%d  Spearman=%.3f  level→median_max=%s" % (label, len(d), rho, med))
    print("  --- 对照: LLM(Qwen 实值) pooled Spearman 0.36 (3-seed), bins 0.10 ---")


if __name__ == "__main__":
    main()
