#!/usr/bin/env python3
"""Generative baseline: fit a Gaussian to FICA coefficients, sample, decode.

This isolates the LLM's contribution. SDForger = LLM models p(coeffs) then FICA-decodes.
Replace the LLM with the simplest possible generative model -- a multivariate Gaussian
fit to the per-activity coefficients -- and run the SAME FICA decode. If SDForger ≈ this,
the LLM adds little over "fit Gaussian in coefficient space + linear decode"; if SDForger
is narrower (lower ACD/MDD than this), that is direct evidence of LLM under-dispersion,
because the Gaussian retains the full empirical coefficient covariance by construction.
Same FICA basis (fit on walking+running combined) as the combined-best run, so decode is
identical; only the prior over coefficients differs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from fms_dgt.public.databuilders.time_series.utils import (
    fica_embed_data, fica_transform_to_original_feature_space, preprocess_train_data)


def load_windows(parquet, ch, tl, wl=300, mwn=30, ts="minimize-overlap"):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walking-parquet", type=Path, required=True)
    p.add_argument("--running-parquet", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    p.add_argument("--n-samples", type=int, default=80, help="samples per label")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    walking = load_windows(a.walking_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, "minimize-overlap")
    running = load_windows(a.running_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, "minimize-overlap")
    combined = np.concatenate([walking, running], axis=0); combined_pre = combined[None, :, :]
    embedded, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, a.embedding_dim, a.variance_explained)
    cols = list(embedded.columns)
    coeffs = embedded[cols].to_numpy(np.float64)
    labels = np.array(["walking"] * len(walking) + ["running"] * len(running))

    for label in ["walking", "running"]:
        sub = coeffs[labels == label]
        mu = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False)
        cov = cov + 1e-6 * np.eye(cov.shape[0])  # regularize
        samples = rng.multivariate_normal(mu, cov, size=a.n_samples)
        dec = fica_transform_to_original_feature_space(samples, combined_pre, data_embedded, mixing, mean)[0]
        path = a.output_dir / f"{label}_final_data.jsonl"
        with path.open("w", encoding="utf-8") as h:
            for w in dec:
                h.write(json.dumps({"task_name": "ts/gaussian_coeff", "is_seed": False, "requested_label": label,
                                    "generated_time_series": {a.channel: [float(v) for v in w]}}) + "\n")
        print("  %-8s sampled=%d  coeff-dim=%d  decoded-absmax=%.2f -> %s"
              % (label, a.n_samples, sub.shape[1], float(np.abs(dec).max()), path.name))


if __name__ == "__main__":
    main()
