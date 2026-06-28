#!/usr/bin/env python3
"""Evaluate a single normalization run in canonical scaled space (reuses the 06-07 inverse-transform)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from evaluate_canonical_scaled_tsgbench import (
    load_baseline_windows, load_jsonl, canonicalize_norm_synthetic)
from evaluate_sdforger_paper_metrics import (
    calculate_mdd, calculate_acd, calculate_sd, calculate_kd, calculate_ed, calculate_dtw)


def metrics_uneven(real, synth):
    """MDD/ACD/SD/KD support different N; ED/DTW pair on min count."""
    r3 = real[:, :, None].astype(np.float64); s3 = synth[:, :, None].astype(np.float64)
    n = min(len(real), len(synth))
    return {"MDD": calculate_mdd(r3, s3), "ACD": calculate_acd(r3, s3), "SD": calculate_sd(r3, s3),
            "KD": calculate_kd(r3, s3), "ED": calculate_ed(r3[:n], s3[:n]), "DTW": calculate_dtw(r3[:n], s3[:n])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode", default="activity_series_zscore")
    p.add_argument("--walking-parquet", type=Path, required=True)
    p.add_argument("--running-parquet", type=Path, required=True)
    p.add_argument("--train-length", type=int, default=15000)
    a = p.parse_args()
    meta = json.loads((a.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cache = {}
    for act, pq in [("walking", a.walking_parquet), ("running", a.running_parquet)]:
        scaled, raw, mean, std = load_baseline_windows(pq, a.train_length, 300, 30, "minimize-overlap")
        cache[act] = {"scaled": scaled, "raw": raw, "mean": mean, "std": std}
    js = StandardScaler().fit(np.concatenate([cache["walking"]["raw"], cache["running"]["raw"]], axis=0))
    for act in ["walking", "running"]:
        mw = load_jsonl(a.run_dir / f"{act}_final_data.jsonl")
        synth = canonicalize_norm_synthetic(a.mode, act, mw, meta, cache, js)
        m = metrics_uneven(cache[act]["scaled"], synth)
        print("  %-8s n=%d  MDD=%.3f ACD=%.3f SD=%.3f KD=%.3f ED=%.2f DTW=%.1f"
              % (act, synth.shape[0], m["MDD"], m["ACD"], m["SD"], m["KD"], m["ED"], m["DTW"]))


if __name__ == "__main__":
    main()
