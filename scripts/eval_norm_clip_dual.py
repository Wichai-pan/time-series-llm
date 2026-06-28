#!/usr/bin/env python3
"""Norm run -> canonical scaled -> per-activity waveform clip -> TSG on train AND held-out.

The multi-subject normalization run exploded (no repair). Hypothesis: clipping the out-of-band
values recovers it. Canonicalize the norm synthetic to scaled space, clip each activity's values
to the train per-activity band, then evaluate against both train (101/102/105) and held-out
(106/108) references. Reports none vs clip x train vs held-out, full 6 metrics.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data
from evaluate_canonical_scaled_tsgbench import load_baseline_windows, load_jsonl, canonicalize_norm_synthetic
from evaluate_sdforger_paper_metrics import (
    calculate_mdd, calculate_acd, calculate_sd, calculate_kd, calculate_ed, calculate_dtw)
import pandas as pd


def scaled_windows(parquet, ch, tl):
    df = pd.read_parquet(parquet)
    s, _o, _ss = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                       augmentation_strategy="univariate", min_windows_length=300,
                                       min_windows_number=30, train_splitting="minimize-overlap")
    return np.asarray(s[0], dtype=np.float64)


def met(real, synth):
    r3 = real[:, :, None]; s3 = synth[:, :, None]; n = min(len(real), len(synth))
    return dict(MDD=calculate_mdd(r3, s3), ACD=calculate_acd(r3, s3), SD=calculate_sd(r3, s3),
               KD=calculate_kd(r3, s3), ED=calculate_ed(r3[:n], s3[:n]), DTW=calculate_dtw(r3[:n], s3[:n]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode", default="activity_series_zscore")
    p.add_argument("--walking-train", type=Path, required=True)
    p.add_argument("--running-train", type=Path, required=True)
    p.add_argument("--walking-heldout", type=Path, required=True)
    p.add_argument("--running-heldout", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--heldout-length", type=int, default=10000)
    a = p.parse_args()
    meta = json.loads((a.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cache = {}
    for act, pq in [("walking", a.walking_train), ("running", a.running_train)]:
        scaled, raw, mean, std = load_baseline_windows(pq, a.train_length, 300, 30, "minimize-overlap")
        cache[act] = {"scaled": scaled, "raw": raw, "mean": mean, "std": std}
    js = StandardScaler().fit(np.concatenate([cache["walking"]["raw"], cache["running"]["raw"]], axis=0))
    heldout = {"walking": scaled_windows(a.walking_heldout, a.channel, a.heldout_length),
               "running": scaled_windows(a.running_heldout, a.channel, a.heldout_length)}
    for act in ["walking", "running"]:
        mw = load_jsonl(a.run_dir / f"{act}_final_data.jsonl")
        syn = canonicalize_norm_synthetic(a.mode, act, mw, meta, cache, js)
        tr = cache[act]["scaled"]
        lo, hi = tr.min(), tr.max()
        syn_c = np.clip(syn, lo, hi)
        frac = float(np.mean((syn < lo) | (syn > hi)))
        print("  -- %s  越界点占比=%.3f  band[%.2f,%.2f] --" % (act, frac, lo, hi))
        for refname, ref in [("train  ", tr), ("heldout", heldout[act])]:
            for cname, S in [("none", syn), ("clip", syn_c)]:
                m = met(ref, S)
                print("    %s ref=%s clip=%-4s MDD=%.3f ACD=%.3f SD=%.3f KD=%.3f ED=%.1f DTW=%.1f"
                      % (act, refname, cname, m["MDD"], m["ACD"], m["SD"], m["KD"], m["ED"], m["DTW"]))


if __name__ == "__main__":
    main()
