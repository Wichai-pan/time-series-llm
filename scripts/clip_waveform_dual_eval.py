#!/usr/bin/env python3
"""Per-activity waveform clip (clip values to train band) + TSG on train AND held-out refs.

Clips each synthetic window's values to the per-activity train value range ("把大值/小值修掉"),
then evaluates against both the training subjects' reference and the held-out subjects' reference.
For a generator that already stays in range (e.g. Qwen), the clip is mild — that is the point.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data
from evaluate_sdforger_paper_metrics import (
    calculate_mdd, calculate_acd, calculate_sd, calculate_kd, calculate_ed, calculate_dtw)


def win(parquet, ch, tl):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=300,
                                           min_windows_number=30, train_splitting="minimize-overlap")
    return np.asarray(scaled[0], dtype=np.float64)


def load_jsonl(path, ch):
    return np.array([json.loads(l)["generated_time_series"][ch] for l in open(path) if l.strip()], dtype=np.float64)


def met(real, synth):
    r3 = real[:, :, None]; s3 = synth[:, :, None]; n = min(len(real), len(synth))
    return dict(MDD=calculate_mdd(r3, s3), ACD=calculate_acd(r3, s3), SD=calculate_sd(r3, s3),
               KD=calculate_kd(r3, s3), ED=calculate_ed(r3[:n], s3[:n]), DTW=calculate_dtw(r3[:n], s3[:n]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--synthetic-jsonl", type=Path, required=True)
    p.add_argument("--train-parquet", type=Path, required=True)
    p.add_argument("--heldout-parquet", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--heldout-length", type=int, default=10000)
    p.add_argument("--band", choices=["minmax", "p01_p99"], default="minmax")
    a = p.parse_args()
    train = win(a.train_parquet, a.channel, a.train_length)
    held = win(a.heldout_parquet, a.channel, a.heldout_length)
    lo, hi = (train.min(), train.max()) if a.band == "minmax" else (np.percentile(train, 1), np.percentile(train, 99))
    syn = load_jsonl(a.synthetic_jsonl, a.channel)
    syn_c = np.clip(syn, lo, hi)
    frac = float(np.mean((syn < lo) | (syn > hi)))
    print("  band[%.2f,%.2f] 越界点占比=%.3f" % (lo, hi, frac))
    for refname, ref in [("train ", train), ("heldout", held)]:
        for cname, S in [("none", syn), ("clip", syn_c)]:
            m = met(ref, S)
            print("  %-8s ref=%s clip=%-4s MDD=%.3f ACD=%.3f SD=%.3f KD=%.3f ED=%.2f DTW=%.1f"
                  % (a.label, refname, cname, m["MDD"], m["ACD"], m["SD"], m["KD"], m["ED"], m["DTW"]))


if __name__ == "__main__":
    main()
