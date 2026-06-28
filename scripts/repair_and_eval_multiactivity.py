#!/usr/bin/env python3
"""N-activity per-activity repair + canonical TSG eval (for the 3-activity push).

Loads N activities (joint FICA, same order as generation), applies per-activity
clip_p05_p95 and shrink_minmax to each activity's generated embeddings, decodes, and
evaluates each (variant, activity) against its train reference with the canonical
evaluator (scaled space). Produces one TSG table.

Usage:
  python scripts/repair_and_eval_multiactivity.py \
    --activity walking=train.parquet --activity running=... --activity cycling=... \
    --generated-dir <gen_dir> --repair-dir <out> --reports-dir <reports> --train-length 15000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import (
    fica_embed_data, fica_transform_to_original_feature_space, preprocess_train_data)

EVAL = REPO_ROOT / "scripts" / "evaluate_sdforger_paper_metrics.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activity", action="append", required=True, metavar="LABEL=PARQUET")
    p.add_argument("--generated-dir", type=Path, required=True)
    p.add_argument("--repair-dir", type=Path, required=True)
    p.add_argument("--reports-dir", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    return p.parse_args()


def load_windows(parquet, ch, tl, wl, mwn, ts):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def latent_shrink(vals, lo, hi):
    ratio = np.full_like(vals, np.inf)
    pos, neg = vals > 0, vals < 0
    hb, lb = np.broadcast_to(hi, vals.shape), np.broadcast_to(lo, vals.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[pos] = hb[pos] / vals[pos]
        ratio[neg] = lb[neg] / vals[neg]
    s = np.clip(np.nanmin(ratio, axis=1), 0.0, 1.0)
    return vals * s[:, None]


def main():
    a = parse_args()
    a.repair_dir.mkdir(parents=True, exist_ok=True)
    a.reports_dir.mkdir(parents=True, exist_ok=True)
    acts = []
    for spec in a.activity:
        lab, _, parq = spec.partition("=")
        acts.append((lab.strip(), Path(parq.strip())))

    windows, labels = [], []
    for lab, parq in acts:
        w = load_windows(parq, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
        windows.append(w); labels += [lab] * w.shape[0]
    combined = np.concatenate(windows, axis=0); combined_pre = combined[None, :, :]
    emb, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, a.embedding_dim, a.variance_explained)
    emb["data"] = labels
    cols = [c for c in emb.columns if c != "data"]

    # per-activity repair
    for lab, _ in acts:
        sub = emb[emb["data"] == lab]
        p05, p95 = sub[cols].quantile(0.05), sub[cols].quantile(0.95)
        lo, hi = sub[cols].min(), sub[cols].max()
        gen = pd.read_csv(a.generated_dir / f"{lab}_generated_embeddings.csv")
        gen[cols] = gen[cols].apply(pd.to_numeric, errors="coerce")
        gen = gen.dropna(subset=cols)
        for vname, lat in (("clip_p05_p95", gen[cols].clip(lower=p05, upper=p95, axis=1).to_numpy(float)),
                           ("shrink_minmax", latent_shrink(gen[cols].to_numpy(float), lo.to_numpy(float), hi.to_numpy(float)))):
            dec = fica_transform_to_original_feature_space(lat, combined_pre, data_embedded, mixing, mean)[0]
            vd = a.repair_dir / vname; vd.mkdir(parents=True, exist_ok=True)
            with (vd / f"{lab}_final_data.jsonl").open("w") as h:
                for w in dec:
                    h.write(json.dumps({"task_name": "ts/multiactivity", "is_seed": False, "requested_label": lab,
                                        "generated_time_series": {a.channel: [float(v) for v in w]}}) + "\n")

    # eval
    rows = []
    for lab, parq in acts:
        for vname in ("clip_p05_p95", "shrink_minmax"):
            stem = f"{vname}_{lab}"
            oj = a.reports_dir / f"{stem}.json"
            cmd = [sys.executable, str(EVAL), "--real-parquet", str(parq),
                   "--synthetic-jsonl", str(a.repair_dir / vname / f"{lab}_final_data.jsonl"),
                   "--channel", a.channel, "--output-json", str(oj), "--output-md", str(a.reports_dir / f"{stem}.md"),
                   "--output-plot", str(a.reports_dir / f"{stem}.png"), "--train-length", str(a.train_length),
                   "--synthetic-space", "scaled", "--subset-mode", "first"]
            try:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                r = json.loads(oj.read_text())
                rows.append({"setting": vname, "activity": lab, "n": r["synthetic_window_count_evaluated"],
                             "MDD": r["mdd"], "ACD": r["acd"], "SD": r["sd"], "KD": r["kd"], "ED": r["ed"], "DTW": r["dtw"]})
            except Exception as e:
                rows.append({"setting": vname, "activity": lab, "n": 0, "error": str(e)[:100]})
    df = pd.DataFrame(rows)
    df.to_csv(a.reports_dir / "multiactivity_tsgbench_summary.csv", index=False)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
