#!/usr/bin/env python3
"""Combined-best config TSG evaluation.

Multi-subject (101/102/105) unified label-conditioned generation + per-activity smooth
repair, evaluated against BOTH the train-subject reference (101/102/105) and the
held-out reference (106/108). Same canonical evaluator / scaled space as T1/canonical.

Usage: python scripts/rerun_combined_best_tsgbench_table.py <output_dir> <repair_dir>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "scripts" / "evaluate_sdforger_paper_metrics.py"
CH = "hand_acc16_x"
DATA = REPO_ROOT / "data" / "public" / "time_series"
VARIANTS = ["clip_p05_p95", "shrink_p05_p95", "shrink_minmax", "waveform_interp"]
REFS = [
    ("train_101_102_105", "pamap2_subject101_102_105_{a}_hand_acc16_x.parquet", 15000),
    ("heldout_106_108", "pamap2_subject106_108_{a}_hand_acc16_x.parquet", 10000),
]


def main() -> None:
    out_dir = Path(sys.argv[1])
    repair_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ref_name, pat, tl in REFS:
        for v in VARIANTS:
            for a in ["walking", "running"]:
                real = DATA / pat.format(a=a)
                syn = repair_dir / v / f"{a}_final_data.jsonl"
                stem = f"{ref_name}_{v}_{a}"
                oj = out_dir / f"{stem}.json"
                cmd = [sys.executable, str(EVAL), "--real-parquet", str(real),
                       "--synthetic-jsonl", str(syn), "--channel", CH,
                       "--output-json", str(oj), "--output-md", str(out_dir / f"{stem}.md"),
                       "--output-plot", str(out_dir / f"{stem}.png"),
                       "--train-length", str(tl), "--synthetic-space", "scaled", "--subset-mode", "first"]
                try:
                    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
                    r = json.loads(oj.read_text(encoding="utf-8"))
                    rows.append({"ref": ref_name, "setting": v, "activity": a,
                                 "n": r["synthetic_window_count_evaluated"], "MDD": r["mdd"],
                                 "ACD": r["acd"], "SD": r["sd"], "KD": r["kd"], "ED": r["ed"], "DTW": r["dtw"]})
                except Exception as e:  # noqa: BLE001
                    rows.append({"ref": ref_name, "setting": v, "activity": a, "n": 0,
                                 "MDD": None, "ACD": None, "SD": None, "KD": None, "ED": None, "DTW": None,
                                 "error": str(e)[:120]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "combined_best_tsgbench_summary.csv", index=False)
    (out_dir / "combined_best_tsgbench_summary.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
