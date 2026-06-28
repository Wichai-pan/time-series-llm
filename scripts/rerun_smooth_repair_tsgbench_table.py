#!/usr/bin/env python3
"""Evaluate smooth-repair variants (clip vs shrink vs waveform-interp) on subject101.

Mirrors rerun_validity_variants_tsgbench_table.py so the new repair variants are
scored with the SAME canonical TSG evaluator (evaluate_sdforger_paper_metrics.py,
--synthetic-space scaled, --subset-mode first) and are directly comparable to the
combined-experiment-table main rows (subject101 unified, train_length 5000).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "scripts" / "evaluate_sdforger_paper_metrics.py"
CHANNEL = "hand_acc16_x"
DATA = REPO_ROOT / "data" / "public" / "time_series"
OUT_ROOT = REPO_ROOT / "output" / "time_series"

import os

DEFAULT_REPAIR_DIR = OUT_ROOT / "pamap2_subject101_unified_label_conditioned_hand_acc16_x_smooth_repair_20260615"
REPAIR_DIR = Path(os.environ.get("SMOOTH_REPAIR_DIR", str(DEFAULT_REPAIR_DIR)))
VARIANTS = ["clip_p05_p95", "shrink_p05_p95", "shrink_minmax", "waveform_interp"]
TRAIN_LENGTH = 5000


def specs() -> list[dict[str, object]]:
    walking = DATA / "pamap2_subject101_walking_hand_acc16_x.parquet"
    running = DATA / "pamap2_subject101_running_hand_acc16_x.parquet"
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for activity, real in (("walking", walking), ("running", running)):
            rows.append(
                {
                    "setting": variant,
                    "activity": activity,
                    "real": real,
                    "synthetic": REPAIR_DIR / variant / f"{activity}_final_data.jsonl",
                }
            )
    return rows


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "reports" / "smooth_repair_tsgbench_20260615"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs():
        stem = f"{spec['setting']}_{spec['activity']}"
        output_json = output_dir / f"{stem}.json"
        cmd = [
            sys.executable,
            str(EVAL),
            "--real-parquet", str(spec["real"]),
            "--synthetic-jsonl", str(spec["synthetic"]),
            "--channel", CHANNEL,
            "--output-json", str(output_json),
            "--output-md", str(output_dir / f"{stem}.md"),
            "--output-plot", str(output_dir / f"{stem}.png"),
            "--train-length", str(TRAIN_LENGTH),
            "--synthetic-space", "scaled",
            "--subset-mode", "first",
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        result = json.loads(output_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "setting": spec["setting"],
                "activity": spec["activity"],
                "n": result["synthetic_window_count_evaluated"],
                "MDD": result["mdd"],
                "ACD": result["acd"],
                "SD": result["sd"],
                "KD": result["kd"],
                "ED": result["ed"],
                "DTW": result["dtw"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "smooth_repair_tsgbench_summary.csv", index=False)
    (output_dir / "smooth_repair_tsgbench_summary.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
