#!/usr/bin/env python3
"""Rerun the original scaled-space TSGBench table for PAMAP2 settings."""

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


def specs() -> list[dict[str, str | Path]]:
    walking = DATA / "pamap2_subject101_walking_hand_acc16_x.parquet"
    running = DATA / "pamap2_subject101_running_hand_acc16_x.parquet"
    return [
        {
            "setting": "baseline",
            "activity": "walking",
            "real": walking,
            "synthetic": OUT_ROOT / "pamap2_subject101_walking_hand_acc16_x_univariate_baseline" / "final_data.jsonl",
        },
        {
            "setting": "baseline",
            "activity": "running",
            "real": running,
            "synthetic": OUT_ROOT / "pamap2_subject101_running_hand_acc16_x_univariate_baseline" / "final_data.jsonl",
        },
        {
            "setting": "label-v1",
            "activity": "walking",
            "real": walking,
            "synthetic": OUT_ROOT / "pamap2_subject101_walking_hand_acc16_x_label_conditioned_univariate" / "final_data.jsonl",
        },
        {
            "setting": "label-v1",
            "activity": "running",
            "real": running,
            "synthetic": OUT_ROOT / "pamap2_subject101_running_hand_acc16_x_label_conditioned_univariate" / "final_data.jsonl",
        },
        {
            "setting": "raw-unified",
            "activity": "walking",
            "real": walking,
            "synthetic": OUT_ROOT / "pamap2_subject101_unified_label_conditioned_hand_acc16_x" / "walking_final_data.jsonl",
        },
        {
            "setting": "raw-unified",
            "activity": "running",
            "real": running,
            "synthetic": OUT_ROOT / "pamap2_subject101_unified_label_conditioned_hand_acc16_x" / "running_final_data.jsonl",
        },
        {
            "setting": "clip_p05_p95",
            "activity": "walking",
            "real": walking,
            "synthetic": OUT_ROOT / "pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints" / "clip_p05_p95" / "walking_final_data.jsonl",
        },
        {
            "setting": "clip_p05_p95",
            "activity": "running",
            "real": running,
            "synthetic": OUT_ROOT / "pamap2_subject101_unified_label_conditioned_hand_acc16_x_constraints" / "clip_p05_p95" / "running_final_data.jsonl",
        },
    ]


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "reports" / "scaled_tsgbench_rerun_20260607"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs():
        stem = f"{spec['setting']}_{spec['activity']}".replace("-", "_")
        output_json = output_dir / f"{stem}.json"
        output_md = output_dir / f"{stem}.md"
        output_plot = output_dir / f"{stem}.png"
        cmd = [
            sys.executable,
            str(EVAL),
            "--real-parquet",
            str(spec["real"]),
            "--synthetic-jsonl",
            str(spec["synthetic"]),
            "--channel",
            CHANNEL,
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-plot",
            str(output_plot),
            "--synthetic-space",
            "scaled",
            "--subset-mode",
            "first",
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        result = json.loads(output_json.read_text())
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
    df.to_csv(output_dir / "scaled_tsgbench_rerun_summary.csv", index=False)
    (output_dir / "scaled_tsgbench_rerun_summary.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
