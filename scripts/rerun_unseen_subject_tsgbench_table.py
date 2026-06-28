#!/usr/bin/env python3
"""Evaluate multi-subject SDForger outputs against held-out PAMAP2 subjects."""

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
    walking = DATA / "pamap2_subject106_108_walking_hand_acc16_x.parquet"
    running = DATA / "pamap2_subject106_108_running_hand_acc16_x.parquet"
    generated = OUT_ROOT / "pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x"
    constrained = OUT_ROOT / "pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x_constraints" / "clip_p05_p95"
    return [
        {
            "setting": "unseen_raw_unified",
            "activity": "walking",
            "real": walking,
            "synthetic": generated / "walking_final_data.jsonl",
        },
        {
            "setting": "unseen_raw_unified",
            "activity": "running",
            "real": running,
            "synthetic": generated / "running_final_data.jsonl",
        },
        {
            "setting": "unseen_clip_p05_p95",
            "activity": "walking",
            "real": walking,
            "synthetic": constrained / "walking_final_data.jsonl",
        },
        {
            "setting": "unseen_clip_p05_p95",
            "activity": "running",
            "real": running,
            "synthetic": constrained / "running_final_data.jsonl",
        },
    ]


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "reports" / "unseen_subject_tsgbench_20260607"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs():
        stem = f"{spec['setting']}_{spec['activity']}"
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
            "--train-length",
            "10000",
            "--synthetic-space",
            "scaled",
            "--subset-mode",
            "first",
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
                "real_windows": result["real_window_count"],
                "paired_windows": result["paired_sample_count"],
                "train_subjects_for_generator": "101/102/105",
                "heldout_reference_subjects": "106/108",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "unseen_subject_tsgbench_summary.csv", index=False)
    (output_dir / "unseen_subject_tsgbench_summary.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
