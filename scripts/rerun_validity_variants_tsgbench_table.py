#!/usr/bin/env python3
"""Evaluate generation-time latent validity variants on train and held-out references."""

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


def specs() -> list[dict[str, str | Path | int]]:
    train_walking = DATA / "pamap2_subject101_102_105_walking_hand_acc16_x.parquet"
    train_running = DATA / "pamap2_subject101_102_105_running_hand_acc16_x.parquet"
    unseen_walking = DATA / "pamap2_subject106_108_walking_hand_acc16_x.parquet"
    unseen_running = DATA / "pamap2_subject106_108_running_hand_acc16_x.parquet"
    root = OUT_ROOT / "pamap2_subject101_102_105_unified_label_conditioned_hand_acc16_x_validity"

    rows: list[dict[str, str | Path | int]] = []
    for reference_name, walking, running, train_length in [
        ("train_reference_101_102_105", train_walking, train_running, 15000),
        ("heldout_reference_106_108", unseen_walking, unseen_running, 10000),
    ]:
        for variant in ["strict_reject_p05_p95", "soft_repair_p05_p95_minmax"]:
            variant_dir = root / variant
            rows.extend(
                [
                    {
                        "reference": reference_name,
                        "setting": variant,
                        "activity": "walking",
                        "real": walking,
                        "synthetic": variant_dir / "walking_final_data.jsonl",
                        "train_length": train_length,
                    },
                    {
                        "reference": reference_name,
                        "setting": variant,
                        "activity": "running",
                        "real": running,
                        "synthetic": variant_dir / "running_final_data.jsonl",
                        "train_length": train_length,
                    },
                ]
            )
    return rows


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "reports" / "validity_variants_tsgbench_20260608"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs():
        stem = f"{spec['reference']}_{spec['setting']}_{spec['activity']}"
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
            str(spec["train_length"]),
            "--synthetic-space",
            "scaled",
            "--subset-mode",
            "first",
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        result = json.loads(output_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "reference": spec["reference"],
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
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "validity_variants_tsgbench_summary.csv", index=False)
    (output_dir / "validity_variants_tsgbench_summary.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
