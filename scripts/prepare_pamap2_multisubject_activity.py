#!/usr/bin/env python3
"""Prepare multi-subject activity-specific PAMAP2 parquet for SDForger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_pamap2_activity_subject import PAMAP2_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--activity-id", type=int, required=True)
    parser.add_argument("--columns", nargs="+", default=["hand_acc16_x"])
    parser.add_argument("--rows-per-subject", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def load_subject(path: Path, activity_id: int, columns: list[str], rows_per_subject: int) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        engine="python",
    )
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing requested columns in {path}: {missing}")

    filtered = raw.loc[raw["activity_id"] == activity_id, columns].copy()
    filtered = filtered.apply(pd.to_numeric, errors="raise")
    filtered = filtered.interpolate(method="linear", limit_direction="both")
    if filtered.isnull().any().any():
        bad = filtered.columns[filtered.isnull().any()].tolist()
        raise ValueError(f"NaNs remain after interpolation in {path}: {bad}")
    if len(filtered) < rows_per_subject:
        raise ValueError(f"{path.name} activity {activity_id} has {len(filtered)} rows; need {rows_per_subject}.")
    return filtered.iloc[:rows_per_subject].copy()


def main() -> None:
    args = parse_args()
    chunks = []
    per_subject_rows = {}
    for subject in args.subjects:
        subject_name = str(subject)
        if not subject_name.startswith("subject"):
            subject_name = f"subject{subject_name}"
        path = args.protocol_dir / f"{subject_name}.dat"
        chunk = load_subject(path, args.activity_id, args.columns, args.rows_per_subject)
        chunks.append(chunk)
        per_subject_rows[subject_name] = int(chunk.shape[0])

    combined = pd.concat(chunks, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.astype("float32").to_parquet(args.output, index=False)

    metadata = {
        "protocol_dir": str(args.protocol_dir),
        "subjects": list(per_subject_rows.keys()),
        "activity_id": args.activity_id,
        "columns": args.columns,
        "rows_per_subject": args.rows_per_subject,
        "per_subject_rows": per_subject_rows,
        "total_rows": int(combined.shape[0]),
        "output": str(args.output),
        "value_space": "raw_pamap2_sensor_units",
        "note": "Each subject/activity contributes the first rows_per_subject rows after filtering and interpolation.",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
