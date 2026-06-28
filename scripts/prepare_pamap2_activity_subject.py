#!/usr/bin/env python3
"""Prepare an activity-specific PAMAP2 subject parquet for SDForger."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAMAP2_COLUMNS = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temp",
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_acc6_x",
    "hand_acc6_y",
    "hand_acc6_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_0",
    "hand_orient_1",
    "hand_orient_2",
    "hand_orient_3",
    "chest_temp",
    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_acc6_x",
    "chest_acc6_y",
    "chest_acc6_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_0",
    "chest_orient_1",
    "chest_orient_2",
    "chest_orient_3",
    "ankle_temp",
    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
    "ankle_acc6_x",
    "ankle_acc6_y",
    "ankle_acc6_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_0",
    "ankle_orient_1",
    "ankle_orient_2",
    "ankle_orient_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activity-id", type=int, required=True)
    parser.add_argument("--columns", nargs="+", default=["hand_acc16_x"])
    parser.add_argument("--min-rows", type=int, default=5000)
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(
        args.input,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        engine="python",
    )
    missing = [column for column in args.columns if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing requested columns: {missing}")

    filtered = raw.loc[raw["activity_id"] == args.activity_id, args.columns].copy()
    filtered = filtered.apply(pd.to_numeric, errors="raise")
    filtered = filtered.interpolate(method="linear", limit_direction="both")
    if filtered.isnull().any().any():
        bad = filtered.columns[filtered.isnull().any()].tolist()
        raise ValueError(f"NaNs remain after interpolation in columns: {bad}")
    if len(filtered) < args.min_rows:
        raise ValueError(
            f"Only {len(filtered)} rows for activity {args.activity_id}; need {args.min_rows}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.astype("float32").to_parquet(args.output, index=False)

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "activity_id": args.activity_id,
        "columns": args.columns,
        "rows": int(len(filtered)),
        "min_rows": args.min_rows,
        "value_space": "raw_pamap2_sensor_units",
        "note": "activity-specific subset; activity_id is not stored in the SDForger parquet because the baseline is univariate sensor-only.",
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(__import__("json").dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
