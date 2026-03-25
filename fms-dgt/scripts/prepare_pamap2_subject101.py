#!/usr/bin/env python3
"""Prepare a Forge-ready PAMAP2 parquet file for subject101 multivariate generation."""

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

SELECTED_COLUMNS = [
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PAMAP2 Protocol subject101 data into Forge-ready parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to PAMAP2 Protocol/subject101.dat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet path",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=60000,
        help="Minimum acceptable number of rows after filtering",
    )
    return parser.parse_args()


def load_pamap2_dataframe(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        engine="python",
    )
    if data.shape[1] != len(PAMAP2_COLUMNS):
        raise ValueError(
            f"Expected {len(PAMAP2_COLUMNS)} columns, found {data.shape[1]} in {path}"
        )
    return data


def build_output_frame(data: pd.DataFrame) -> pd.DataFrame:
    if "activity_id" not in data.columns:
        raise ValueError("Input dataframe is missing the activity_id column.")

    filtered = data.loc[data["activity_id"] != 0, SELECTED_COLUMNS].copy()
    filtered = filtered.apply(pd.to_numeric, errors="raise")
    filtered = filtered.interpolate(method="linear", limit_direction="both")

    if filtered.isnull().any().any():
        missing = filtered.columns[filtered.isnull().any()].tolist()
        raise ValueError(f"NaN values remain after interpolation in columns: {missing}")

    if list(filtered.columns) != SELECTED_COLUMNS:
        raise ValueError("Output columns do not match the expected Forge schema.")

    return filtered.astype("float32")


def main() -> None:
    args = parse_args()
    raw = load_pamap2_dataframe(args.input)
    prepared = build_output_frame(raw)

    if len(prepared) < args.min_rows:
        raise ValueError(
            f"Prepared dataframe has {len(prepared)} rows, expected at least {args.min_rows}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)

    print(f"Wrote {len(prepared)} rows to {args.output}")
    print("Columns:", ", ".join(prepared.columns))


if __name__ == "__main__":
    main()
