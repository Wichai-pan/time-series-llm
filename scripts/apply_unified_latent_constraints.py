#!/usr/bin/env python3
"""Apply latent constraints to unified label-conditioned SDForger embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import (
    fica_embed_data,
    fica_transform_to_original_feature_space,
    preprocess_train_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-parquet", type=Path, required=True)
    parser.add_argument("--running-parquet", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--variance-explained", type=float, default=0.7)
    parser.add_argument("--embedding-dim", default="auto")
    return parser.parse_args()


def load_activity_windows(args: argparse.Namespace, parquet: Path) -> np.ndarray:
    df = pd.read_parquet(parquet)
    scaled, _original, _scalers = preprocess_train_data(
        df,
        train_channels=[args.channel],
        train_length=args.train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=args.window_length,
        min_windows_number=args.min_windows_number,
        train_splitting=args.train_splitting,
    )
    return np.asarray(scaled[0], dtype=np.float64)


def write_jsonl(path: Path, channel: str, label: str, windows: np.ndarray, variant: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for window in windows:
            record = {
                "task_name": f"time_series/pamap2_subject101_unified_label_conditioned_{variant}",
                "is_seed": False,
                "task_description": f"Unified label-conditioned output with latent constraint {variant}.",
                "requested_label": label,
                "constraint_variant": variant,
                "generated_time_series": {channel: [float(v) for v in window]},
            }
            handle.write(json.dumps(record) + "\n")


def constraint_bounds(train: pd.DataFrame, variant: str, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    if variant == "clip_minmax":
        return train[cols].min(), train[cols].max()
    if variant == "clip_p05_p95":
        return train[cols].quantile(0.05), train[cols].quantile(0.95)
    if variant == "reject_iqr3":
        q1 = train[cols].quantile(0.25)
        q3 = train[cols].quantile(0.75)
        iqr = q3 - q1
        return q1 - 3.0 * iqr, q3 + 3.0 * iqr
    raise ValueError(f"Unknown variant: {variant}")


def apply_variant(gen: pd.DataFrame, train: pd.DataFrame, variant: str, cols: list[str]) -> pd.DataFrame:
    out = gen.copy()
    lower, upper = constraint_bounds(train, variant, cols)
    if variant.startswith("clip_"):
        out[cols] = out[cols].clip(lower=lower, upper=upper, axis=1)
        return out
    mask = pd.Series(True, index=out.index)
    for col in cols:
        mask &= out[col].between(lower[col], upper[col])
    return out.loc[mask].copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walking = load_activity_windows(args, args.walking_parquet)
    running = load_activity_windows(args, args.running_parquet)
    combined = np.concatenate([walking, running], axis=0)
    combined_preprocessed = combined[None, :, :]

    train_embeddings, embedding_dims, data_embedded, fica_mixing, fica_mean = fica_embed_data(
        combined_preprocessed,
        args.embedding_dim,
        args.variance_explained,
    )
    train_embeddings["data"] = ["walking"] * walking.shape[0] + ["running"] * running.shape[0]
    cols = [c for c in train_embeddings.columns if c != "data"]

    summary = []
    for label in ["walking", "running"]:
        generated = pd.read_csv(args.generated_dir / f"{label}_generated_embeddings.csv")
        generated[cols] = generated[cols].apply(pd.to_numeric, errors="coerce")
        generated = generated.dropna(subset=cols).copy()
        for variant in ["clip_minmax", "clip_p05_p95", "reject_iqr3"]:
            constrained = apply_variant(generated, train_embeddings, variant, cols)
            variant_dir = args.output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            constrained.to_csv(variant_dir / f"{label}_generated_embeddings.csv", index=False)
            if constrained.empty:
                summary.append(
                    {
                        "variant": variant,
                        "label": label,
                        "input_rows": int(generated.shape[0]),
                        "output_rows": 0,
                        "latent_abs_max": None,
                    }
                )
                continue
            reconstructed = fica_transform_to_original_feature_space(
                constrained[cols].to_numpy(dtype=np.float64),
                combined_preprocessed,
                data_embedded,
                fica_mixing,
                fica_mean,
            )[0]
            write_jsonl(
                variant_dir / f"{label}_final_data.jsonl",
                args.channel,
                label,
                reconstructed,
                variant,
            )
            summary.append(
                {
                    "variant": variant,
                    "label": label,
                    "input_rows": int(generated.shape[0]),
                    "output_rows": int(constrained.shape[0]),
                    "latent_abs_max": float(np.abs(constrained[cols].to_numpy()).max()),
                    "decoded_abs_max": float(np.abs(reconstructed).max()),
                    "decoded_std_mean": float(np.std(reconstructed, axis=1).mean()),
                }
            )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output_dir / "latent_constraint_summary.csv", index=False)
    (args.output_dir / "latent_constraint_summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "channel": args.channel,
                    "embedding_dims": embedding_dims,
                    "train_windows": int(combined.shape[0]),
                    "walking_train_windows": int(walking.shape[0]),
                    "running_train_windows": int(running.shape[0]),
                    "value_space": "standardized_sdforger_window_space",
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
