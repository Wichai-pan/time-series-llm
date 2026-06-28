#!/usr/bin/env python3
"""Apply generation-time-style latent validity rules to parsed SDForger embeddings."""

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
    parser.add_argument("--train-length", type=int, default=15000)
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
                "task_name": f"time_series/pamap2_generation_time_validity_{variant}",
                "is_seed": False,
                "task_description": f"Unified label-conditioned output with generation-time-style latent validity {variant}.",
                "requested_label": label,
                "constraint_variant": variant,
                "generated_time_series": {channel: [float(v) for v in window]},
            }
            handle.write(json.dumps(record) + "\n")


def row_mask_between(df: pd.DataFrame, lower: pd.Series, upper: pd.Series, cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= df[col].between(lower[col], upper[col])
    return mask


def decode_and_write(
    variant_dir: Path,
    label: str,
    variant: str,
    channel: str,
    embeddings: pd.DataFrame,
    cols: list[str],
    combined_preprocessed: np.ndarray,
    data_embedded: list[np.ndarray],
    fica_mixing: list[np.ndarray],
    fica_mean: list[np.ndarray],
) -> tuple[int, float | None, float | None, float | None]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(variant_dir / f"{label}_generated_embeddings.csv", index=False)
    if embeddings.empty:
        (variant_dir / f"{label}_final_data.jsonl").write_text("", encoding="utf-8")
        return 0, None, None, None
    reconstructed = fica_transform_to_original_feature_space(
        embeddings[cols].to_numpy(dtype=np.float64),
        combined_preprocessed,
        data_embedded,
        fica_mixing,
        fica_mean,
    )[0]
    write_jsonl(variant_dir / f"{label}_final_data.jsonl", channel, label, reconstructed, variant)
    return (
        int(embeddings.shape[0]),
        float(np.abs(embeddings[cols].to_numpy(dtype=np.float64)).max()),
        float(np.abs(reconstructed).max()),
        float(np.std(reconstructed, axis=1).mean()),
    )


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

    p05 = train_embeddings[cols].quantile(0.05)
    p95 = train_embeddings[cols].quantile(0.95)
    hard_min = train_embeddings[cols].min()
    hard_max = train_embeddings[cols].max()

    summary: list[dict[str, object]] = []
    for label in ["walking", "running"]:
        generated = pd.read_csv(args.generated_dir / f"{label}_generated_embeddings.csv")
        generated[cols] = generated[cols].apply(pd.to_numeric, errors="coerce")
        parsed = generated.dropna(subset=cols).copy()

        strict_mask = row_mask_between(parsed, p05, p95, cols)
        strict = parsed.loc[strict_mask].copy()
        output_rows, latent_abs_max, decoded_abs_max, decoded_std_mean = decode_and_write(
            args.output_dir / "strict_reject_p05_p95",
            label,
            "strict_reject_p05_p95",
            args.channel,
            strict,
            cols,
            combined_preprocessed,
            data_embedded,
            fica_mixing,
            fica_mean,
        )
        summary.append(
            {
                "variant": "strict_reject_p05_p95",
                "label": label,
                "input_rows": int(parsed.shape[0]),
                "clean_accept_rows": int(strict.shape[0]),
                "repaired_rows": 0,
                "rejected_rows": int(parsed.shape[0] - strict.shape[0]),
                "output_rows": output_rows,
                "latent_abs_max": latent_abs_max,
                "decoded_abs_max": decoded_abs_max,
                "decoded_std_mean": decoded_std_mean,
                "hard_rule": "all latent dims inside train p05-p95",
                "repair_rule": "none",
            }
        )

        hard_mask = row_mask_between(parsed, hard_min, hard_max, cols)
        soft = parsed.loc[hard_mask].copy()
        clean_mask_in_soft = row_mask_between(soft, p05, p95, cols)
        repaired_rows = int((~clean_mask_in_soft).sum())
        soft[cols] = soft[cols].clip(lower=p05, upper=p95, axis=1)
        output_rows, latent_abs_max, decoded_abs_max, decoded_std_mean = decode_and_write(
            args.output_dir / "soft_repair_p05_p95_minmax",
            label,
            "soft_repair_p05_p95_minmax",
            args.channel,
            soft,
            cols,
            combined_preprocessed,
            data_embedded,
            fica_mixing,
            fica_mean,
        )
        summary.append(
            {
                "variant": "soft_repair_p05_p95_minmax",
                "label": label,
                "input_rows": int(parsed.shape[0]),
                "clean_accept_rows": int(clean_mask_in_soft.sum()),
                "repaired_rows": repaired_rows,
                "rejected_rows": int(parsed.shape[0] - soft.shape[0]),
                "output_rows": output_rows,
                "latent_abs_max": latent_abs_max,
                "decoded_abs_max": decoded_abs_max,
                "decoded_std_mean": decoded_std_mean,
                "hard_rule": "reject if any latent dim outside train min-max",
                "repair_rule": "clip remaining p05-p95 violations to p05-p95",
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output_dir / "generation_time_validity_summary.csv", index=False)
    (args.output_dir / "generation_time_validity_summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "channel": args.channel,
                    "embedding_dims": embedding_dims,
                    "train_windows": int(combined.shape[0]),
                    "walking_train_windows": int(walking.shape[0]),
                    "running_train_windows": int(running.shape[0]),
                    "value_space": "standardized_sdforger_window_space",
                    "scope_note": "This applies validity rules to parser-accepted generated latent rows. Raw malformed text generations are already filtered upstream by SDForger parsing.",
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
