#!/usr/bin/env python3
"""Unified walking/running label-conditioned SDForger-style run.

This script is intentionally narrow for the PAMAP2 subject101 smoke:
it builds window-level labels after SDForger preprocessing, trains one
conditional text model, and generates separate requested-label outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.base.registry import get_block
from fms_dgt.core.blocks.llm import MODEL_ID_OR_PATH
import fms_dgt.core.blocks.llm.vllm.vllm_in_proc  # noqa: F401 - registers the vllm block
from fms_dgt.public.databuilders.time_series.trainer import SDForgerTuningBlock
from fms_dgt.public.databuilders.time_series.utils import (
    convert_texts_to_tabular_data,
    fica_embed_data,
    fica_transform_to_original_feature_space,
    get_feature_distribution,
    preprocess_train_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-parquet", type=Path, required=True)
    parser.add_argument("--running-parquet", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id-or-path", default="gpt2")
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--variance-explained", type=float, default=0.7)
    parser.add_argument("--embedding-dim", default="auto")
    parser.add_argument("--min-per-label", type=int, default=50)
    parser.add_argument("--max-per-label", type=int, default=100)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--input-tokens-precision", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_activity_windows(
    parquet: Path,
    channel: str,
    train_length: int,
    window_length: int,
    min_windows_number: int,
    train_splitting: str,
) -> np.ndarray:
    df = pd.read_parquet(parquet)
    scaled, _original, _scalers = preprocess_train_data(
        df,
        train_channels=[channel],
        train_length=train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=window_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    return np.asarray(scaled[0], dtype=np.float64)


def make_prompts(
    *,
    label: str,
    numerical_columns: list[str],
    tokenizer,
    batch_size: int,
    permute: bool = True,
) -> list[str]:
    prompts = []
    for _ in range(batch_size):
        cols = numerical_columns.copy()
        if permute:
            random.shuffle(cols)
        text_input = ", ".join(f"{col} is [blank]" for col in cols)
        prompts.append(f"Condition: data is {label} [sep] Input: {text_input} [sep] Target:")

    lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    max_len = max(lengths)
    pad = tokenizer.pad_token
    return [pad * (max_len - length) + prompt for prompt, length in zip(prompts, lengths)]


def filter_generated(
    generated_df: pd.DataFrame,
    numerical_columns: list[str],
    requested_label: str,
) -> pd.DataFrame:
    if generated_df.empty:
        return generated_df
    generated_df = generated_df.replace("NaN", np.nan).dropna(how="all")
    generated_df = generated_df.dropna(subset=numerical_columns + ["data"])
    generated_df = generated_df.loc[generated_df["data"].astype(str).str.strip() == requested_label]
    for col in numerical_columns:
        generated_df[col] = pd.to_numeric(generated_df[col], errors="coerce")
    generated_df = generated_df.dropna(subset=numerical_columns)
    return generated_df


def generate_for_label(
    *,
    llm,
    original_dataset: pd.DataFrame,
    numerical_columns: list[str],
    label: str,
    min_per_label: int,
    max_per_label: int,
    generation_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[pd.DataFrame] = []
    raw_count = 0
    parsed_count = 0
    accepted_count = 0
    attempts = 0

    while accepted_count < min_per_label and attempts < 12:
        prompts = make_prompts(
            label=label,
            numerical_columns=numerical_columns,
            tokenizer=llm.tokenizer,
            batch_size=generation_batch_size,
        )
        outputs = llm(
            [
                {
                    "input": prompt,
                    "gen_kwargs": {
                        "repetition_penalty": 1.0,
                    },
                }
                for prompt in prompts
            ]
        )
        texts = [(output["input"] + output["result"]).replace("\n", " ") for output in outputs]
        raw_count += len(texts)
        generated_df = convert_texts_to_tabular_data(
            text=texts,
            original_dataset=original_dataset,
            text_template="fim_template_textual_encoding",
        )
        parsed_count += int(generated_df.shape[0])
        generated_df = filter_generated(generated_df, numerical_columns, label)
        if not generated_df.empty:
            rows.append(generated_df)
        if rows:
            combined = (
                pd.concat(rows, ignore_index=True)
                .round(3)
                .drop_duplicates(subset=numerical_columns)
                .reset_index(drop=True)
            )
            accepted_count = int(combined.shape[0])
        attempts += 1

    if rows:
        combined = (
            pd.concat(rows, ignore_index=True)
            .round(3)
            .drop_duplicates(subset=numerical_columns)
            .reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=numerical_columns + ["data"])
    combined = combined.iloc[:max_per_label].copy()
    stats = {
        "raw_outputs": raw_count,
        "parsed_rows": parsed_count,
        "accepted_rows": int(combined.shape[0]),
        "attempts": attempts,
    }
    return combined, stats


def write_jsonl(path: Path, task_name: str, task_description: str, channel: str, label: str, windows: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for window in windows:
            record = {
                "task_name": task_name,
                "is_seed": False,
                "task_description": task_description,
                "requested_label": label,
                "generated_time_series": {channel: [float(v) for v in window]},
            }
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walking_windows = load_activity_windows(
        args.walking_parquet,
        args.channel,
        args.train_length,
        args.window_length,
        args.min_windows_number,
        args.train_splitting,
    )
    running_windows = load_activity_windows(
        args.running_parquet,
        args.channel,
        args.train_length,
        args.window_length,
        args.min_windows_number,
        args.train_splitting,
    )
    combined_windows = np.concatenate([walking_windows, running_windows], axis=0)
    combined_preprocessed = combined_windows[None, :, :]

    embedded_data, embedding_dims, data_embedded, fica_mixing, fica_mean = fica_embed_data(
        combined_preprocessed,
        args.embedding_dim,
        args.variance_explained,
    )
    labels = ["walking"] * walking_windows.shape[0] + ["running"] * running_windows.shape[0]
    embedded_data["data"] = labels
    numerical_columns = [col for col in embedded_data.columns if col != "data"]
    embedded_data.to_csv(args.output_dir / "training_embeddings_with_labels.csv", index=False)

    sdforger_params = {
        "k_bit": None,
        "embedding_type": "fica",
        "embedding_dim": args.embedding_dim,
        "variance_explained": args.variance_explained,
        "min_windows_number": args.min_windows_number,
        "min_windows_length": args.window_length,
        "input_tokens_precision": args.input_tokens_precision,
        "text_template": "fim_template_textual_encoding",
    }
    trainer = SDForgerTuningBlock(
        model_id_or_path=args.model_id_or_path,
        learning_rate=0.00008,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        seed=args.seed,
    )
    model_args = {
        "dtype": torch.float32,
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True,
    }
    tuned_path = trainer(
        output_dir=str(args.output_dir / "model"),
        dataset=embedded_data,
        model_args=model_args,
        sdforger_params=sdforger_params,
        model_id_or_path=args.model_id_or_path,
    )

    target_args = {
        "type": "vllm",
        MODEL_ID_OR_PATH: tuned_path,
        "dtype": "float32",
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True,
        "temperature": 1.3,
        "max_tokens": 1000,
    }
    llm = get_block("vllm", **target_args)

    generated_by_label: dict[str, pd.DataFrame] = {}
    generation_stats: dict[str, dict[str, int]] = {}
    for label in ["walking", "running"]:
        generated_df, stats = generate_for_label(
            llm=llm,
            original_dataset=embedded_data,
            numerical_columns=numerical_columns,
            label=label,
            min_per_label=args.min_per_label,
            max_per_label=args.max_per_label,
            generation_batch_size=args.generation_batch_size,
        )
        generated_by_label[label] = generated_df
        generation_stats[label] = stats
        generated_df.to_csv(args.output_dir / f"{label}_generated_embeddings.csv", index=False)

        reconstructed = fica_transform_to_original_feature_space(
            generated_df[numerical_columns].to_numpy(dtype=np.float64),
            combined_preprocessed,
            data_embedded,
            fica_mixing,
            fica_mean,
        )[0]
        write_jsonl(
            args.output_dir / f"{label}_final_data.jsonl",
            task_name="time_series/pamap2_subject101_unified_label_conditioned_hand_acc16_x",
            task_description="Unified walking/running label-conditioned SDForger-style PAMAP2 subject101 hand_acc16_x run.",
            channel=args.channel,
            label=label,
            windows=reconstructed,
        )

    metadata = {
        "walking_train_windows": int(walking_windows.shape[0]),
        "running_train_windows": int(running_windows.shape[0]),
        "combined_train_windows": int(combined_windows.shape[0]),
        "channel": args.channel,
        "train_length_per_activity": args.train_length,
        "window_length": args.window_length,
        "embedding_type": "fica",
        "embedding_dims": embedding_dims,
        "model_id_or_path": args.model_id_or_path,
        "tuned_model_path": str(tuned_path),
        "generation_stats": generation_stats,
        "value_space": "standardized_sdforger_window_space",
        "note": "window-level labels are assigned after independent activity-specific preprocessing and before joint FICA embedding.",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
