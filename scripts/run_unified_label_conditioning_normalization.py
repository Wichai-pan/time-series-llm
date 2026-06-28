#!/usr/bin/env python3
"""Unified label-conditioned SDForger run with controlled normalization modes.

The original unified script used SDForger preprocessing separately per activity,
which standardizes window columns before joint FICA. This variant makes that
choice explicit and adds alternatives that change the actual FICA input space.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

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
    preprocess_train_data,
)


class HFLLM:
    """transformers-backed drop-in for the vLLM block (vLLM crashes on Qwen).

    Mimics the minimal interface used by generate_for_label: a ``.tokenizer``
    attribute and a callable taking [{"input": prompt, "gen_kwargs": {...}}] and
    returning [{"input": "", "result": full_decoded_text}] (input+result = full text,
    same as the vLLM block, so downstream parsing is unchanged)."""

    def __init__(self, model_path, temperature=1.3, max_new_tokens=1000, dtype=None, batch_size=16):
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype or _torch.float16, trust_remote_code=True).to("cuda").eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._torch = _torch

    def __call__(self, batch):
        prompts = [d["input"] for d in batch]
        rep = batch[0].get("gen_kwargs", {}).get("repetition_penalty", 1.0) if batch else 1.0
        outs = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i:i + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with self._torch.no_grad():
                g = self.model.generate(**enc, do_sample=True, temperature=self.temperature, top_p=0.9,
                                        max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                                        repetition_penalty=rep)
            for j in range(len(chunk)):
                full = self.tokenizer.decode(g[j], skip_special_tokens=True).replace("\n", " ")
                outs.append({"input": "", "result": full})
        return outs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-parquet", type=Path, required=True)
    parser.add_argument("--running-parquet", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id-or-path", default="gpt2")
    parser.add_argument("--gen-backend", choices=["vllm", "hf"], default="vllm",
                        help="hf=transformers (use for Qwen; vLLM crashes on Qwen)")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--repair", choices=["none", "clip"], default="none",
                        help="clip: per-activity p05/p95 clip of generated coeffs IN THIS NORM BASIS before decode")
    parser.add_argument("--normalization-mode", required=True, choices=[
        "current_activity_window_zscore",
        "joint_window_zscore",
        "global_series_zscore",
        "activity_series_zscore",
        "raw_window",
    ])
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


def zscore_df(df: pd.DataFrame, channel: str, mean: float, std: float) -> pd.DataFrame:
    out = df.copy()
    safe_std = 1.0 if np.isclose(std, 0.0) else float(std)
    out[channel] = (pd.to_numeric(out[channel], errors="raise").astype(float) - float(mean)) / safe_std
    return out


def load_preprocessed(
    parquet: Path,
    channel: str,
    train_length: int,
    window_length: int,
    min_windows_number: int,
    train_splitting: str,
    *,
    series_mean: float | None = None,
    series_std: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet)
    if series_mean is not None and series_std is not None:
        df = zscore_df(df, channel, series_mean, series_std)
    scaled, original, _scalers = preprocess_train_data(
        df,
        train_channels=[channel],
        train_length=train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=window_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    return np.asarray(scaled[0], dtype=np.float64), np.asarray(original[0], dtype=np.float64)


def training_values(parquet: Path, channel: str, train_length: int) -> np.ndarray:
    df = pd.read_parquet(parquet)
    return pd.to_numeric(df[channel].iloc[:train_length], errors="raise").to_numpy(dtype=np.float64)


def build_windows(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    norm = args.normalization_mode
    metadata: dict[str, object] = {"normalization_mode": norm}

    if norm == "current_activity_window_zscore":
        walking_scaled, walking_original = load_preprocessed(
            args.walking_parquet, args.channel, args.train_length, args.window_length,
            args.min_windows_number, args.train_splitting
        )
        running_scaled, running_original = load_preprocessed(
            args.running_parquet, args.channel, args.train_length, args.window_length,
            args.min_windows_number, args.train_splitting
        )
        metadata["fica_input_space"] = "activity-specific SDForger timestamp-wise StandardScaler output"
        metadata["note"] = "This reproduces the prior raw-unified preprocessing path."
        return walking_scaled, running_scaled, metadata

    if norm == "global_series_zscore":
        values = np.concatenate([
            training_values(args.walking_parquet, args.channel, args.train_length),
            training_values(args.running_parquet, args.channel, args.train_length),
        ])
        mean, std = float(values.mean()), float(values.std())
        walking_scaled, walking_original = load_preprocessed(
            args.walking_parquet, args.channel, args.train_length, args.window_length,
            args.min_windows_number, args.train_splitting, series_mean=mean, series_std=std
        )
        running_scaled, running_original = load_preprocessed(
            args.running_parquet, args.channel, args.train_length, args.window_length,
            args.min_windows_number, args.train_splitting, series_mean=mean, series_std=std
        )
        metadata.update({
            "fica_input_space": "raw segmented windows after one global scalar z-score; SDForger window scaler is bypassed",
            "global_mean": mean,
            "global_std": std,
        })
        return walking_original, running_original, metadata

    if norm == "activity_series_zscore":
        walking_values = training_values(args.walking_parquet, args.channel, args.train_length)
        running_values = training_values(args.running_parquet, args.channel, args.train_length)
        w_mean, w_std = float(walking_values.mean()), float(walking_values.std())
        r_mean, r_std = float(running_values.mean()), float(running_values.std())
        walking_scaled, walking_original = load_preprocessed(
            args.walking_parquet, args.channel, args.train_length, args.window_length,
            args.min_windows_number, args.train_splitting, series_mean=w_mean, series_std=w_std
        )
        running_scaled, running_original = load_preprocessed(
            args.running_parquet, args.channel, args.train_length, args.window_length,
            args.min_windows_number, args.train_splitting, series_mean=r_mean, series_std=r_std
        )
        metadata.update({
            "fica_input_space": "raw segmented windows after activity-specific scalar z-score; SDForger window scaler is bypassed",
            "walking_mean": w_mean,
            "walking_std": w_std,
            "running_mean": r_mean,
            "running_std": r_std,
        })
        return walking_original, running_original, metadata

    walking_scaled, walking_original = load_preprocessed(
        args.walking_parquet, args.channel, args.train_length, args.window_length,
        args.min_windows_number, args.train_splitting
    )
    running_scaled, running_original = load_preprocessed(
        args.running_parquet, args.channel, args.train_length, args.window_length,
        args.min_windows_number, args.train_splitting
    )

    if norm == "raw_window":
        metadata["fica_input_space"] = "raw segmented windows; SDForger window scaler is bypassed"
        return walking_original, running_original, metadata

    if norm == "joint_window_zscore":
        combined_original = np.concatenate([walking_original, running_original], axis=0)
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_original)
        n_walking = walking_original.shape[0]
        metadata["fica_input_space"] = "one joint timestamp-wise StandardScaler fitted over walking+running windows"
        metadata["joint_scaler_mean_shape"] = list(scaler.mean_.shape)
        return combined_scaled[:n_walking], combined_scaled[n_walking:], metadata

    raise ValueError(f"Unsupported normalization mode: {norm}")


def make_prompts(*, label: str, numerical_columns: list[str], tokenizer, batch_size: int, permute: bool = True) -> list[str]:
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


def filter_generated(generated_df: pd.DataFrame, numerical_columns: list[str], requested_label: str) -> pd.DataFrame:
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
            [{"input": prompt, "gen_kwargs": {"repetition_penalty": 1.0}} for prompt in prompts]
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


def write_jsonl(path: Path, task_name: str, task_description: str, channel: str, label: str, windows: np.ndarray, norm: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for window in windows:
            record = {
                "task_name": task_name,
                "is_seed": False,
                "task_description": task_description,
                "requested_label": label,
                "normalization_mode": norm,
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

    walking_windows, running_windows, normalization_metadata = build_windows(args)
    combined_windows = np.concatenate([walking_windows, running_windows], axis=0)
    combined_preprocessed = combined_windows[None, :, :]

    np.save(args.output_dir / "walking_real_windows_model_space.npy", walking_windows)
    np.save(args.output_dir / "running_real_windows_model_space.npy", running_windows)

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
        per_device_train_batch_size=args.train_batch_size,
        seed=args.seed,
    )
    model_args = {
        "dtype": torch.float16 if args.dtype == "float16" else torch.float32,
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

    if args.gen_backend == "hf":
        llm = HFLLM(tuned_path, temperature=1.3, max_new_tokens=1000, dtype=torch.float16)
    else:
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

    generation_stats: dict[str, dict[str, int]] = {}
    decoded_summary: dict[str, dict[str, float | int]] = {}
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
        generation_stats[label] = stats
        generated_df.to_csv(args.output_dir / f"{label}_generated_embeddings.csv", index=False)

        gen_coeffs = generated_df[numerical_columns].to_numpy(dtype=np.float64)
        if args.repair == "clip":
            tr = embedded_data[embedded_data["data"] == label][numerical_columns].to_numpy(dtype=np.float64)
            lo = np.percentile(tr, 5, axis=0); hi = np.percentile(tr, 95, axis=0)
            gen_coeffs = np.clip(gen_coeffs, lo, hi)  # per-activity p05/p95 clip in this norm-space FICA basis
        reconstructed = fica_transform_to_original_feature_space(
            gen_coeffs,
            combined_preprocessed,
            data_embedded,
            fica_mixing,
            fica_mean,
        )[0]
        np.save(args.output_dir / f"{label}_synthetic_windows_model_space.npy", reconstructed)
        write_jsonl(
            args.output_dir / f"{label}_final_data.jsonl",
            task_name=f"time_series/pamap2_subject101_unified_label_conditioned_{args.normalization_mode}",
            task_description=f"Unified walking/running label-conditioned SDForger-style run with normalization mode {args.normalization_mode}.",
            channel=args.channel,
            label=label,
            windows=reconstructed,
            norm=args.normalization_mode,
        )
        decoded_summary[label] = {
            "synthetic_windows": int(reconstructed.shape[0]),
            "synthetic_abs_max": float(np.max(np.abs(reconstructed))) if reconstructed.size else float("nan"),
            "synthetic_std_mean": float(np.std(reconstructed, axis=1).mean()) if reconstructed.size else float("nan"),
        }

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
        "decoded_summary": decoded_summary,
        "value_space": "normalization-mode-specific model space",
        "normalization": normalization_metadata,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
