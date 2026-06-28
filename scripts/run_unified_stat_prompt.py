#!/usr/bin/env python3
"""Unified walking/running STAT-PROMPT SDForger-style run (T2).

Same pipeline as run_unified_label_conditioning.py, except the `Condition:` token
carries a per-window statistical profile in addition to the activity label:

    Condition: data is walking|m2|s3|n1|x4 [sep] Input: ... [sep] Target:

where m/s/n/x are quantile bins of the window's mean / std / min / max, computed
per activity. The hypothesis (T2): giving the LLM an amplitude/shape prior at the
source reduces out-of-range latent generation, lowering the need for post-hoc repair.

The composite is folded into the SINGLE existing `data` categorical column, so the
existing text encoder/parser and all T1 repair/eval scripts (which read value_* and
decode) work unchanged. Only the label match in filtering uses the label prefix.

Outputs (per label): {label}_generated_embeddings.csv (value_* + composite data),
{label}_final_data.jsonl (decoded raw, standardized space), stat_adherence.csv
(requested bins vs realized window stats), run_metadata.json.
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
    preprocess_train_data,
)

STAT_KEYS = ["m", "s", "n", "x"]  # mean, std, min, max


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
    parser.add_argument("--n-bins", type=int, default=4, help="quantile bins per stat")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_activity_windows(parquet, channel, train_length, window_length, min_windows_number, train_splitting) -> np.ndarray:
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(
        df, train_channels=[channel], train_length=train_length, train_samples=1,
        augmentation_strategy="univariate", min_windows_length=window_length,
        min_windows_number=min_windows_number, train_splitting=train_splitting,
    )
    return np.asarray(scaled[0], dtype=np.float64)


def window_stat_values(windows: np.ndarray) -> dict[str, np.ndarray]:
    return {"m": windows.mean(axis=1), "s": windows.std(axis=1),
            "n": windows.min(axis=1), "x": windows.max(axis=1)}


def bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(values, qs)


def composites_for_activity(windows: np.ndarray, label: str, n_bins: int) -> tuple[list[str], dict]:
    """Return one composite string per window + the per-stat bin edges (for reuse)."""
    stat_vals = window_stat_values(windows)
    edges = {k: bin_edges(stat_vals[k], n_bins) for k in STAT_KEYS}
    bins = {k: np.digitize(stat_vals[k], edges[k]) for k in STAT_KEYS}  # 0..n_bins-1
    composites = [
        f"{label}|m{bins['m'][i]}|s{bins['s'][i]}|n{bins['n'][i]}|x{bins['x'][i]}"
        for i in range(windows.shape[0])
    ]
    return composites, edges


def make_prompts(*, composite_pool: list[str], numerical_columns, tokenizer, batch_size, permute=True) -> tuple[list[str], list[str]]:
    prompts, chosen = [], []
    for _ in range(batch_size):
        cols = numerical_columns.copy()
        if permute:
            random.shuffle(cols)
        comp = random.choice(composite_pool)
        chosen.append(comp)
        text_input = ", ".join(f"{col} is [blank]" for col in cols)
        prompts.append(f"Condition: data is {comp} [sep] Input: {text_input} [sep] Target:")
    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    max_len = max(lengths)
    pad = tokenizer.pad_token
    prompts = [pad * (max_len - n) + p for p, n in zip(prompts, lengths)]
    return prompts, chosen


def filter_generated(generated_df: pd.DataFrame, numerical_columns, requested_label) -> pd.DataFrame:
    if generated_df.empty:
        return generated_df
    generated_df = generated_df.replace("NaN", np.nan).dropna(how="all")
    generated_df = generated_df.dropna(subset=numerical_columns + ["data"])
    # label is the prefix of the composite "label|m..|s..|n..|x.."
    label_prefix = generated_df["data"].astype(str).str.split("|").str[0].str.strip()
    generated_df = generated_df.loc[label_prefix == requested_label]
    for col in numerical_columns:
        generated_df[col] = pd.to_numeric(generated_df[col], errors="coerce")
    return generated_df.dropna(subset=numerical_columns)


def generate_for_label(*, llm, original_dataset, numerical_columns, label, composite_pool,
                       min_per_label, max_per_label, generation_batch_size):
    rows, raw_count, parsed_count, accepted_count, attempts = [], 0, 0, 0, 0
    while accepted_count < min_per_label and attempts < 12:
        prompts, _chosen = make_prompts(
            composite_pool=composite_pool, numerical_columns=numerical_columns,
            tokenizer=llm.tokenizer, batch_size=generation_batch_size)
        outputs = llm([{"input": p, "gen_kwargs": {"repetition_penalty": 1.0}} for p in prompts])
        texts = [(o["input"] + o["result"]).replace("\n", " ") for o in outputs]
        raw_count += len(texts)
        gdf = convert_texts_to_tabular_data(text=texts, original_dataset=original_dataset,
                                            text_template="fim_template_textual_encoding")
        parsed_count += int(gdf.shape[0])
        gdf = filter_generated(gdf, numerical_columns, label)
        if not gdf.empty:
            rows.append(gdf)
        if rows:
            combined = (pd.concat(rows, ignore_index=True).round(3)
                        .drop_duplicates(subset=numerical_columns).reset_index(drop=True))
            accepted_count = int(combined.shape[0])
        attempts += 1
    if rows:
        combined = (pd.concat(rows, ignore_index=True).round(3)
                    .drop_duplicates(subset=numerical_columns).reset_index(drop=True))
    else:
        combined = pd.DataFrame(columns=numerical_columns + ["data"])
    combined = combined.iloc[:max_per_label].copy()
    stats = {"raw_outputs": raw_count, "parsed_rows": parsed_count,
             "accepted_rows": int(combined.shape[0]), "attempts": attempts}
    return combined, stats


def write_jsonl(path, task_name, task_description, channel, label, windows) -> None:
    with path.open("w", encoding="utf-8") as h:
        for w in windows:
            h.write(json.dumps({
                "task_name": task_name, "is_seed": False, "task_description": task_description,
                "requested_label": label, "generated_time_series": {channel: [float(v) for v in w]},
            }) + "\n")


def requested_bins_from_composite(comp: str) -> dict[str, int]:
    out = {}
    for part in str(comp).split("|")[1:]:
        out[part[0]] = int(part[1:])
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walking = load_activity_windows(args.walking_parquet, args.channel, args.train_length,
                                    args.window_length, args.min_windows_number, args.train_splitting)
    running = load_activity_windows(args.running_parquet, args.channel, args.train_length,
                                    args.window_length, args.min_windows_number, args.train_splitting)
    combined = np.concatenate([walking, running], axis=0)
    combined_pre = combined[None, :, :]

    embedded, embedding_dims, data_embedded, fica_mixing, fica_mean = fica_embed_data(
        combined_pre, args.embedding_dim, args.variance_explained)

    walk_comp, walk_edges = composites_for_activity(walking, "walking", args.n_bins)
    run_comp, run_edges = composites_for_activity(running, "running", args.n_bins)
    embedded["data"] = walk_comp + run_comp
    numerical_columns = [c for c in embedded.columns if c != "data"]
    composite_pool = {"walking": walk_comp, "running": run_comp}
    embedded.to_csv(args.output_dir / "training_embeddings_with_stat_labels.csv", index=False)

    sdforger_params = {"k_bit": None, "embedding_type": "fica", "embedding_dim": args.embedding_dim,
                       "variance_explained": args.variance_explained, "min_windows_number": args.min_windows_number,
                       "min_windows_length": args.window_length, "input_tokens_precision": args.input_tokens_precision,
                       "text_template": "fim_template_textual_encoding"}
    trainer = SDForgerTuningBlock(model_id_or_path=args.model_id_or_path, learning_rate=0.00008,
                                  num_train_epochs=100, per_device_train_batch_size=32, seed=args.seed)
    tuned_path = trainer(output_dir=str(args.output_dir / "model"), dataset=embedded,
                         model_args={"dtype": torch.float32, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                         sdforger_params=sdforger_params, model_id_or_path=args.model_id_or_path)

    llm = get_block("vllm", **{"type": "vllm", MODEL_ID_OR_PATH: tuned_path, "dtype": "float32",
                               "trust_remote_code": True, "ignore_mismatched_sizes": True,
                               "temperature": 1.3, "max_tokens": 1000})

    generation_stats, adherence_rows = {}, []
    for label in ["walking", "running"]:
        gdf, stats = generate_for_label(
            llm=llm, original_dataset=embedded, numerical_columns=numerical_columns, label=label,
            composite_pool=composite_pool[label], min_per_label=args.min_per_label,
            max_per_label=args.max_per_label, generation_batch_size=args.generation_batch_size)
        generation_stats[label] = stats
        gdf.to_csv(args.output_dir / f"{label}_generated_embeddings.csv", index=False)

        if not gdf.empty:
            decoded = fica_transform_to_original_feature_space(
                gdf[numerical_columns].to_numpy(dtype=np.float64), combined_pre,
                data_embedded, fica_mixing, fica_mean)[0]
            write_jsonl(args.output_dir / f"{label}_final_data.jsonl",
                        "time_series/pamap2_subject101_unified_stat_prompt_hand_acc16_x",
                        "Unified walking/running stat-prompt SDForger-style run.",
                        args.channel, label, decoded)
            # stat-adherence: requested bin vs realized decoded-window stat
            realized = window_stat_values(decoded)
            for i, comp in enumerate(gdf["data"].tolist()):
                req = requested_bins_from_composite(comp)
                adherence_rows.append({
                    "label": label, "composite": comp,
                    **{f"req_{k}": req.get(k) for k in STAT_KEYS},
                    "realized_mean": float(realized["m"][i]), "realized_std": float(realized["s"][i]),
                    "realized_min": float(realized["n"][i]), "realized_max": float(realized["x"][i]),
                })

    pd.DataFrame(adherence_rows).to_csv(args.output_dir / "stat_adherence.csv", index=False)
    metadata = {
        "walking_train_windows": int(walking.shape[0]), "running_train_windows": int(running.shape[0]),
        "combined_train_windows": int(combined.shape[0]), "channel": args.channel,
        "train_length_per_activity": args.train_length, "window_length": args.window_length,
        "embedding_type": "fica", "embedding_dims": embedding_dims, "n_bins": args.n_bins,
        "model_id_or_path": args.model_id_or_path, "tuned_model_path": str(tuned_path),
        "generation_stats": generation_stats, "value_space": "standardized_sdforger_window_space",
        "conditioning": "label + quantized per-window mean/std/min/max bins, folded into composite data token",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
