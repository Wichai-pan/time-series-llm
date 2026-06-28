#!/usr/bin/env python3
"""Stage B: synthetic x1/x3 amplitude control — maximally-separable numeric conditioning test.

Tests H1 (encoding too sparse) vs H3 (GPT-2 cannot condition on numbers) cleanly:
each training window's latent coefficients are duplicated as two rows — original (token
'a0', x1) and x3-scaled (token 'a1', x3) — so the conditioning token PERFECTLY and
maximally separates a numeric amplitude. If the model learns this trivial contrast,
numeric conditioning IS learnable and the earlier T2 failure was encoding sparsity (H1);
if it cannot, that is strong H3 evidence.

Generation uses BALANCED requests (equal N per amp), temperature 1.3 (the distributional
level where conditioning operates — A3 showed temp=0 collapses even working label cond),
and records latent L2 (the directly-controllable target) + decoded window stats.

Outputs: amplitude_adherence.csv (requested_amp, latent coeffs/L2, realized window stats),
run_metadata.json.
"""
from __future__ import annotations

import argparse
import json
import random
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
import fms_dgt.core.blocks.llm.vllm.vllm_in_proc  # noqa: F401
from fms_dgt.public.databuilders.time_series.trainer import SDForgerTuningBlock
from fms_dgt.public.databuilders.time_series.utils import (
    convert_texts_to_tabular_data,
    fica_embed_data,
    fica_transform_to_original_feature_space,
    preprocess_train_data,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walking-parquet", type=Path, required=True)
    p.add_argument("--running-parquet", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id-or-path", default="gpt2")
    p.add_argument("--train-length", type=int, default=5000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    p.add_argument("--scale", type=float, default=3.0, help="x3 amplitude factor for the 'a1' group")
    p.add_argument("--per-amp", type=int, default=40, help="balanced target accepted windows per (label, amp)")
    p.add_argument("--max-per-amp", type=int, default=80)
    p.add_argument("--generation-batch-size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_windows(parquet, channel, tl, wl, mwn, ts) -> np.ndarray:
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[channel], train_length=tl,
                                            train_samples=1, augmentation_strategy="univariate",
                                            min_windows_length=wl, min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def make_prompts(composite, cols, tokenizer, batch_size):
    prompts = []
    for _ in range(batch_size):
        c = cols.copy(); random.shuffle(c)
        ti = ", ".join(f"{x} is [blank]" for x in c)
        prompts.append(f"Condition: data is {composite} [sep] Input: {ti} [sep] Target:")
    lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    mx = max(lens); pad = tokenizer.pad_token
    return [pad * (mx - n) + p for p, n in zip(prompts, lens)]


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walking = load_windows(args.walking_parquet, args.channel, args.train_length, args.window_length,
                           args.min_windows_number, args.train_splitting)
    running = load_windows(args.running_parquet, args.channel, args.train_length, args.window_length,
                           args.min_windows_number, args.train_splitting)
    combined = np.concatenate([walking, running], axis=0)
    combined_pre = combined[None, :, :]
    embedded, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, args.embedding_dim, args.variance_explained)
    cols = list(embedded.columns)
    labels = ["walking"] * walking.shape[0] + ["running"] * running.shape[0]
    base = embedded[cols].to_numpy(dtype=np.float64)

    # x1 (a0) and x3 (a1) duplicate -> perfectly separable amplitude token
    r_x1 = pd.DataFrame(base, columns=cols); r_x1["data"] = [f"{l}|a0" for l in labels]
    r_x3 = pd.DataFrame(base * args.scale, columns=cols); r_x3["data"] = [f"{l}|a1" for l in labels]
    train = pd.concat([r_x1, r_x3], ignore_index=True)
    train.to_csv(args.output_dir / "training_amplitude_control.csv", index=False)

    sdforger_params = {"k_bit": None, "embedding_type": "fica", "embedding_dim": args.embedding_dim,
                       "variance_explained": args.variance_explained, "min_windows_number": args.min_windows_number,
                       "min_windows_length": args.window_length, "input_tokens_precision": 4,
                       "text_template": "fim_template_textual_encoding"}
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    trainer = SDForgerTuningBlock(model_id_or_path=args.model_id_or_path, learning_rate=0.00008,
                                  num_train_epochs=args.epochs, per_device_train_batch_size=args.train_batch_size, seed=args.seed)
    tuned = trainer(output_dir=str(args.output_dir / "model"), dataset=train,
                    model_args={"dtype": torch_dtype, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                    sdforger_params=sdforger_params, model_id_or_path=args.model_id_or_path)

    llm = get_block("vllm", **{"type": "vllm", MODEL_ID_OR_PATH: tuned, "dtype": args.dtype,
                               "trust_remote_code": True, "ignore_mismatched_sizes": True,
                               "temperature": args.temperature, "max_tokens": 1000})

    rows, gstats = [], {}
    for label in ["walking", "running"]:
        for amp in [0, 1]:
            composite = f"{label}|a{amp}"
            accepted, attempts, raw = [], 0, 0
            while sum(len(a) for a in accepted) < args.per_amp and attempts < 12:
                prompts = make_prompts(composite, cols, llm.tokenizer, args.generation_batch_size)
                outs = llm([{"input": p, "gen_kwargs": {"repetition_penalty": 1.0}} for p in prompts])
                texts = [(o["input"] + o["result"]).replace("\n", " ") for o in outs]
                raw += len(texts)
                gdf = convert_texts_to_tabular_data(text=texts, original_dataset=train, text_template="fim_template_textual_encoding")
                gdf[cols] = gdf[cols].apply(pd.to_numeric, errors="coerce")
                gdf = gdf.dropna(subset=cols)
                gdf = gdf[gdf["data"].astype(str).str.split("|").str[0].str.strip() == label]
                if not gdf.empty:
                    accepted.append(gdf)
                attempts += 1
            gstats[composite] = {"raw": raw, "accepted": int(sum(len(a) for a in accepted)), "attempts": attempts}
            if not accepted:
                continue
            g = pd.concat(accepted, ignore_index=True).drop_duplicates(subset=cols).reset_index(drop=True).iloc[:args.max_per_amp]
            lat = g[cols].to_numpy(dtype=np.float64)
            decoded = fica_transform_to_original_feature_space(lat, combined_pre, data_embedded, mixing, mean)[0]
            for i in range(len(g)):
                rows.append({"label": label, "requested_amp": amp, "composite": composite,
                             "latent_l2": float(np.linalg.norm(lat[i])),
                             "latent_maxabs": float(np.abs(lat[i]).max()),
                             "realized_max": float(decoded[i].max()), "realized_min": float(decoded[i].min()),
                             "realized_std": float(decoded[i].std()),
                             "realized_absmax": float(np.abs(decoded[i]).max())})

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "amplitude_adherence.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(
        {"scale": args.scale, "temperature": args.temperature, "train_rows": len(train),
         "generation_stats": gstats, "generated_rows": len(out),
         "note": "synthetic x1(a0)/x3(a1) amplitude control; balanced per-amp requests"}, indent=2) + "\n")
    print("generated rows=%d" % len(out))
    print(json.dumps(gstats, indent=2))


if __name__ == "__main__":
    main()
