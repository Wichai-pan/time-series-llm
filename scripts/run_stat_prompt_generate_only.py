#!/usr/bin/env python3
"""A3 probe: generation-only from an EXISTING stat-prompt checkpoint, greedy (temp=0).

No fine-tuning. Loads the already-trained T2 model, requests each DISTINCT training
composite once (greedy -> deterministic one window per composite, no sampling noise,
no permute), decodes, and records realized window stats + the directly-emitted latent
magnitude per window. Lets us test: with sampling randomness removed, does realized
max (or latent L2) track the requested max-bin? (Isolates H2-randomness from H3-capability.)

Outputs: stat_adherence_temp0.csv (composite, req bins, realized stats, latent_maxabs/L2).
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
from fms_dgt.public.databuilders.time_series.utils import (
    convert_texts_to_tabular_data,
    fica_embed_data,
    fica_transform_to_original_feature_space,
    preprocess_train_data,
)

STAT_KEYS = ["m", "s", "n", "x"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walking-parquet", type=Path, required=True)
    p.add_argument("--running-parquet", type=Path, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--train-length", type=int, default=5000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_windows(parquet, channel, tl, wl, mwn, ts) -> np.ndarray:
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[channel], train_length=tl,
                                            train_samples=1, augmentation_strategy="univariate",
                                            min_windows_length=wl, min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def stat_values(w):
    return {"m": w.mean(axis=1), "s": w.std(axis=1), "n": w.min(axis=1), "x": w.max(axis=1)}


def composites(windows, label, n_bins):
    sv = stat_values(windows)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = {k: np.quantile(sv[k], qs) for k in STAT_KEYS}
    b = {k: np.digitize(sv[k], edges[k]) for k in STAT_KEYS}
    return [f"{label}|m{b['m'][i]}|s{b['s'][i]}|n{b['n'][i]}|x{b['x'][i]}" for i in range(windows.shape[0])]


def req_bins(comp):
    return {p[0]: int(p[1:]) for p in str(comp).split("|")[1:]}


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
    embedded["data"] = composites(walking, "walking", args.n_bins) + composites(running, "running", args.n_bins)
    cols = [c for c in embedded.columns if c != "data"]
    pools = {"walking": composites(walking, "walking", args.n_bins), "running": composites(running, "running", args.n_bins)}

    llm = get_block("vllm", **{"type": "vllm", MODEL_ID_OR_PATH: args.model_path, "dtype": "float32",
                               "trust_remote_code": True, "ignore_mismatched_sizes": True,
                               "temperature": args.temperature, "max_tokens": 1000})

    rows = []
    for label in ["walking", "running"]:
        distinct = sorted(set(pools[label]))  # one deterministic greedy decode per distinct composite
        prompts = []
        for comp in distinct:
            ti = ", ".join(f"{c} is [blank]" for c in cols)  # no permute -> deterministic
            prompts.append(f"Condition: data is {comp} [sep] Input: {ti} [sep] Target:")
        lengths = [len(llm.tokenizer(p)["input_ids"]) for p in prompts]
        mx = max(lengths); pad = llm.tokenizer.pad_token
        prompts = [pad * (mx - n) + p for p, n in zip(prompts, lengths)]
        outs = llm([{"input": p, "gen_kwargs": {"repetition_penalty": 1.0}} for p in prompts])
        texts = [(o["input"] + o["result"]).replace("\n", " ") for o in outs]
        gdf = convert_texts_to_tabular_data(text=texts, original_dataset=embedded, text_template="fim_template_textual_encoding")
        gdf[cols] = gdf[cols].apply(pd.to_numeric, errors="coerce")
        gdf = gdf.dropna(subset=cols)
        gdf = gdf[gdf["data"].astype(str).str.split("|").str[0].str.strip() == label]
        if gdf.empty:
            continue
        lat = gdf[cols].to_numpy(dtype=np.float64)
        decoded = fica_transform_to_original_feature_space(lat, combined_pre, data_embedded, mixing, mean)[0]
        rv = stat_values(decoded)
        for i, comp in enumerate(gdf["data"].tolist()):
            rb = req_bins(comp)
            rows.append({"label": label, "composite": comp,
                         **{f"req_{k}": rb.get(k) for k in STAT_KEYS},
                         "realized_max": float(rv["x"][i]), "realized_std": float(rv["s"][i]),
                         "realized_min": float(rv["n"][i]), "realized_mean": float(rv["m"][i]),
                         "latent_maxabs": float(np.abs(lat[i]).max()),
                         "latent_l2": float(np.linalg.norm(lat[i]))})

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "stat_adherence_temp0.csv", index=False)
    (args.output_dir / "probe_metadata.json").write_text(json.dumps(
        {"model_path": args.model_path, "temperature": args.temperature,
         "distinct_walking": len(set(pools["walking"])), "distinct_running": len(set(pools["running"])),
         "generated_rows": len(out)}, indent=2) + "\n")
    print("rows=%d  walking=%d running=%d" % (len(out), int((out.label == "walking").sum()), int((out.label == "running").sum())))


if __name__ == "__main__":
    main()
