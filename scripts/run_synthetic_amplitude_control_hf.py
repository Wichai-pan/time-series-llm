#!/usr/bin/env python3
"""Synthetic x1/x3 amplitude control with HuggingFace-transformers generation.

Identical to run_synthetic_amplitude_control.py (training via SDForgerTuningBlock) but
generates with transformers model.generate() instead of vLLM, to avoid the vLLM+Qwen
'LLVM ERROR: Failed to compute parent layout' crash on this cluster. Lets us run the
stronger-model (Qwen2.5-1.5B) numeric-conditioning test.
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

from transformers import AutoModelForCausalLM, AutoTokenizer

from fms_dgt.public.databuilders.time_series.trainer import SDForgerTuningBlock
from fms_dgt.public.databuilders.time_series.utils import (
    convert_texts_to_tabular_data, fica_embed_data,
    fica_transform_to_original_feature_space, preprocess_train_data)


def parse_args():
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
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--per-amp", type=int, default=40)
    p.add_argument("--max-per-amp", type=int, default=80)
    p.add_argument("--generation-batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_windows(parquet, ch, tl, wl, mwn, ts):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[ch], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def main():
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float16 if a.dtype == "float16" else torch.float32

    walking = load_windows(a.walking_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
    running = load_windows(a.running_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
    combined = np.concatenate([walking, running], axis=0); combined_pre = combined[None, :, :]
    embedded, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, a.embedding_dim, a.variance_explained)
    cols = list(embedded.columns)
    labels = ["walking"] * walking.shape[0] + ["running"] * running.shape[0]
    base = embedded[cols].to_numpy(dtype=np.float64)
    r1 = pd.DataFrame(base, columns=cols); r1["data"] = [f"{l}|a0" for l in labels]
    r3 = pd.DataFrame(base * a.scale, columns=cols); r3["data"] = [f"{l}|a1" for l in labels]
    train = pd.concat([r1, r3], ignore_index=True)

    sdforger_params = {"k_bit": None, "embedding_type": "fica", "embedding_dim": a.embedding_dim,
                       "variance_explained": a.variance_explained, "min_windows_number": a.min_windows_number,
                       "min_windows_length": a.window_length, "input_tokens_precision": 4,
                       "text_template": "fim_template_textual_encoding"}
    trainer = SDForgerTuningBlock(model_id_or_path=a.model_id_or_path, learning_rate=0.00008,
                                  num_train_epochs=a.epochs, per_device_train_batch_size=a.train_batch_size, seed=a.seed)
    tuned = trainer(output_dir=str(a.output_dir / "model"), dataset=train,
                    model_args={"dtype": torch_dtype, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                    sdforger_params=sdforger_params, model_id_or_path=a.model_id_or_path)

    # ---- transformers generation (avoids vLLM) ----
    tok = AutoTokenizer.from_pretrained(tuned, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(tuned, torch_dtype=torch_dtype, trust_remote_code=True).to("cuda").eval()

    def generate(prompts):
        outs = []
        for i in range(0, len(prompts), a.generation_batch_size):
            batch = prompts[i:i + a.generation_batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=a.top_p,
                                     max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id,
                                     repetition_penalty=1.05)
            for j in range(len(batch)):
                outs.append(tok.decode(gen[j], skip_special_tokens=True).replace("\n", " "))
        return outs

    rows, gstats = [], {}
    for label in ["walking", "running"]:
        for amp in [0, 1]:
            comp = f"{label}|a{amp}"
            accepted, attempts, raw = [], 0, 0
            while sum(len(x) for x in accepted) < a.per_amp and attempts < 8:
                prompts = []
                for _ in range(a.generation_batch_size):
                    c = cols.copy(); random.shuffle(c)
                    ti = ", ".join(f"{x} is [blank]" for x in c)
                    prompts.append(f"Condition: data is {comp} [sep] Input: {ti} [sep] Target:")
                texts = generate(prompts); raw += len(texts)
                g = convert_texts_to_tabular_data(text=texts, original_dataset=train, text_template="fim_template_textual_encoding")
                g[cols] = g[cols].apply(pd.to_numeric, errors="coerce")
                g = g.dropna(subset=cols)
                g = g[g["data"].astype(str).str.split("|").str[0].str.strip() == label]
                if not g.empty:
                    accepted.append(g)
                attempts += 1
            gstats[comp] = {"raw": raw, "accepted": int(sum(len(x) for x in accepted)), "attempts": attempts}
            if not accepted:
                continue
            g = pd.concat(accepted, ignore_index=True).drop_duplicates(subset=cols).reset_index(drop=True).iloc[:a.max_per_amp]
            lat = g[cols].to_numpy(dtype=np.float64)
            dec = fica_transform_to_original_feature_space(lat, combined_pre, data_embedded, mixing, mean)[0]
            for i in range(len(g)):
                rows.append({"label": label, "requested_amp": amp,
                             "latent_l2": float(np.linalg.norm(lat[i])), "latent_maxabs": float(np.abs(lat[i]).max()),
                             "realized_max": float(dec[i].max()), "realized_absmax": float(np.abs(dec[i]).max()),
                             "realized_std": float(dec[i].std())})

    out = pd.DataFrame(rows)
    out.to_csv(a.output_dir / "amplitude_adherence.csv", index=False)
    (a.output_dir / "run_metadata.json").write_text(json.dumps(
        {"model": a.model_id_or_path, "scale": a.scale, "temperature": a.temperature, "epochs": a.epochs,
         "generation_backend": "transformers", "generation_stats": gstats, "generated_rows": len(out)}, indent=2) + "\n")
    print("generated rows=%d" % len(out)); print(json.dumps(gstats, indent=2))


if __name__ == "__main__":
    main()
