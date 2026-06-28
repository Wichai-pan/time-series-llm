#!/usr/bin/env python3
"""Unified N-activity label-conditioned SDForger run (generalizes the 2-class version).

Takes any number of --activity LABEL=PARQUET pairs (e.g. walking=...parquet running=...
cycling=...). Each activity is preprocessed independently, all windows are pooled into one
joint FICA embedding with a window-level activity label, one conditional GPT-2 is trained,
and per-label outputs are generated. Output format matches run_unified_label_conditioning.py
(per-label generated_embeddings.csv + final_data.jsonl) so the existing repair/eval reuse.
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


class HFLLM:
    """transformers-backed drop-in for the vLLM block (vLLM crashes on Qwen)."""

    def __init__(self, model_path, temperature=1.3, max_new_tokens=1000, dtype=None, batch_size=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype or torch.float16, trust_remote_code=True).to("cuda").eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def __call__(self, batch):
        prompts = [d["input"] for d in batch]
        rep = batch[0].get("gen_kwargs", {}).get("repetition_penalty", 1.0) if batch else 1.0
        outs = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i:i + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            with torch.no_grad():
                g = self.model.generate(**enc, do_sample=True, temperature=self.temperature, top_p=0.9,
                                        max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                                        repetition_penalty=rep)
            for j in range(len(chunk)):
                full = self.tokenizer.decode(g[j], skip_special_tokens=True).replace("\n", " ")
                outs.append({"input": "", "result": full})
        return outs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activity", action="append", required=True, metavar="LABEL=PARQUET",
                   help="repeatable, e.g. --activity walking=path.parquet --activity cycling=path.parquet")
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id-or-path", default="gpt2")
    p.add_argument("--gen-backend", choices=["vllm", "hf"], default="vllm",
                   help="hf=transformers (use for Qwen; vLLM crashes on Qwen)")
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--train-length", type=int, default=15000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    p.add_argument("--min-per-label", type=int, default=50)
    p.add_argument("--max-per-label", type=int, default=100)
    p.add_argument("--generation-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_windows(parquet, channel, tl, wl, mwn, ts):
    df = pd.read_parquet(parquet)
    scaled, _o, _s = preprocess_train_data(df, train_channels=[channel], train_length=tl, train_samples=1,
                                           augmentation_strategy="univariate", min_windows_length=wl,
                                           min_windows_number=mwn, train_splitting=ts)
    return np.asarray(scaled[0], dtype=np.float64)


def make_prompts(label, cols, tokenizer, batch_size):
    prompts = []
    for _ in range(batch_size):
        c = cols.copy(); random.shuffle(c)
        ti = ", ".join(f"{x} is [blank]" for x in c)
        prompts.append(f"Condition: data is {label} [sep] Input: {ti} [sep] Target:")
    lens = [len(tokenizer(p)["input_ids"]) for p in prompts]
    mx = max(lens); pad = tokenizer.pad_token
    return [pad * (mx - n) + p for p, n in zip(prompts, lens)]


def generate_for_label(llm, original, cols, label, min_per, max_per, bs):
    rows, raw, parsed, attempts = [], 0, 0, 0
    acc = 0
    while acc < min_per and attempts < 12:
        prompts = make_prompts(label, cols, llm.tokenizer, bs)
        outs = llm([{"input": p, "gen_kwargs": {"repetition_penalty": 1.0}} for p in prompts])
        texts = [(o["input"] + o["result"]).replace("\n", " ") for o in outs]
        raw += len(texts)
        g = convert_texts_to_tabular_data(text=texts, original_dataset=original, text_template="fim_template_textual_encoding")
        parsed += int(g.shape[0])
        g = g.replace("NaN", np.nan).dropna(how="all").dropna(subset=cols + ["data"])
        g = g[g["data"].astype(str).str.split("|").str[0].str.strip() == label]
        for c in cols:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        g = g.dropna(subset=cols)
        if not g.empty:
            rows.append(g)
        if rows:
            acc = int(pd.concat(rows, ignore_index=True).round(3).drop_duplicates(subset=cols).shape[0])
        attempts += 1
    if rows:
        combined = pd.concat(rows, ignore_index=True).round(3).drop_duplicates(subset=cols).reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=cols + ["data"])
    combined = combined.iloc[:max_per].copy()
    return combined, {"raw_outputs": raw, "parsed_rows": parsed, "accepted_rows": int(combined.shape[0]), "attempts": attempts}


def write_jsonl(path, channel, label, windows):
    with path.open("w", encoding="utf-8") as h:
        for w in windows:
            h.write(json.dumps({"task_name": "time_series/pamap2_multiactivity_unified",
                                "is_seed": False, "requested_label": label,
                                "generated_time_series": {channel: [float(v) for v in w]}}) + "\n")


def main() -> None:
    a = parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.output_dir.exists():
        shutil.rmtree(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    activities = []  # (label, parquet)
    for spec in a.activity:
        label, _, parq = spec.partition("=")
        activities.append((label.strip(), Path(parq.strip())))

    windows_by_label, labels_col, all_windows = {}, [], []
    for label, parq in activities:
        w = load_windows(parq, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
        windows_by_label[label] = w
        labels_col += [label] * w.shape[0]
        all_windows.append(w)
    combined = np.concatenate(all_windows, axis=0)
    combined_pre = combined[None, :, :]

    embedded, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, a.embedding_dim, a.variance_explained)
    embedded["data"] = labels_col
    cols = [c for c in embedded.columns if c != "data"]
    embedded.to_csv(a.output_dir / "training_embeddings_with_labels.csv", index=False)

    sdforger_params = {"k_bit": None, "embedding_type": "fica", "embedding_dim": a.embedding_dim,
                       "variance_explained": a.variance_explained, "min_windows_number": a.min_windows_number,
                       "min_windows_length": a.window_length, "input_tokens_precision": 4,
                       "text_template": "fim_template_textual_encoding"}
    _dt = torch.float16 if a.dtype == "float16" else torch.float32
    trainer = SDForgerTuningBlock(model_id_or_path=a.model_id_or_path, learning_rate=0.00008,
                                  num_train_epochs=100, per_device_train_batch_size=a.train_batch_size, seed=a.seed)
    tuned = trainer(output_dir=str(a.output_dir / "model"), dataset=embedded,
                    model_args={"dtype": _dt, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                    sdforger_params=sdforger_params, model_id_or_path=a.model_id_or_path)
    if a.gen_backend == "hf":
        llm = HFLLM(tuned, temperature=1.3, max_new_tokens=1000, dtype=torch.float16)
    else:
        llm = get_block("vllm", **{"type": "vllm", MODEL_ID_OR_PATH: tuned, "dtype": "float32",
                                   "trust_remote_code": True, "ignore_mismatched_sizes": True,
                                   "temperature": 1.3, "max_tokens": 1000})

    gstats = {}
    for label, _ in activities:
        g, st = generate_for_label(llm, embedded, cols, label, a.min_per_label, a.max_per_label, a.generation_batch_size)
        gstats[label] = st
        g.to_csv(a.output_dir / f"{label}_generated_embeddings.csv", index=False)
        if not g.empty:
            decoded = fica_transform_to_original_feature_space(g[cols].to_numpy(dtype=np.float64), combined_pre, data_embedded, mixing, mean)[0]
            write_jsonl(a.output_dir / f"{label}_final_data.jsonl", a.channel, label, decoded)

    meta = {"activities": [l for l, _ in activities], "train_windows_by_label": {l: int(w.shape[0]) for l, w in windows_by_label.items()},
            "combined_train_windows": int(combined.shape[0]), "channel": a.channel, "train_length_per_activity": a.train_length,
            "window_length": a.window_length, "embedding_dims": dims, "model_id_or_path": a.model_id_or_path,
            "tuned_model_path": str(tuned), "generation_stats": gstats, "value_space": "standardized_sdforger_window_space"}
    (a.output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
