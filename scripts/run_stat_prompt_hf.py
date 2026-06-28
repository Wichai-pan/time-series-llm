#!/usr/bin/env python3
"""Clean stat-prompt adherence test (HF generation), bins vs real-VALUES encoding.

Conditions generation on a single window statistic (max, the explosion-relevant one),
encoded EITHER as a coarse bin (our T2) OR as the real value to 3 decimals (teammate's
structured_v2). CLEAN protocol: stats from TRAIN windows only (no prompt_split=test
leakage), BALANCED requests (equal count per requested level), per-sample rank adherence
(Spearman of requested level vs realized max) + per-level medians. transformers generation
(vLLM crashes on Qwen). Tests whether a stronger model + real-value encoding produces the
per-sample numeric adherence that gpt2+bins did not (settles the teammate reconciliation).
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
from fms_dgt.public.databuilders.time_series.trainer import SDForgerTuningBlock
from fms_dgt.public.databuilders.time_series.utils import (
    convert_texts_to_tabular_data, fica_embed_data,
    fica_transform_to_original_feature_space, preprocess_train_data)

STAT_CODE = {"max": "mx", "std": "sd", "period": "pr", "range": "rg", "mean": "mn"}


def window_stat(w2d, stat):
    """Per-window pattern descriptor. period = dominant period (samples) via FFT."""
    w2d = np.atleast_2d(np.asarray(w2d, dtype=np.float64))
    if stat == "std":
        return w2d.std(axis=1)
    if stat == "mean":
        return w2d.mean(axis=1)
    if stat == "range":
        return w2d.max(axis=1) - w2d.min(axis=1)
    if stat == "period":
        out = []
        for w in w2d:
            sp = np.abs(np.fft.rfft(w - w.mean())); sp[0] = 0.0
            k = int(np.argmax(sp))
            out.append(float(len(w) / k) if k > 0 else float(len(w)))
        return np.asarray(out)
    return w2d.max(axis=1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--walking-parquet", type=Path, required=True)
    p.add_argument("--running-parquet", type=Path, required=True)
    p.add_argument("--channel", default="hand_acc16_x")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id-or-path", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--encoding", choices=["bins", "values", "percentile", "none"], default="values",
                   help="percentile: encode window-max as its scale-free rank within the activity (multi-subject fix)")
    p.add_argument("--n-levels", type=int, default=4, help="bins: #bins; values: #request quantile levels")
    p.add_argument("--per-level", type=int, default=24, help="balanced requests per (label, level)")
    p.add_argument("--stat", choices=["max", "std", "period", "range", "mean"], default="max",
                   help="which window statistic (pattern descriptor) to condition on")
    p.add_argument("--train-length", type=int, default=5000)
    p.add_argument("--window-length", type=int, default=300)
    p.add_argument("--min-windows-number", type=int, default=30)
    p.add_argument("--train-splitting", default="minimize-overlap")
    p.add_argument("--variance-explained", type=float, default=0.7)
    p.add_argument("--embedding-dim", default="auto")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    p.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf",
                   help="hf=transformers (Qwen); vllm=in-proc vLLM (gpt2 — HF backend gives malformed gpt2 output)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--generation-batch-size", type=int, default=16)
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
    dt = torch.float16 if a.dtype == "float16" else torch.float32

    walking = load_windows(a.walking_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
    running = load_windows(a.running_parquet, a.channel, a.train_length, a.window_length, a.min_windows_number, a.train_splitting)
    combined = np.concatenate([walking, running], axis=0); combined_pre = combined[None, :, :]
    embedded, dims, data_embedded, mixing, mean = fica_embed_data(combined_pre, a.embedding_dim, a.variance_explained)
    cols = list(embedded.columns)
    win_by_label = {"walking": walking, "running": running}
    labels = ["walking"] * walking.shape[0] + ["running"] * running.shape[0]

    # per-window pattern descriptor from TRAIN windows (no leakage); encode into composite data token
    sc = STAT_CODE[a.stat]
    train_stat = {"walking": window_stat(walking, a.stat), "running": window_stat(running, a.stat)}
    edges = {l: np.quantile(train_stat[l], np.linspace(0, 1, a.n_levels + 1)[1:-1]) for l in train_stat}

    def encode(label, val):
        if a.encoding == "none":
            return label, 0.0
        if a.encoding == "bins":
            b = int(np.digitize(val, edges[label]))
            return f"{label}|{sc}b{b}", b
        if a.encoding == "percentile":
            pr = float((train_stat[label] < val).mean())  # scale-free rank within the activity
            return f"{label}|{sc}q={pr:.2f}", pr
        return f"{label}|{sc}={val:.3f}", float(val)

    comp_col, level_col = [], []
    for l in ["walking", "running"]:
        for val in train_stat[l]:
            c, lv = encode(l, val); comp_col.append(c); level_col.append(lv)
    embedded["data"] = comp_col

    sdf = {"k_bit": None, "embedding_type": "fica", "embedding_dim": a.embedding_dim,
           "variance_explained": a.variance_explained, "min_windows_number": a.min_windows_number,
           "min_windows_length": a.window_length, "input_tokens_precision": 4,
           "text_template": "fim_template_textual_encoding"}
    trainer = SDForgerTuningBlock(model_id_or_path=a.model_id_or_path, learning_rate=0.00008,
                                  num_train_epochs=a.epochs, per_device_train_batch_size=a.train_batch_size, seed=a.seed)
    tuned = trainer(output_dir=str(a.output_dir / "model"), dataset=embedded,
                    model_args={"dtype": dt, "trust_remote_code": True, "ignore_mismatched_sizes": True},
                    sdforger_params=sdf, model_id_or_path=a.model_id_or_path)

    if a.gen_backend == "vllm":
        from fms_dgt.base.registry import get_block
        from fms_dgt.core.blocks.llm import MODEL_ID_OR_PATH
        import fms_dgt.core.blocks.llm.vllm.vllm_in_proc  # noqa: F401
        llm = get_block("vllm", **{"type": "vllm", MODEL_ID_OR_PATH: tuned, "dtype": a.dtype,
                                   "trust_remote_code": True, "ignore_mismatched_sizes": True,
                                   "temperature": a.temperature, "max_tokens": a.max_new_tokens})

        def generate(prompts):
            outs = llm([{"input": p, "gen_kwargs": {"repetition_penalty": 1.05}} for p in prompts])
            return [(o["input"] + o["result"]).replace("\n", " ") for o in outs]
    else:
        tok = AutoTokenizer.from_pretrained(tuned, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(tuned, torch_dtype=dt, trust_remote_code=True).to("cuda").eval()

        def generate(prompts):
            outs = []
            for i in range(0, len(prompts), a.generation_batch_size):
                b = prompts[i:i + a.generation_batch_size]
                enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    g = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=0.9,
                                       max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id, repetition_penalty=1.05)
                for j in range(len(b)):
                    outs.append(tok.decode(g[j], skip_special_tokens=True).replace("\n", " "))
            return outs

    # BALANCED requests: per label, request n_levels target levels equally
    rows = []
    decoded_by_label = {"walking": [], "running": []}
    counts = {"raw": 0, "accepted": 0}
    breakdown = {"raw_texts": 0, "parser_rows": 0, "numeric_fail": 0, "label_mismatch": 0, "dedup_drop": 0, "cap_drop": 0}
    dbg = []
    for label in ["walking", "running"]:
        # target levels: none -> just label ; bins -> 0..n-1 ; values -> quantiles of train max
        if a.encoding == "none":
            targets = [(label, 0.0)]; req_per = a.per_level * a.n_levels
        elif a.encoding == "bins":
            targets = [(f"{label}|{sc}b{b}", b) for b in range(a.n_levels)]; req_per = a.per_level
        elif a.encoding == "percentile":
            ps = np.linspace(0.1, 0.9, a.n_levels)
            targets = [(f"{label}|{sc}q={p:.2f}", float(p)) for p in ps]; req_per = a.per_level
        else:
            qs = np.quantile(train_stat[label], np.linspace(0.1, 0.9, a.n_levels))
            targets = [(f"{label}|{sc}={v:.3f}", float(v)) for v in qs]; req_per = a.per_level
        for comp, lv in targets:
            acc = []
            for _ in range((req_per + a.generation_batch_size - 1) // a.generation_batch_size):
                prompts = []
                for _ in range(a.generation_batch_size):
                    c = cols.copy(); random.shuffle(c)
                    ti = ", ".join(f"{x} is [blank]" for x in c)
                    prompts.append(f"Condition: data is {comp} [sep] Input: {ti} [sep] Target:")
                texts = generate(prompts); counts["raw"] += len(texts); breakdown["raw_texts"] += len(texts)
                if len(dbg) < 16: dbg.extend(texts[:4])
                g = convert_texts_to_tabular_data(text=texts, original_dataset=embedded, text_template="fim_template_textual_encoding")
                n0 = len(g); breakdown["parser_rows"] += n0
                g[cols] = g[cols].apply(pd.to_numeric, errors="coerce"); g = g.dropna(subset=cols)
                n1 = len(g); breakdown["numeric_fail"] += (n0 - n1)
                g = g[g["data"].astype(str).str.split("|").str[0].str.strip() == label]
                n2 = len(g); breakdown["label_mismatch"] += (n1 - n2)
                if not g.empty: acc.append(g)
            if not acc: continue
            concat = pd.concat(acc, ignore_index=True)
            deduped = concat.drop_duplicates(subset=cols)
            breakdown["dedup_drop"] += (len(concat) - len(deduped))
            breakdown["cap_drop"] += max(0, len(deduped) - req_per)
            G = deduped.iloc[:req_per]
            counts["accepted"] += len(G)
            dec = fica_transform_to_original_feature_space(G[cols].to_numpy(np.float64), combined_pre, data_embedded, mixing, mean)[0]
            decoded_by_label[label].append(dec)
            for i in range(len(G)):
                rows.append({"label": label, "requested_level": lv, "composite": comp,
                             "realized_max": float(window_stat(dec[i], a.stat)[0]), "realized_std": float(dec[i].std())})

    # save decoded windows per label as final_data.jsonl (for TSG eval)
    for label in ["walking", "running"]:
        path = a.output_dir / f"{label}_final_data.jsonl"
        with path.open("w", encoding="utf-8") as h:
            if decoded_by_label[label]:
                for w in np.concatenate(decoded_by_label[label], axis=0):
                    h.write(json.dumps({"task_name": "ts/stat_prompt_hf", "is_seed": False, "requested_label": label,
                                        "generated_time_series": {a.channel: [float(v) for v in w]}}) + "\n")

    out = pd.DataFrame(rows)
    out.to_csv(a.output_dir / "stat_adherence.csv", index=False)
    (a.output_dir / "debug_raw_generations.txt").write_text("\n----\n".join(dbg), encoding="utf-8")
    # adherence readout
    summary = {}
    for label in ["walking", "running", "ALL"]:
        if len(out) == 0 or "label" not in out.columns:
            break
        d = out if label == "ALL" else out[out.label == label]
        if len(d) > 5 and d.requested_level.nunique() > 1:
            rho = spearmanr(d.requested_level, d.realized_max).correlation
            med = d.groupby("requested_level").realized_max.median().round(3).to_dict()
            summary[label] = {"n": len(d), "spearman_req_vs_realmax": float(rho), "level_to_median_realized": {str(k): v for k, v in med.items()}}
    absmax = np.abs(out.realized_max.values) if len(out) else np.array([0.0])
    (a.output_dir / "run_metadata.json").write_text(json.dumps(
        {"model": a.model_id_or_path, "encoding": a.encoding, "n_levels": a.n_levels, "epochs": a.epochs,
         "temperature": a.temperature, "no_leakage": "stats from train windows only",
         "raw_generated": counts["raw"], "accepted": counts["accepted"],
         "accept_rate": round(counts["accepted"] / max(counts["raw"], 1), 3),
         "frac_absmax_lt4": round(float(np.mean(absmax < 4)), 3), "max_absmax": round(float(absmax.max()), 2),
         "reject_breakdown": breakdown,
         "no_row_texts": breakdown["raw_texts"] - breakdown["parser_rows"],
         "adherence": summary}, indent=2) + "\n")
    print(json.dumps({"breakdown": breakdown, "accept_rate": round(counts["accepted"] / max(counts["raw"], 1), 3)}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
