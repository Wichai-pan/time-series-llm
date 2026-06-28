#!/usr/bin/env python3
"""Smooth repair variants for unified label-conditioned SDForger outputs.

Motivation
----------
The existing post-hoc fix clips each FICA latent dimension to a train-distribution
box (``clip_minmax`` / ``clip_p05_p95`` in ``apply_unified_latent_constraints.py``).
Hard per-dimension clipping rescues value explosions but distorts the latent vector
shape and leaves the decoded rhythm/ACD largely unfixed.

This script adds two smoother repair families that target the advisor feedback
("differencing / linear interpolation, smoother and smaller-amplitude peaks"):

  * ``shrink_*`` (latent space): scale the WHOLE latent row by a single factor so
    the worst-violating dimension lands exactly on its train bound. This preserves
    the latent vector direction (the temporal shape) and only reduces amplitude.

  * ``waveform_interp`` (signal space): decode the raw generated latents, then in
    the decoded window replace samples that fall outside the real per-activity value
    band by linear interpolation between the nearest in-band neighbours. Windows
    whose out-of-band fraction exceeds ``--reject-frac`` are rejected rather than
    fabricated, and clean/repaired/rejected counts are reported.

``clip_p05_p95`` is re-emitted here too so a single run produces one directly
comparable table (clip vs shrink vs waveform-interp) under the SAME FICA basis.

Value space: standardized SDForger window space (same as the canonical TSG table).
"""

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
    parser.add_argument("--generated-dir", type=Path, required=True,
                        help="Dir containing raw {label}_generated_embeddings.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--variance-explained", type=float, default=0.7)
    parser.add_argument("--embedding-dim", default="auto")
    # variant C: real-window value band + per-window reject guard
    parser.add_argument("--band", choices=["minmax", "p01_p99", "p005_p995"], default="minmax",
                        help="Real per-activity value band used by waveform_interp.")
    parser.add_argument("--reject-frac", type=float, default=0.5,
                        help="If a window has more than this fraction of out-of-band "
                             "samples, reject it instead of interpolating.")
    parser.add_argument("--per-activity-bounds", action="store_true",
                        help="Compute clip/shrink latent bounds from each activity's own "
                             "train latents instead of the pooled walking+running latents.")
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
                "task_name": f"time_series/pamap2_smooth_latent_repair_{variant}",
                "is_seed": False,
                "task_description": f"Unified label-conditioned output with smooth repair {variant}.",
                "requested_label": label,
                "constraint_variant": variant,
                "generated_time_series": {channel: [float(v) for v in window]},
            }
            handle.write(json.dumps(record) + "\n")


def decode(embeddings: pd.DataFrame, cols: list[str], combined_preprocessed, data_embedded,
           fica_mixing, fica_mean) -> np.ndarray:
    return fica_transform_to_original_feature_space(
        embeddings[cols].to_numpy(dtype=np.float64),
        combined_preprocessed,
        data_embedded,
        fica_mixing,
        fica_mean,
    )[0]


# --------------------------------------------------------------------------- #
#  Variant A: latent global proportional shrink (shape-preserving)
# --------------------------------------------------------------------------- #
def latent_shrink(gen: pd.DataFrame, lower: pd.Series, upper: pd.Series,
                  cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    out = gen.copy()
    vals = out[cols].to_numpy(dtype=np.float64)
    lo = lower[cols].to_numpy(dtype=np.float64)
    hi = upper[cols].to_numpy(dtype=np.float64)

    # Smallest positive scale s per row so that s * v_d stays within [lo_d, hi_d]
    # for every dimension. v_d > 0 is bounded by hi_d, v_d < 0 is bounded by lo_d.
    ratio = np.full_like(vals, np.inf)
    pos = vals > 0
    neg = vals < 0
    hi_b = np.broadcast_to(hi, vals.shape)
    lo_b = np.broadcast_to(lo, vals.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[pos] = hi_b[pos] / vals[pos]
        ratio[neg] = lo_b[neg] / vals[neg]
    # Only shrink (never amplify in-range rows); guard against negatives.
    s = np.clip(np.nanmin(ratio, axis=1), 0.0, 1.0)
    out[cols] = vals * s[:, None]
    return out, s


# --------------------------------------------------------------------------- #
#  Variant C: waveform-domain linear interpolation repair
# --------------------------------------------------------------------------- #
def real_value_band(real_windows: np.ndarray, band: str) -> tuple[float, float]:
    flat = real_windows.reshape(-1)
    if band == "minmax":
        return float(flat.min()), float(flat.max())
    if band == "p01_p99":
        return float(np.quantile(flat, 0.01)), float(np.quantile(flat, 0.99))
    if band == "p005_p995":
        return float(np.quantile(flat, 0.005)), float(np.quantile(flat, 0.995))
    raise ValueError(band)


def waveform_interp(decoded: np.ndarray, v_lo: float, v_hi: float,
                    reject_frac: float) -> tuple[np.ndarray, int, int, int]:
    repaired: list[np.ndarray] = []
    clean = repaired_count = rejected = 0
    n = decoded.shape[1]
    x = np.arange(n)
    for w in decoded:
        invalid = (w < v_lo) | (w > v_hi)
        frac = float(invalid.mean())
        if frac > reject_frac:
            rejected += 1
            continue
        if invalid.any():
            good = ~invalid
            w2 = w.copy()
            # np.interp clamps to nearest good value at the edges (limit both).
            w2[invalid] = np.interp(x[invalid], x[good], w[good])
            repaired.append(w2)
            repaired_count += 1
        else:
            repaired.append(w.copy())
            clean += 1
    out = np.asarray(repaired, dtype=np.float64) if repaired else np.empty((0, n))
    return out, clean, repaired_count, rejected


def summarize(variant: str, label: str, input_rows: int, windows: np.ndarray,
              **extra) -> dict:
    rec = {"variant": variant, "label": label, "input_rows": int(input_rows),
           "output_rows": int(windows.shape[0])}
    rec.update(extra)
    if windows.shape[0]:
        rec["decoded_abs_max"] = float(np.abs(windows).max())
        rec["decoded_std_mean"] = float(np.std(windows, axis=1).mean())
    else:
        rec["decoded_abs_max"] = None
        rec["decoded_std_mean"] = None
    return rec


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    walking = load_activity_windows(args, args.walking_parquet)
    running = load_activity_windows(args, args.running_parquet)
    real_by_label = {"walking": walking, "running": running}
    combined = np.concatenate([walking, running], axis=0)
    combined_preprocessed = combined[None, :, :]

    train_embeddings, embedding_dims, data_embedded, fica_mixing, fica_mean = fica_embed_data(
        combined_preprocessed, args.embedding_dim, args.variance_explained,
    )
    train_embeddings["data"] = ["walking"] * walking.shape[0] + ["running"] * running.shape[0]
    cols = [c for c in train_embeddings.columns if c != "data"]

    def bounds_for(label: str):
        sub = train_embeddings[train_embeddings["data"] == label] if args.per_activity_bounds else train_embeddings
        return (sub[cols].quantile(0.05), sub[cols].quantile(0.95), sub[cols].min(), sub[cols].max())

    summary: list[dict] = []
    for label in ["walking", "running"]:
        p05, p95, hard_min, hard_max = bounds_for(label)
        gen = pd.read_csv(args.generated_dir / f"{label}_generated_embeddings.csv")
        gen[cols] = gen[cols].apply(pd.to_numeric, errors="coerce")
        parsed = gen.dropna(subset=cols).copy()
        n_in = parsed.shape[0]

        # Reference variant: clip_p05_p95 (per-dim hard clip) under the same basis.
        clip = parsed.copy()
        clip[cols] = clip[cols].clip(lower=p05, upper=p95, axis=1)
        decoded = decode(clip, cols, combined_preprocessed, data_embedded, fica_mixing, fica_mean)
        _emit(args, "clip_p05_p95", label, decoded)
        summary.append(summarize("clip_p05_p95", label, n_in, decoded,
                                  clean_accept_rows=n_in, repaired_rows=0, rejected_rows=0))

        # Variant A: latent global shrink to p05-p95, and to min-max.
        for vname, lo, hi in (("shrink_p05_p95", p05, p95), ("shrink_minmax", hard_min, hard_max)):
            shr, s = latent_shrink(parsed, lo, hi, cols)
            decoded = decode(shr, cols, combined_preprocessed, data_embedded, fica_mixing, fica_mean)
            _emit(args, vname, label, decoded)
            summary.append(summarize(vname, label, n_in, decoded,
                                     clean_accept_rows=int((s >= 1.0).sum()),
                                     repaired_rows=int((s < 1.0).sum()),
                                     rejected_rows=0,
                                     shrink_factor_mean=float(s.mean()),
                                     shrink_factor_min=float(s.min())))

        # Variant C: waveform-domain linear interpolation against the real band.
        v_lo, v_hi = real_value_band(real_by_label[label], args.band)
        raw_decoded = decode(parsed, cols, combined_preprocessed, data_embedded,
                             fica_mixing, fica_mean)
        windows, clean, rep, rej = waveform_interp(raw_decoded, v_lo, v_hi, args.reject_frac)
        _emit(args, "waveform_interp", label, windows)
        summary.append(summarize("waveform_interp", label, n_in, windows,
                                 clean_accept_rows=clean, repaired_rows=rep, rejected_rows=rej,
                                 band=args.band, band_low=v_lo, band_high=v_hi))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output_dir / "smooth_repair_summary.csv", index=False)
    (args.output_dir / "smooth_repair_summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "channel": args.channel,
                    "embedding_dims": embedding_dims,
                    "train_windows": int(combined.shape[0]),
                    "walking_train_windows": int(walking.shape[0]),
                    "running_train_windows": int(running.shape[0]),
                    "generated_dir": str(args.generated_dir),
                    "band": args.band,
                    "reject_frac": args.reject_frac,
                    "per_activity_bounds": bool(args.per_activity_bounds),
                    "value_space": "standardized_sdforger_window_space",
                },
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))


def _emit(args: argparse.Namespace, variant: str, label: str, windows: np.ndarray) -> None:
    variant_dir = args.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    if windows.shape[0] == 0:
        (variant_dir / f"{label}_final_data.jsonl").write_text("", encoding="utf-8")
        return
    write_jsonl(variant_dir / f"{label}_final_data.jsonl", args.channel, label, windows, variant)


if __name__ == "__main__":
    main()
