#!/usr/bin/env python3
"""Unified raw-like evaluation for PAMAP2 SDForger settings.

This script compares old clean SDForger baselines, unified label-conditioned
runs, latent-clipped runs, and normalization ablations under one evaluation
contract: real raw windows vs synthetic windows mapped back to raw-like units.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, welch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


ACTIVITIES = ("walking", "running")
LABEL_ID = {"walking": 0, "running": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walking-parquet", type=Path, required=True)
    parser.add_argument("--running-parquet", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--window-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument("--train-splitting", default="minimize-overlap")
    parser.add_argument("--sampling-rate", type=float, default=100.0)
    parser.add_argument("--max-acf-lag", type=int, default=150)
    parser.add_argument("--skip-dtw", action="store_true")
    return parser.parse_args()


def activity_parquet(args: argparse.Namespace, activity: str) -> Path:
    return args.walking_parquet if activity == "walking" else args.running_parquet


def load_real_train_windows(
    parquet: Path,
    channel: str,
    train_length: int,
    window_length: int,
    min_windows_number: int,
    train_splitting: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet)
    scaled, original, _ = preprocess_train_data(
        df,
        train_channels=[channel],
        train_length=train_length,
        train_samples=1,
        augmentation_strategy="univariate",
        min_windows_length=window_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    scaled_windows = np.asarray(scaled[0], dtype=np.float64)
    raw_windows = np.asarray(original[0], dtype=np.float64)
    mean = raw_windows.mean(axis=0)
    std = raw_windows.std(axis=0)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return scaled_windows, raw_windows, mean, std


def load_real_test_windows(parquet: Path, channel: str, train_length: int, window_length: int) -> np.ndarray:
    df = pd.read_parquet(parquet)
    values = pd.to_numeric(df[channel], errors="raise").to_numpy(dtype=np.float64)
    heldout = values[train_length:]
    n_windows = heldout.size // window_length
    if n_windows <= 0:
        raise ValueError(f"No held-out windows available in {parquet}")
    return heldout[: n_windows * window_length].reshape(n_windows, window_length)


def load_jsonl_windows(path: Path, channel: str) -> np.ndarray:
    windows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            windows.append(np.asarray(record["generated_time_series"][channel], dtype=np.float64))
    if not windows:
        raise ValueError(f"No generated windows found in {path}")
    return np.stack(windows, axis=0)


def inverse_activity_window_standardized(synthetic_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return synthetic_scaled * std + mean


def setting_specs(root: Path, channel: str) -> dict[str, dict[str, object]]:
    return {
        "clean_unconditioned": {
            "kind": "activity_jsonl",
            "conditional": False,
            "dirs": {
                "walking": root / f"pamap2_subject101_walking_{channel}_univariate_baseline",
                "running": root / f"pamap2_subject101_running_{channel}_univariate_baseline",
            },
            "files": {"walking": "final_data.jsonl", "running": "final_data.jsonl"},
        },
        "raw_unified_label_conditioned": {
            "kind": "shared_jsonl",
            "conditional": True,
            "dir": root / f"pamap2_subject101_unified_label_conditioned_{channel}",
            "files": {"walking": "walking_final_data.jsonl", "running": "running_final_data.jsonl"},
        },
        "clip_p05_p95": {
            "kind": "shared_jsonl",
            "conditional": True,
            "dir": root / f"pamap2_subject101_unified_label_conditioned_{channel}_constraints" / "clip_p05_p95",
            "files": {"walking": "walking_final_data.jsonl", "running": "running_final_data.jsonl"},
        },
        "global_series_zscore": {
            "kind": "global_norm_arrays",
            "conditional": True,
            "dir": root / f"pamap2_subject101_norm_ablation_global_series_zscore_{channel}",
        },
    }


def load_setting_raw_like(
    name: str,
    spec: dict[str, object],
    real_cache: dict[str, dict[str, np.ndarray]],
    channel: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    metadata: dict[str, object] = {
        "setting": name,
        "conditional": bool(spec["conditional"]),
        "space": "raw_like",
    }
    real_raw = {activity: real_cache[activity]["raw_train"] for activity in ACTIVITIES}

    if spec["kind"] in {"activity_jsonl", "shared_jsonl"}:
        synthetic_raw: dict[str, np.ndarray] = {}
        for activity in ACTIVITIES:
            if spec["kind"] == "activity_jsonl":
                jsonl_path = spec["dirs"][activity] / spec["files"][activity]  # type: ignore[index]
            else:
                jsonl_path = spec["dir"] / spec["files"][activity]  # type: ignore[index]
            synthetic_scaled = load_jsonl_windows(jsonl_path, channel)
            synthetic_raw[activity] = inverse_activity_window_standardized(
                synthetic_scaled,
                real_cache[activity]["mean"],
                real_cache[activity]["std"],
            )
        metadata["inverse_protocol"] = "activity train-window timestamp mean/std, matching old HAR smoke"
        return real_raw, synthetic_raw, metadata

    if spec["kind"] == "global_norm_arrays":
        run_dir = spec["dir"]
        run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        norm = run_metadata.get("normalization", {})
        mean = float(norm["global_mean"])
        std = float(norm["global_std"])
        real_raw = {}
        synthetic_raw = {}
        for activity in ACTIVITIES:
            real_model = np.load(run_dir / f"{activity}_real_windows_model_space.npy")
            synthetic_model = np.load(run_dir / f"{activity}_synthetic_windows_model_space.npy")
            real_raw[activity] = real_model * std + mean
            synthetic_raw[activity] = synthetic_model * std + mean
        metadata["inverse_protocol"] = "global scalar inverse z-score from run_metadata"
        metadata["global_mean"] = mean
        metadata["global_std"] = std
        return real_raw, synthetic_raw, metadata

    raise ValueError(f"Unsupported setting kind: {spec['kind']}")


def acf_1d(signal: np.ndarray, max_lag: int) -> np.ndarray:
    max_lag = min(max_lag, signal.size - 1)
    centered = signal - np.mean(signal)
    denom = np.dot(centered, centered)
    if np.isclose(denom, 0.0):
        out = np.zeros(max_lag + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 : corr.size // 2 + max_lag + 1]
    counts = np.arange(signal.size, signal.size - max_lag - 1, -1, dtype=np.float64)
    return (corr / counts) / (denom / signal.size)


def acf_matrix(windows: np.ndarray, max_lag: int) -> np.ndarray:
    return np.stack([acf_1d(window, max_lag) for window in windows], axis=0)


def psd_matrix(windows: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    powers = []
    freqs = None
    for window in windows:
        f, p = welch(window - np.mean(window), fs=sampling_rate, nperseg=min(window.size, 300), scaling="density")
        freqs = f
        powers.append(p)
    assert freqs is not None
    return freqs, np.stack(powers, axis=0)


def dominant_peak(values: np.ndarray, x_values: np.ndarray, min_index: int = 1) -> tuple[float, float]:
    candidates, _ = find_peaks(values[min_index:])
    candidates = candidates + min_index
    if candidates.size == 0:
        idx = int(np.argmax(values[min_index:]) + min_index)
    else:
        idx = int(candidates[np.argmax(values[candidates])])
    return float(x_values[idx]), float(values[idx])


def rhythm_summary(
    setting: str,
    activity: str,
    real: np.ndarray,
    synthetic: np.ndarray,
    sampling_rate: float,
    max_acf_lag: int,
) -> dict[str, object]:
    real_acf = acf_matrix(real, max_acf_lag)
    synthetic_acf = acf_matrix(synthetic, max_acf_lag)
    lags = np.arange(real_acf.shape[1])
    real_lag, real_acf_score = dominant_peak(real_acf.mean(axis=0), lags, min_index=2)
    synthetic_lag, synthetic_acf_score = dominant_peak(synthetic_acf.mean(axis=0), lags, min_index=2)

    freqs, real_psd = psd_matrix(real, sampling_rate)
    _freqs2, synthetic_psd = psd_matrix(synthetic, sampling_rate)
    freq_mask = freqs > 0
    real_hz, real_power = dominant_peak(real_psd.mean(axis=0)[freq_mask], freqs[freq_mask], min_index=0)
    synthetic_hz, synthetic_power = dominant_peak(
        synthetic_psd.mean(axis=0)[freq_mask],
        freqs[freq_mask],
        min_index=0,
    )
    real_abs_max = float(np.max(np.abs(real)))
    synthetic_abs_max = float(np.max(np.abs(synthetic)))
    return {
        "setting": setting,
        "activity": activity,
        "real_windows": int(real.shape[0]),
        "synthetic_windows": int(synthetic.shape[0]),
        "real_abs_max": real_abs_max,
        "synthetic_abs_max": synthetic_abs_max,
        "real_std_mean": float(np.std(real, axis=1).mean()),
        "synthetic_std_mean": float(np.std(synthetic, axis=1).mean()),
        "amplitude_ratio": synthetic_abs_max / max(real_abs_max, 1e-12),
        "real_acf_lag": real_lag,
        "synthetic_acf_lag": synthetic_lag,
        "acf_lag_diff": abs(real_lag - synthetic_lag),
        "real_acf_score": real_acf_score,
        "synthetic_acf_score": synthetic_acf_score,
        "real_psd_hz": real_hz,
        "synthetic_psd_hz": synthetic_hz,
        "psd_hz_diff": abs(real_hz - synthetic_hz),
        "real_psd_peak_power": real_power,
        "synthetic_psd_peak_power": synthetic_power,
    }


def histogram_torch(x: torch.Tensor, n_bins: int, density: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = x.min().item(), x.max().item()
    b = b + 1e-5 if b == a else b
    bins = torch.linspace(a, b, n_bins + 1, device=x.device, dtype=x.dtype)
    delta = bins[1] - bins[0]
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class Loss(nn.Module):
    def __init__(self, name: str, reg: float = 1.0, transform=lambda x: x, norm_foo=lambda x: x):
        super().__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.norm_foo = norm_foo

    def forward(self, x_fake: torch.Tensor) -> torch.Tensor:
        return self.reg * self.compute(x_fake).mean()

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HistoLoss(Loss):
    def __init__(self, x_real: torch.Tensor, n_bins: int, **kwargs):
        super().__init__(**kwargs)
        self.densities = []
        self.locs = []
        self.deltas = []
        for i in range(x_real.shape[2]):
            tmp_densities = []
            tmp_locs = []
            tmp_deltas = []
            for t in range(x_real.shape[1]):
                density, bins = histogram_torch(x_real[:, t, i].reshape(-1, 1), n_bins, density=True)
                tmp_densities.append(nn.Parameter(density).to(x_real.device))
                delta = bins[1:2] - bins[:1]
                loc = 0.5 * (bins[1:] + bins[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        losses = []
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = ((self.deltas[i][t].to(x_fake.device) / 2.0 - dist) > 0.0).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                losses.append(torch.mean(torch.abs(density - self.densities[i][t].to(x_fake.device)), 0))
        return torch.stack(losses)


def acf_torch(x: torch.Tensor, max_lag: int, dim: tuple[int, int] = (0, 1)) -> torch.Tensor:
    acf_list = []
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    std = torch.where(torch.isclose(std, torch.zeros_like(std)), torch.ones_like(std), std)
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    return torch.cat(acf_list, 1)


class ACFLoss(Loss):
    def __init__(self, x_real: torch.Tensor, max_lag: int = 64, **kwargs):
        super().__init__(norm_foo=lambda x: torch.sqrt(torch.pow(x, 2).sum(0)), **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.acf_real = acf_torch(self.transform(x_real), self.max_lag, dim=(0, 1))

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        return self.norm_foo(acf_torch(self.transform(x_fake), self.max_lag) - self.acf_real.to(x_fake.device))


def skew_torch(x: torch.Tensor, dim: tuple[int, int] = (0, 1), dropdims: bool = True) -> torch.Tensor:
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    x_std_3 = torch.where(torch.isclose(x_std_3, torch.zeros_like(x_std_3)), torch.ones_like(x_std_3), x_std_3)
    skew = x_3 / x_std_3
    return skew[0, 0] if dropdims else skew


def kurtosis_torch(x: torch.Tensor, dim: tuple[int, int] = (0, 1), dropdims: bool = True) -> torch.Tensor:
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    x_var2 = torch.where(torch.isclose(x_var2, torch.zeros_like(x_var2)), torch.ones_like(x_var2), x_var2)
    kurtosis = x_4 / x_var2 - 3
    return kurtosis[0, 0] if dropdims else kurtosis


def to_tsg_shape(windows: np.ndarray) -> np.ndarray:
    return windows[:, :, None].astype(np.float32)


def calculate_tsgbench_style(real: np.ndarray, synthetic: np.ndarray, skip_dtw: bool) -> dict[str, float | None]:
    n = min(real.shape[0], synthetic.shape[0])
    real = to_tsg_shape(real[:n])
    synthetic = to_tsg_shape(synthetic[:n])
    real_t = torch.as_tensor(real, dtype=torch.float32)
    synthetic_t = torch.as_tensor(synthetic, dtype=torch.float32)
    out: dict[str, float | None] = {
        "MDD": float(HistoLoss(real_t[:, 1:, :], n_bins=50, name="marginal_distribution")(synthetic_t[:, 1:, :]).item()),
        "ACD": float(ACFLoss(real_t, name="auto_correlation")(synthetic_t).item()),
        "SD": float(torch.abs(skew_torch(synthetic_t) - skew_torch(real_t)).mean().item()),
        "KD": float(torch.abs(kurtosis_torch(synthetic_t) - kurtosis_torch(real_t)).mean().item()),
        "ED": float(np.mean(np.linalg.norm(real[:, :, 0] - synthetic[:, :, 0], axis=1))),
    }
    out["DTW"] = None if skip_dtw else calculate_dtw(real, synthetic)
    return out


def dtw_distance_fallback(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    cost = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost[i, j] = np.linalg.norm(x[i - 1] - y[j - 1]) + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )
    return float(cost[len(x), len(y)])


def calculate_dtw(real: np.ndarray, synthetic: np.ndarray) -> float:
    try:
        from dtaidistance.dtw_ndim import distance as multi_dtw_distance
    except ImportError:
        multi_dtw_distance = None

    distances = []
    for i in range(real.shape[0]):
        if multi_dtw_distance is not None:
            distance = multi_dtw_distance(real[i].astype(np.double), synthetic[i].astype(np.double), use_c=True)
        else:
            distance = dtw_distance_fallback(real[i], synthetic[i])
        distances.append(distance)
    return float(np.mean(np.asarray(distances)))


def flat_xy(walking: np.ndarray, running: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([walking, running], axis=0)
    y = np.concatenate(
        [
            np.zeros(walking.shape[0], dtype=np.int64),
            np.ones(running.shape[0], dtype=np.int64),
        ]
    )
    return x.reshape(x.shape[0], -1), y


def evaluate_classifier(name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, dummy: bool = False) -> dict[str, object]:
    if dummy:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "condition": name,
        "train_samples": int(x_train.shape[0]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }


def har_utility(
    setting: str,
    real_train: dict[str, np.ndarray],
    synthetic: dict[str, np.ndarray],
    real_test: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    x_test, y_test = flat_xy(real_test["walking"], real_test["running"])
    x_real, y_real = flat_xy(real_train["walking"], real_train["running"])
    x_syn, y_syn = flat_xy(synthetic["walking"], synthetic["running"])
    rows = [
        evaluate_classifier("majority", x_real, y_real, x_test, y_test, dummy=True),
        evaluate_classifier("real-only", x_real, y_real, x_test, y_test),
        evaluate_classifier("synthetic-only-all", x_syn, y_syn, x_test, y_test),
        evaluate_classifier(
            "real+synthetic-all",
            np.concatenate([x_real, x_syn], axis=0),
            np.concatenate([y_real, y_syn], axis=0),
            x_test,
            y_test,
        ),
    ]
    for row in rows:
        row["setting"] = setting
    return rows


def label_controllability(setting: str, real_train: dict[str, np.ndarray], synthetic: dict[str, np.ndarray]) -> dict[str, object]:
    x_real, y_real = flat_xy(real_train["walking"], real_train["running"])
    x_syn, y_requested = flat_xy(synthetic["walking"], synthetic["running"])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_real, y_real)
    pred = clf.predict(x_syn)
    return {
        "setting": setting,
        "label_accuracy": float(accuracy_score(y_requested, pred)),
        "label_balanced_accuracy": float(balanced_accuracy_score(y_requested, pred)),
        "label_macro_f1": float(f1_score(y_requested, pred, average="macro")),
        "walking_requested_accuracy": float((pred[y_requested == LABEL_ID["walking"]] == LABEL_ID["walking"]).mean()),
        "running_requested_accuracy": float((pred[y_requested == LABEL_ID["running"]] == LABEL_ID["running"]).mean()),
        "label_confusion_matrix": confusion_matrix(y_requested, pred, labels=[0, 1]).tolist(),
    }


def choose_representative(real: np.ndarray, synthetic: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    real_mean = real.mean(axis=0)
    real_idx = int(np.argmin(np.linalg.norm(real - real_mean[None, :], axis=1)))
    synthetic_idx = int(np.argmin(np.linalg.norm(synthetic - real_mean[None, :], axis=1)))
    return real[real_idx], synthetic[synthetic_idx], real_idx, synthetic_idx


def best_lag_align(reference: np.ndarray, candidate: np.ndarray, max_shift: int = 80) -> tuple[np.ndarray, int]:
    best_score = -math.inf
    best_shift = 0
    best_candidate = candidate
    ref = reference - reference.mean()
    for shift in range(-max_shift, max_shift + 1):
        shifted = np.roll(candidate, shift)
        cand = shifted - shifted.mean()
        denom = np.linalg.norm(ref) * np.linalg.norm(cand)
        score = -math.inf if np.isclose(denom, 0.0) else float(np.dot(ref, cand) / denom)
        if score > best_score:
            best_score = score
            best_shift = shift
            best_candidate = shifted
    return best_candidate, best_shift


def plot_overlay(
    output_dir: Path,
    activity: str,
    real_train_by_setting: dict[str, np.ndarray],
    synthetic_by_setting: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    reference_setting = "clean_unconditioned"
    real, _synthetic, real_idx, _synthetic_idx = choose_representative(
        real_train_by_setting[reference_setting],
        synthetic_by_setting[reference_setting],
    )
    ax.plot(real, color="#0072B2", linewidth=2.3, label=f"real representative #{real_idx}")
    colors = {
        "clean_unconditioned": "#999999",
        "raw_unified_label_conditioned": "#D55E00",
        "clip_p05_p95": "#009E73",
        "global_series_zscore": "#CC79A7",
    }
    labels = {
        "clean_unconditioned": "clean unconditioned",
        "raw_unified_label_conditioned": "raw unified label",
        "clip_p05_p95": "clip p05-p95",
        "global_series_zscore": "global z-score",
    }
    for setting, synthetic in synthetic_by_setting.items():
        _real_ref, synthetic_rep, _rid, synthetic_idx = choose_representative(real_train_by_setting[setting], synthetic)
        aligned, shift = best_lag_align(real, synthetic_rep)
        ax.plot(
            aligned,
            color=colors.get(setting),
            linewidth=1.6,
            alpha=0.88,
            label=f"{labels.get(setting, setting)} #{synthetic_idx}, shift={shift}",
        )
    ax.set_title(f"{activity}: representative real vs synthetic settings (raw-like, lag-aligned)")
    ax.set_xlabel("time index")
    ax.set_ylabel("hand_acc16_x raw-like value")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{activity}_setting_overlay_raw_like.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    value_df: pd.DataFrame,
    tsg_df: pd.DataFrame,
    har_df: pd.DataFrame,
    label_df: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    lines = [
        "# Unified Raw-like Evaluation",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key, value in metadata.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Value / rhythm metrics",
        "",
        value_df.to_markdown(index=False),
        "",
        "## TSGBench-style metrics",
        "",
        tsg_df.to_markdown(index=False),
        "",
        "## HAR utility",
        "",
        har_df.to_markdown(index=False),
        "",
        "## Label controllability",
        "",
        label_df.to_markdown(index=False),
        "",
        "Lower is better for MDD/ACD/SD/KD/ED/DTW, ACF lag diff, PSD Hz diff, and amplitude ratio distance from 1. Higher is better for HAR and label accuracies.",
    ]
    (output_dir / "unified_raw_like_evaluation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_cache = {}
    real_test = {}
    for activity in ACTIVITIES:
        scaled, raw_train, mean, std = load_real_train_windows(
            activity_parquet(args, activity),
            args.channel,
            args.train_length,
            args.window_length,
            args.min_windows_number,
            args.train_splitting,
        )
        real_cache[activity] = {
            "scaled_train": scaled,
            "raw_train": raw_train,
            "mean": mean,
            "std": std,
        }
        real_test[activity] = load_real_test_windows(
            activity_parquet(args, activity),
            args.channel,
            args.train_length,
            args.window_length,
        )

    value_rows: list[dict[str, object]] = []
    tsg_rows: list[dict[str, object]] = []
    har_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    all_real_train: dict[str, dict[str, np.ndarray]] = {}
    all_synthetic: dict[str, dict[str, np.ndarray]] = {}
    setting_metadata = []

    for setting, spec in setting_specs(args.remote_output_root, args.channel).items():
        real_train, synthetic, meta = load_setting_raw_like(setting, spec, real_cache, args.channel)
        setting_metadata.append(meta)
        all_real_train[setting] = real_train
        all_synthetic[setting] = synthetic

        for activity in ACTIVITIES:
            value_rows.append(
                rhythm_summary(
                    setting,
                    activity,
                    real_train[activity],
                    synthetic[activity],
                    args.sampling_rate,
                    args.max_acf_lag,
                )
            )
            tsg = calculate_tsgbench_style(real_train[activity], synthetic[activity], args.skip_dtw)
            tsg_rows.append({"setting": setting, "activity": activity, **tsg})

        har_rows.extend(har_utility(setting, real_train, synthetic, real_test))
        if bool(spec["conditional"]):
            label_rows.append(label_controllability(setting, real_train, synthetic))
        else:
            label_rows.append({"setting": setting, "label_accuracy": None, "note": "not label-conditioned"})

    for activity in ACTIVITIES:
        plot_overlay(
            args.output_dir / "figures",
            activity,
            {setting: all_real_train[setting][activity] for setting in all_real_train},
            {setting: all_synthetic[setting][activity] for setting in all_synthetic},
        )

    value_df = pd.DataFrame(value_rows)
    tsg_df = pd.DataFrame(tsg_rows)
    har_df = pd.DataFrame(har_rows)
    label_df = pd.DataFrame(label_rows)
    value_df.to_csv(args.output_dir / "value_rhythm_raw_like.csv", index=False)
    tsg_df.to_csv(args.output_dir / "tsgbench_style_raw_like.csv", index=False)
    har_df.to_csv(args.output_dir / "har_utility_raw_like.csv", index=False)
    label_df.to_csv(args.output_dir / "label_controllability_raw_like.csv", index=False)
    (args.output_dir / "setting_metadata.json").write_text(json.dumps(setting_metadata, indent=2) + "\n", encoding="utf-8")
    write_report(
        args.output_dir,
        value_df,
        tsg_df,
        har_df,
        label_df,
        {
            "channel": args.channel,
            "train_length": args.train_length,
            "window_length": args.window_length,
            "space": "raw_like",
            "settings": ", ".join(setting_specs(args.remote_output_root, args.channel).keys()),
            "skip_dtw": args.skip_dtw,
        },
    )
    print((args.output_dir / "unified_raw_like_evaluation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
