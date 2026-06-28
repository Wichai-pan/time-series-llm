#!/usr/bin/env python3
"""Evaluate one SDForger run with benchmark-aligned similarity metrics.

This script replaces the earlier approximate implementation with the exact
TSGBench formulas for:
MDD, ACD, SD, KD, ED, DTW.

Important note:
- This script now includes the supplementary SIDL-based shapelet
  reconstruction path used for SHAP-RE / SHR. It runs on the same standardized
  SDForger window space as the other metrics.
- SDForger may generate more samples than the number of real training windows.
  Feature-based metrics natively support different sample counts. Distance-based
  metrics require paired shapes, so we down-sample synthetic windows to the
  real window count with a deterministic rule and record that protocol.
- fms-dgt writes generated time series in the same standardized space used for
  embedding/reconstruction. The default `--synthetic-space=scaled` therefore
  compares generated windows directly against the standardized real windows.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from dtaidistance.dtw_ndim import distance as multi_dtw_distance
except ImportError:
    multi_dtw_distance = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fms_dgt.public.databuilders.time_series.utils import preprocess_train_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-parquet", type=Path, required=True)
    parser.add_argument("--synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--train-length", type=int, default=5000)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument(
        "--augmentation-strategy",
        default="univariate",
        choices=["univariate", "multivariate", "multisample"],
    )
    parser.add_argument("--min-windows-length", type=int, default=300)
    parser.add_argument("--min-windows-number", type=int, default=30)
    parser.add_argument(
        "--train-splitting",
        default="minimize-overlap",
        choices=["maximize-overlap", "minimize-overlap", "no-periodicity"],
    )
    parser.add_argument(
        "--subset-mode",
        default="first",
        choices=["first", "random"],
        help="How to down-sample synthetic windows when ED/DTW need equal sample counts.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Seed used when --subset-mode=random.",
    )
    parser.add_argument(
        "--synthetic-space",
        default="scaled",
        choices=["scaled", "original"],
        help=(
            "`scaled` means generated JSONL values are already in SDForger's "
            "standardized window space. `original` standardizes them from the "
            "raw real-window mean/std before evaluation."
        ),
    )
    parser.add_argument(
        "--max-synthetic-samples",
        type=int,
        default=None,
        help=(
            "Optional hard cap applied before all metrics. Use this when the "
            "generator overshoots the intended paper protocol sample count."
        ),
    )
    parser.add_argument(
        "--compute-shr",
        action="store_true",
        help="Compute SIDL-based shapelet reconstruction error (paper SHAP-RE / SHR).",
    )
    parser.add_argument(
        "--shr-seed",
        type=int,
        default=42,
        help="Random seed used inside the SIDL-based SHAP-RE / SHR computation.",
    )
    return parser.parse_args()


def load_synthetic_windows(path: Path, channel: str) -> np.ndarray:
    windows = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            generated = record.get("generated_time_series", {})
            if channel not in generated:
                raise KeyError(f"{channel} is missing in generated_time_series.")
            windows.append(np.asarray(generated[channel], dtype=np.float64))
    if not windows:
        raise ValueError(f"No synthetic windows found in {path}")
    lengths = {window.shape[0] for window in windows}
    if len(lengths) != 1:
        raise ValueError(f"Synthetic windows have inconsistent lengths: {sorted(lengths)}")
    return np.stack(windows, axis=0)


def cap_synthetic_windows(synth_windows: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None:
        return synth_windows
    if max_samples <= 0:
        raise ValueError("--max-synthetic-samples must be positive when provided.")
    if synth_windows.shape[0] < max_samples:
        raise ValueError(
            f"Requested --max-synthetic-samples={max_samples}, "
            f"but only found {synth_windows.shape[0]} synthetic windows."
        )
    return synth_windows[:max_samples]


def build_real_training_windows(
    real_parquet: Path,
    channel: str,
    train_length: int,
    train_samples: int,
    augmentation_strategy: str,
    min_windows_length: int,
    min_windows_number: int,
    train_splitting: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(real_parquet)
    if channel not in df.columns:
        raise KeyError(f"{channel} is missing from {real_parquet}")

    scaled, original, _ = preprocess_train_data(
        df,
        train_channels=[channel],
        train_length=train_length,
        train_samples=train_samples,
        augmentation_strategy=augmentation_strategy,
        min_windows_length=min_windows_length,
        min_windows_number=min_windows_number,
        train_splitting=train_splitting,
    )
    return scaled[0], np.asarray(original[0], dtype=np.float64)


def standardize_original_synthetic_from_real(
    real_original: np.ndarray,
    synth_original: np.ndarray,
) -> np.ndarray:
    mean = np.mean(real_original, axis=0)
    std = np.std(real_original, axis=0)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return (synth_original - mean) / std


def choose_synthetic_subset(
    synth_windows: np.ndarray,
    target_count: int,
    subset_mode: str,
    subset_seed: int,
) -> np.ndarray:
    if synth_windows.shape[0] < target_count:
        raise ValueError(
            f"Need at least {target_count} synthetic windows, found {synth_windows.shape[0]}"
        )
    if target_count == synth_windows.shape[0]:
        return synth_windows
    if subset_mode == "first":
        return synth_windows[:target_count]
    rng = random.Random(subset_seed)
    indices = sorted(rng.sample(range(synth_windows.shape[0]), target_count))
    return synth_windows[indices]


def op_shift(shapelets: np.ndarray, offsets: np.ndarray, target_dim: int) -> np.ndarray:
    basis_count, basis_length = shapelets.shape
    result = np.zeros((basis_count, target_dim), dtype=np.float64)
    for idx in range(basis_count):
        offset = int(offsets[idx])
        if offset + basis_length > target_dim:
            raise ValueError(
                f"Offset {offset} with basis length {basis_length} exceeds target_dim {target_dim}"
            )
        result[idx, offset : offset + basis_length] = shapelets[idx]
    return result


def unsup_obj_sid(
    data: np.ndarray,
    shapelets: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    lambda_: float,
) -> float:
    n_samples, series_length = data.shape
    _basis_count, _basis_length = shapelets.shape
    objective = 0.0
    for sample_idx in range(n_samples):
        shifted_shapelets = op_shift(shapelets, offsets[sample_idx], series_length)
        reconstruction = np.dot(coefficients[sample_idx], shifted_shapelets)
        objective += (
            0.5 * np.linalg.norm(data[sample_idx] - reconstruction) ** 2
            + lambda_ * np.linalg.norm(coefficients[sample_idx], 1)
        )
    return float(objective)


def update_a_par_sid(
    data: np.ndarray,
    shapelets: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    lambda_: float,
    max_iter: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_samples, series_length = data.shape
    basis_count, basis_length = shapelets.shape
    segment_indices = np.add.outer(np.arange(series_length - basis_length + 1), np.arange(basis_length))

    objective_history: list[float] = []
    for _ in range(int(max_iter)):
        for sample_idx in range(n_samples):
            sample = data[sample_idx]
            sample_offsets = offsets[sample_idx]
            shifted_shapelets = op_shift(shapelets, sample_offsets, series_length)

            for basis_idx in np.random.permutation(basis_count):
                basis = shapelets[basis_idx]
                temp_coefficients = coefficients[sample_idx].copy()
                temp_coefficients[basis_idx] = 0.0

                residue = sample - np.dot(temp_coefficients, shifted_shapelets)
                basis_norm2 = np.linalg.norm(basis) ** 2

                segments = residue[segment_indices]
                dot_products = np.dot(segments, basis)

                best_segment_idx = int(np.argmax(np.abs(dot_products)))
                best_dot_product = dot_products[best_segment_idx]

                if np.abs(best_dot_product) <= lambda_:
                    coefficient_star = 0.0
                else:
                    coefficient_star = (
                        np.sign(best_dot_product) * (np.abs(best_dot_product) - lambda_) / basis_norm2
                    )
                    offset_star = best_segment_idx

                coefficients[sample_idx, basis_idx] = coefficient_star
                if coefficient_star != 0.0:
                    shifted_shapelets[basis_idx, :] = 0.0
                    shifted_shapelets[basis_idx, offset_star : offset_star + basis_length] = basis
                    offsets[sample_idx, basis_idx] = offset_star

        objective = unsup_obj_sid(data, shapelets, coefficients, offsets, lambda_)
        objective_history.append(objective)
        if len(objective_history) > 1:
            previous = objective_history[-2]
            if previous != 0 and abs(objective_history[-1] - previous) / previous < epsilon:
                return coefficients, offsets, objective

    return coefficients, offsets, float(objective_history[-1])


def update_s_sid(
    data: np.ndarray,
    shapelets: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    lambda_: float,
    c: float,
    max_iter: int,
    epsilon: float,
) -> np.ndarray:
    n_samples, series_length = data.shape
    basis_count, basis_length = shapelets.shape
    objective_history: list[float] = []

    for _ in range(int(max_iter)):
        for basis_idx in range(basis_count):
            activation_norm2 = np.linalg.norm(coefficients[:, basis_idx]) ** 2
            if activation_norm2 == 0:
                continue

            new_basis = np.zeros(basis_length, dtype=np.float64)
            for sample_idx in range(n_samples):
                temp_coefficients = coefficients[sample_idx].copy()
                temp_coefficients[basis_idx] = 0.0
                shifted_shapelets = op_shift(shapelets, offsets[sample_idx], series_length)
                residue = data[sample_idx] - np.dot(temp_coefficients, shifted_shapelets)

                offset = int(offsets[sample_idx, basis_idx])
                new_basis += coefficients[sample_idx, basis_idx] * residue[offset : offset + basis_length]

            if activation_norm2 <= np.linalg.norm(new_basis) / np.sqrt(c):
                new_basis = np.sqrt(c) / np.linalg.norm(new_basis) * new_basis
            else:
                new_basis = new_basis / activation_norm2

            shapelets[basis_idx, :] = new_basis

        objective = unsup_obj_sid(data, shapelets, coefficients, offsets, lambda_)
        objective_history.append(objective)
        if len(objective_history) > 1:
            previous = objective_history[-2]
            if previous != 0 and abs(objective_history[-1] - previous) / previous < epsilon:
                return shapelets

    return shapelets


def usidl(
    data: np.ndarray,
    lambda_: float,
    basis_count: int,
    basis_length: int,
    c: float,
    epsilon: float,
    max_iter: int,
    max_inner_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, series_length = data.shape
    shapelets = np.random.randn(basis_count, basis_length)
    coefficients = np.random.randn(n_samples, basis_count)
    offsets = np.random.randint(0, series_length - basis_length + 1, (n_samples, basis_count))

    objective_history: list[float] = []
    for _ in range(int(max_iter)):
        coefficients, offsets, _objective = update_a_par_sid(
            data, shapelets, coefficients, offsets, lambda_, max_inner_iter, epsilon
        )
        shapelets = update_s_sid(data, shapelets, coefficients, offsets, lambda_, c, max_inner_iter, epsilon)
        objective = unsup_obj_sid(data, shapelets, coefficients, offsets, lambda_)
        objective_history.append(objective)
        if len(objective_history) > 1:
            previous = objective_history[-2]
            if previous != 0 and abs(objective_history[-1] - previous) / previous < epsilon:
                break

    return shapelets, coefficients, offsets


def calculate_shr_sid(
    ori_data: np.ndarray,
    gen_data: np.ndarray,
    seed: int,
    basis_count: int = 20,
    lambda_: float = 0.1,
    basis_ratio: float = 0.25,
    c: float = 100.0,
    epsilon: float = 1e-5,
    max_iter: int = 1000,
    max_inner_iter: int = 5,
) -> float:
    np.random.seed(seed)

    train_data = np.asarray(ori_data, dtype=np.float64).reshape(ori_data.shape[0], ori_data.shape[1])
    test_data = np.asarray(gen_data, dtype=np.float64).reshape(gen_data.shape[0], gen_data.shape[1])
    _n_train, series_length = train_data.shape
    n_test, _ = test_data.shape

    basis_length = int(np.ceil(series_length * basis_ratio))
    shapelets, _coefficients, _offsets = usidl(
        train_data, lambda_, basis_count, basis_length, c, epsilon, max_iter, max_inner_iter
    )

    test_coefficients = np.random.randn(n_test, basis_count)
    test_offsets = np.random.randint(0, series_length - basis_length + 1, (n_test, basis_count))
    test_coefficients, test_offsets, _ = update_a_par_sid(
        test_data, shapelets, test_coefficients, test_offsets, lambda_, max_iter, epsilon
    )
    reconstruction_error = unsup_obj_sid(test_data, shapelets, test_coefficients, test_offsets, 0.0) / n_test
    return float(reconstruction_error)


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
    def __init__(
        self,
        name: str,
        reg: float = 1.0,
        transform=lambda x: x,
        threshold: float = 10.0,
        backward: bool = False,
        norm_foo=lambda x: x,
    ):
        super().__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake: torch.Tensor) -> torch.Tensor:
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

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
                x_ti = x_real[:, t, i].reshape(-1, 1)
                density, bins = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(density).to(x_real.device))
                delta = bins[1:2] - bins[:1]
                loc = 0.5 * (bins[1:] + bins[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        loss = []

        def relu(x: torch.Tensor) -> torch.Tensor:
            return x * (x >= 0.0).float()

        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(x_fake.device) / 2.0 - dist) > 0.0).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        return torch.stack(loss)


def acf_torch(x: torch.Tensor, max_lag: int, dim: tuple[int, int] = (0, 1)) -> torch.Tensor:
    acf_list = []
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    return torch.cat(acf_list, 1)


def acf_diff(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.pow(x, 2).sum(0))


class ACFLoss(Loss):
    def __init__(self, x_real: torch.Tensor, max_lag: int = 64, **kwargs):
        super().__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.acf_real = acf_torch(self.transform(x_real), self.max_lag, dim=(0, 1))

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


def skew_torch(x: torch.Tensor, dim: tuple[int, int] = (0, 1), dropdims: bool = True) -> torch.Tensor:
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


class SkewnessLoss(Loss):
    def __init__(self, x_real: torch.Tensor, **kwargs):
        super().__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(x_real)

    def compute(self, x_fake: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.norm_foo(skew_torch(x_fake) - self.skew_real)


def kurtosis_torch(
    x: torch.Tensor, dim: tuple[int, int] = (0, 1), excess: bool = True, dropdims: bool = True
) -> torch.Tensor:
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


class KurtosisLoss(Loss):
    def __init__(self, x_real: torch.Tensor, **kwargs):
        super().__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(x_real)

    def compute(self, x_fake: torch.Tensor) -> torch.Tensor:
        return self.norm_foo(kurtosis_torch(x_fake) - self.kurtosis_real)


def calculate_mdd(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    ori = torch.as_tensor(ori_data, dtype=torch.float32)
    gen = torch.as_tensor(gen_data, dtype=torch.float32)
    score = HistoLoss(ori[:, 1:, :], n_bins=50, name="marginal_distribution")(gen[:, 1:, :])
    return float(score.detach().cpu().numpy().item())


def calculate_acd(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    ori = torch.as_tensor(ori_data, dtype=torch.float32)
    gen = torch.as_tensor(gen_data, dtype=torch.float32)
    score = ACFLoss(ori, name="auto_correlation")(gen)
    return float(score.detach().cpu().numpy().item())


def calculate_sd(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    ori = torch.as_tensor(ori_data, dtype=torch.float32)
    gen = torch.as_tensor(gen_data, dtype=torch.float32)
    score = SkewnessLoss(x_real=ori, name="skew").compute(gen).mean()
    return float(score.detach().cpu().numpy())


def calculate_kd(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    ori = torch.as_tensor(ori_data, dtype=torch.float32)
    gen = torch.as_tensor(gen_data, dtype=torch.float32)
    score = KurtosisLoss(x_real=ori, name="kurtosis").compute(gen).mean()
    return float(score.detach().cpu().numpy())


def calculate_ed(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    n_samples = ori_data.shape[0]
    n_series = ori_data.shape[2]
    distance_eu = []
    for i in range(n_samples):
        total_distance_eu = 0.0
        for j in range(n_series):
            total_distance_eu += np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
        distance_eu.append(total_distance_eu / n_series)
    return float(np.mean(np.asarray(distance_eu)))


def calculate_dtw(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    distance_dtw = []
    for i in range(ori_data.shape[0]):
        if multi_dtw_distance is not None:
            distance = multi_dtw_distance(
                ori_data[i].astype(np.double),
                gen_data[i].astype(np.double),
                use_c=True,
            )
        else:
            distance = dtw_distance_fallback(ori_data[i], gen_data[i])
        distance_dtw.append(distance)
    return float(np.mean(np.asarray(distance_dtw)))


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
            dist = np.linalg.norm(x[i - 1] - y[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[len(x), len(y)])


def save_overlay_plot(
    real_windows: np.ndarray,
    synth_windows: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_count = min(10, real_windows.shape[0], synth_windows.shape[0])
    plt.figure(figsize=(10, 4))
    for idx in range(pair_count):
        plt.plot(real_windows[idx], color="#23b5d3", alpha=0.45, linestyle="--", linewidth=1)
        plt.plot(synth_windows[idx], color="#d96cfa", alpha=0.45, linewidth=1)
    plt.title(title)
    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def write_markdown(metrics: dict[str, object], output_path: Path, channel: str, synthetic_path: Path) -> None:
    lines = [
        f"# SDForger Benchmark Metrics ({channel})",
        "",
        f"- Synthetic file: `{synthetic_path}`",
        f"- Real training windows: `{metrics['real_window_count']}`",
        f"- Synthetic windows generated: `{metrics['synthetic_window_count']}`",
        f"- Synthetic windows evaluated: `{metrics['synthetic_window_count_evaluated']}`",
        f"- Distance-metric subset mode: `{metrics['distance_subset_mode']}`",
        f"- Distance-metric paired sample count: `{metrics['paired_sample_count']}`",
        f"- Synthetic value space: `{metrics['synthetic_space']}`",
        f"- SHR computed: `{metrics['compute_shr']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| `MDD` | `{format_metric(metrics['mdd'])}` |",
        f"| `ACD` | `{format_metric(metrics['acd'])}` |",
        f"| `SD` | `{format_metric(metrics['sd'])}` |",
        f"| `KD` | `{format_metric(metrics['kd'])}` |",
        f"| `ED` | `{format_metric(metrics['ed'])}` |",
        f"| `DTW` | `{format_metric(metrics['dtw'])}` |",
        f"| `SHR` | `{format_metric(metrics['shr'])}` |",
        "",
        "## Notes",
        "",
        "- `MDD/ACD/SD/KD/ED/DTW` follow the TSGBench code path.",
        "- `SHR` / `SHAP-RE` uses the SIDL-based shapelet reconstruction path from the supplementary implementation when `--compute-shr` is enabled.",
        "- `ED/DTW` require equal sample counts, so synthetic windows are deterministically reduced to the real training-window count before pairing.",
        "- `--max-synthetic-samples` applies a hard cap before all metrics when a run generates more samples than the paper protocol intends to evaluate.",
        "- fms-dgt generated JSONL values are usually already standardized; use `--synthetic-space=original` only for raw-scale synthetic JSONL.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    synth_windows_raw = load_synthetic_windows(args.synthetic_jsonl, args.channel)
    synth_windows = cap_synthetic_windows(synth_windows_raw, args.max_synthetic_samples)
    real_scaled_2d, real_original_2d = build_real_training_windows(
        real_parquet=args.real_parquet,
        channel=args.channel,
        train_length=args.train_length,
        train_samples=args.train_samples,
        augmentation_strategy=args.augmentation_strategy,
        min_windows_length=args.min_windows_length,
        min_windows_number=args.min_windows_number,
        train_splitting=args.train_splitting,
    )

    if synth_windows.shape[1] != real_original_2d.shape[1]:
        raise ValueError(
            "Window length mismatch: "
            f"real={real_original_2d.shape[1]} vs synthetic={synth_windows.shape[1]}"
        )

    if args.synthetic_space == "scaled":
        synth_scaled_all = synth_windows
    else:
        synth_scaled_all = standardize_original_synthetic_from_real(real_original_2d, synth_windows)

    real_scaled_all = np.expand_dims(np.asarray(real_scaled_2d, dtype=np.float64), axis=2)
    synth_scaled_all_3d = np.expand_dims(np.asarray(synth_scaled_all, dtype=np.float64), axis=2)

    paired_count = min(real_scaled_all.shape[0], synth_scaled_all_3d.shape[0])
    if paired_count <= 0:
        raise ValueError("Need at least one real and one synthetic window for paired metrics")
    real_scaled_paired = real_scaled_all[:paired_count]
    synth_scaled_paired = choose_synthetic_subset(
        synth_scaled_all_3d,
        target_count=paired_count,
        subset_mode=args.subset_mode,
        subset_seed=args.subset_seed,
    )
    synth_plot_paired = choose_synthetic_subset(
        synth_scaled_all,
        target_count=paired_count,
        subset_mode=args.subset_mode,
        subset_seed=args.subset_seed,
    )

    metrics: dict[str, object] = {
        "mdd": calculate_mdd(real_scaled_all, synth_scaled_all_3d),
        "acd": calculate_acd(real_scaled_all, synth_scaled_all_3d),
        "sd": calculate_sd(real_scaled_all, synth_scaled_all_3d),
        "kd": calculate_kd(real_scaled_all, synth_scaled_all_3d),
        "ed": calculate_ed(real_scaled_paired, synth_scaled_paired),
        "dtw": calculate_dtw(real_scaled_paired, synth_scaled_paired),
        "shr": (
            calculate_shr_sid(real_scaled_all, synth_scaled_all_3d, seed=args.shr_seed)
            if args.compute_shr
            else None
        ),
        "real_window_count": int(real_scaled_all.shape[0]),
        "synthetic_window_count": int(synth_windows_raw.shape[0]),
        "synthetic_window_count_evaluated": int(synth_scaled_all_3d.shape[0]),
        "paired_sample_count": int(paired_count),
        "paired_sample_policy": "min(real_window_count, synthetic_window_count_evaluated)",
        "sequence_length": int(real_scaled_all.shape[1]),
        "distance_subset_mode": args.subset_mode,
        "distance_subset_seed": int(args.subset_seed),
        "max_synthetic_samples": args.max_synthetic_samples,
        "compute_shr": bool(args.compute_shr),
        "shr_seed": int(args.shr_seed),
        "train_length": int(args.train_length),
        "train_samples": int(args.train_samples),
        "augmentation_strategy": args.augmentation_strategy,
        "min_windows_length": int(args.min_windows_length),
        "min_windows_number": int(args.min_windows_number),
        "train_splitting": args.train_splitting,
        "synthetic_space": args.synthetic_space,
        "metric_impl": "tsgbench_exact_for_mdd_acd_sd_kd_ed_dtw",
        "shr_status": "sidl_shapelet_reconstruction" if args.compute_shr else "skipped",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")
    write_markdown(metrics, args.output_md, args.channel, args.synthetic_jsonl)
    if args.synthetic_space == "scaled":
        plot_title = "Real vs Synthetic (standardized SDForger space, paired subset)"
        real_plot = np.asarray(real_scaled_2d[:paired_count], dtype=np.float64)
    else:
        plot_title = "Real vs Synthetic (synthetic standardized from original scale, paired subset)"
        real_plot = np.asarray(real_scaled_2d[:paired_count], dtype=np.float64)
    save_overlay_plot(real_plot, synth_plot_paired, args.output_plot, title=plot_title)


if __name__ == "__main__":
    main()
