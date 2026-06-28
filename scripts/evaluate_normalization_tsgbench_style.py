#!/usr/bin/env python3
"""Compute TSGBench-style metrics for normalization ablation model-space arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

MODES = [
    "current_activity_window_zscore",
    "joint_window_zscore",
    "global_series_zscore",
    "activity_series_zscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="hand_acc16_x")
    parser.add_argument("--subset-mode", choices=["first", "random"], default="first")
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--skip-dtw",
        action="store_true",
        help="Skip DTW if the environment lacks the fast dtaidistance backend.",
    )
    return parser.parse_args()


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
    std = torch.where(torch.isclose(std, torch.zeros_like(std)), torch.ones_like(std), std)
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
    x_std_3 = torch.where(torch.isclose(x_std_3, torch.zeros_like(x_std_3)), torch.ones_like(x_std_3), x_std_3)
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
    x_var2 = torch.where(torch.isclose(x_var2, torch.zeros_like(x_var2)), torch.ones_like(x_var2), x_var2)
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
    distance_eu = []
    for i in range(ori_data.shape[0]):
        total_distance_eu = 0.0
        for j in range(ori_data.shape[2]):
            total_distance_eu += np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
        distance_eu.append(total_distance_eu / ori_data.shape[2])
    return float(np.mean(np.asarray(distance_eu)))


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


def calculate_dtw(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    try:
        from dtaidistance.dtw_ndim import distance as multi_dtw_distance
    except ImportError:
        multi_dtw_distance = None

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


def choose_synthetic_subset(synth: np.ndarray, target_count: int, subset_mode: str, subset_seed: int) -> np.ndarray:
    if synth.shape[0] < target_count:
        raise ValueError(f"Need at least {target_count} synthetic windows, got {synth.shape[0]}")
    if subset_mode == "first":
        return synth[:target_count]
    rng = np.random.default_rng(subset_seed)
    indices = rng.choice(synth.shape[0], size=target_count, replace=False)
    return synth[np.sort(indices)]


def as_3d(windows: np.ndarray) -> np.ndarray:
    return np.expand_dims(np.asarray(windows, dtype=np.float64), axis=2)


def compute_metrics(real_2d: np.ndarray, synth_2d: np.ndarray, subset_mode: str, subset_seed: int, skip_dtw: bool) -> dict[str, object]:
    real_3d = as_3d(real_2d)
    synth_3d = as_3d(synth_2d)
    synth_paired = choose_synthetic_subset(synth_3d, real_3d.shape[0], subset_mode, subset_seed)
    return {
        "MDD": calculate_mdd(real_3d, synth_3d),
        "ACD": calculate_acd(real_3d, synth_3d),
        "SD": calculate_sd(real_3d, synth_3d),
        "KD": calculate_kd(real_3d, synth_3d),
        "ED": calculate_ed(real_3d, synth_paired),
        "DTW": None if skip_dtw else calculate_dtw(real_3d, synth_paired),
        "real_windows": int(real_3d.shape[0]),
        "synthetic_windows": int(synth_3d.shape[0]),
        "paired_windows": int(real_3d.shape[0]),
        "dtw_status": "skipped" if skip_dtw else "computed",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for mode in MODES:
        run_dir = args.base_output_root / f"pamap2_subject101_norm_ablation_{mode}_{args.channel}"
        if not run_dir.exists():
            rows.append({"mode": mode, "activity": "missing", "status": "missing_dir"})
            continue
        for activity in ["walking", "running"]:
            real = np.load(run_dir / f"{activity}_real_windows_model_space.npy")
            synth = np.load(run_dir / f"{activity}_synthetic_windows_model_space.npy")
            metrics = compute_metrics(real, synth, args.subset_mode, args.subset_seed, args.skip_dtw)
            rows.append({"mode": mode, "activity": activity, "status": "ok", **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "normalization_tsgbench_style_metrics.csv", index=False)
    (args.output_dir / "normalization_tsgbench_style_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Normalization Ablation TSGBench-style Metrics",
        "",
        "These metrics use each normalization setting's saved model-space real and synthetic windows.",
        "They are diagnostic metrics, not a final cross-setting raw-space benchmark.",
        "",
        df.to_markdown(index=False),
        "",
    ]
    (args.output_dir / "normalization_tsgbench_style_metrics.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
