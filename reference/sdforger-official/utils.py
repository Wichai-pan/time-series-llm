# Copyright The DiGiT Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import Counter
from typing import Any, List, Literal, Tuple, Union
import logging
import random
import warnings

# Third Party
from numpy import linalg as LA
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats

# Local
from fms_dgt.constants import BASE_LOGGER_NAME

_logger = logging.getLogger(BASE_LOGGER_NAME)

# ===========================================================================
#                        PREPROCESSING BLOCK
# ===========================================================================


# paper-note-added: 预处理总入口（对应论文 Sec3.1 第1步“周期感知分段”的上游数据准备）。
# paper-note-added: 根据数据增强策略分派：multisample（多样本各取窗口）或 uni/multivariate（单/多通道按周期切窗）。
def preprocess_train_data(
    data,
    train_channels,
    train_length,
    train_samples,
    augmentation_strategy,
    min_windows_length,
    min_windows_number=10,
    train_splitting="minimize-overlap",
):
    if augmentation_strategy in ("multisample"):
        return preprocess_multisample_data(data, train_channels, train_samples, train_length)
    elif augmentation_strategy in ("univariate", "multivariate"):
        return preprocess_uni_multi_variate_data(
            data,
            train_channels,
            train_length,
            min_windows_length,
            min_windows_number,
            train_splitting,
        )
    else:
        raise ValueError(
            f"Unsupported augmentation strategy: '{augmentation_strategy}'. "
            "Choose from 'multisample', 'univariate', or 'multivariate'."
        )


# paper-note-added: 对应论文 A.1 单序列分段。先对每个通道用 ACF 估计主周期 P，再以“跨通道最频繁周期”统一切窗，约 I=15 个实例即可。
def preprocess_uni_multi_variate_data(
    data,
    train_channels,
    train_length,
    min_windows_length,
    min_windows_number,
    train_splitting,
    logger=None,
):
    log = logger or _logger
    if train_splitting not in {"minimize-overlap", "maximize-overlap", "no-periodicity"}:
        raise ValueError(
            f"Unsupported train_splitting strategy: '{train_splitting}'. "
            "Expected one of: 'minimize-overlap', 'maximize-overlap', 'no-periodicity'."
        )

    start = 0
    end_train = train_length
    train_data = get_data(data, start, end_train, train_channels)

    preprocessed_train = []
    chosen_period_per_channel = []

    # Iterate over each channel and calculate its period and window length
    for idx, channel_data in enumerate(train_data):
        channel_name = train_channels[idx]
        log.info("CHANNEL: %s", channel_name)

        # paper-note-added: A.1 用 ACF 估计该通道的主周期 P（排除 lag0 的显著峰、按自相关排序、取满足 P<L/2 的最大者）。
        chosen_period, _ = compute_period(
            channel_data,
            min_points=min_windows_length,
            min_windows=min_windows_number,
            logger=logger,
        )
        log.info("Chosen period %d for channel (%s)", chosen_period, channel_name)
        chosen_period_per_channel.append(chosen_period)

    # Use the most frequent period across channels
    # paper-note-added: A.1 多通道场景下，取所有通道中出现最频繁的周期作为统一切窗周期，保证各通道窗口对齐。
    chosen_period = max(chosen_period_per_channel, key=chosen_period_per_channel.count)
    log.info("Overall chosen period: %d", chosen_period)

    # Construct windows for each channel using the chosen period
    # paper-note-added: A.1 用统一周期为每个通道切出 I 个窗口（实例），窗口步长 s 调整为周期 P 的整数倍。
    for idx, channel_data in enumerate(train_data):
        verbose = idx == 0
        windows, _ = construct_windows(
            channel_data,
            chosen_period,
            min_windows_length,
            min_windows_number,
            train_splitting,
            verbose,
            logger=logger,
        )
        preprocessed_train.append(windows)

    # Scale the preprocessed data
    scaled_preprocessed_train, list_fitted_scaler = standardscale_train(
        preprocessed_train, logger=logger
    )

    return (scaled_preprocessed_train, preprocessed_train, list_fitted_scaler)


# paper-note-added: 框架管线代码（multisample 增强分支）：从多个样本列中随机抽 train_samples 个作为实例，与论文核心方法无直接对应。
def preprocess_multisample_data(data, train_channels, train_samples, train_length, logger=None):

    sample_str = train_channels[0]
    sample_columns = [col for col in data.columns if col.startswith(sample_str)]

    if len(list(sample_columns)) <= train_samples:
        (logger or _logger).info("Enough samples, choosing windows in different samples")
        raise ValueError(
            f"Number of samples ({len(list(sample_columns))}) "
            f"is less than "
            f"the total number of samples wanted ({train_samples})."
        )

    id_samples = random.sample(list(sample_columns), train_samples)

    start = 0
    end = train_length

    # train data
    id_train_samples = id_samples[0:train_samples]
    train_data = get_data(data, start, end, id_train_samples)
    train_data = [train_data.tolist()]

    scaled_preprocessed_train, list_fitted_scaler = standardscale_train(train_data, logger=logger)

    return (scaled_preprocessed_train, train_data, list_fitted_scaler)


# paper-note-added: 工具函数（框架管线）：判断某列是数值型还是类别型并给出分布摘要，供文本→表格解码时正确解析系数，无直接论文公式对应。
def get_feature_distribution(column: List[Any]) -> Tuple[str, bool, Union[dict, Counter]]:
    if isinstance(column[0], float):
        _, p_value = stats.shapiro(column)
        normal_distr = p_value > 0.05
        stats_summary = {
            "min": min(column),
            "max": max(column),
            "mean": np.mean(column),
            "median": np.median(column),
            "std": np.std(column),
            "data": column,
        }
        return "numerical", normal_distr, stats_summary
    else:
        return "categorical", False, Counter(column)


# ===========================================================================
#                        Data upload and preprocessing
# ===========================================================================


# paper-note-added: 框架管线工具：按通道名与时间区间从 DataFrame 取出原始序列并转为 numpy，无直接论文方法对应。
def get_data(data, start, end, id_channel):

    indices_channel = [data.columns.get_loc(channel) for channel in id_channel]
    data = data.T

    if start < 0 or end > data.shape[1]:
        raise ValueError(
            f"Not enough data present!!! Trying to get data from indices:{start} to {end} but available data is {data.shape[1]} "
        )

    train_data = data.to_numpy()[indices_channel, start:end]
    train_data = np.array(train_data, dtype=np.float32)
    train_data = interpolate_na(train_data)

    return train_data


# paper-note-added: 框架管线工具：对每个通道线性插值填补缺失值，属数据清洗，无直接论文方法对应。
def interpolate_na(data):
    """
    Interpolates NaN values in each channel of the train data using linear interpolation.

    Args:
        data (np.array): 2D numpy array of shape (n_channels, n_points),
                               where each row is a time series with NaN values.

    Returns:
        np.array: 2D numpy array with NaN values interpolated.
    """
    interpolated_data = []

    for channel in data:
        # Convert to a Pandas Series to use interpolate method
        channel_series = pd.Series(channel)
        # Interpolate NaN values linearly
        channel_interpolated = channel_series.interpolate(method="linear", limit_direction="both")
        # Append the interpolated channel back to list
        interpolated_data.append(channel_interpolated.values)

    # Convert the list back to a numpy array
    return np.array(interpolated_data)


# paper-note-added: 对应 Sec3.1 嵌入前的标准化步骤：逐通道对窗口做 StandardScaler，使后续 FPC/ICA 在零均值单位方差数据上计算，并保留 scaler 供解码时反标准化。
def standardscale_train(preprocessed_train, logger=None):
    # Standard Scaling the data useful for aggregated scores in evaluation
    log = logger or _logger

    # -- by timestamps --
    scaled_preprocessed_train = []
    list_fitted_scaler = []

    for idx, channel_data in enumerate(preprocessed_train):
        arr_train = np.array(channel_data)
        scaler = StandardScaler()
        scaled_arr_train = scaler.fit_transform(arr_train)

        scaled_preprocessed_train.append(scaled_arr_train)
        list_fitted_scaler.append(scaler)

        log.info("Channel %d train data standardized. Shape: %s", idx, scaled_arr_train.shape)

    return np.array(scaled_preprocessed_train), list_fitted_scaler


# ===========================================================================
#                       Periodicity Aware Segmentation
# ===========================================================================


# paper-note-added: 对应论文 A.1（第1步周期感知分段的核心）：用自相关函数 ACF 估计主周期，返回按自相关强度排序的候选 lag 列表。
def estimate_period_acf(time_series):
    """
    Estimate the dominant periods in a time series based on its autocorrelation function (ACF).

    Parameters:
    time_series (array-like): The input time series data.

    Returns:
    list: Lags corresponding to the most significant peaks in the ACF, sorted by significance.
    """
    # Compute the ACF of the time series
    # paper-note-added: A.1 计算自相关函数；nlags=L/2 体现“主周期 P 应满足 P < L/2”的约束。
    autocorr = acf(time_series, nlags=len(time_series) // 2, fft=True)

    # Identify peaks in the ACF, excluding lag 0
    # paper-note-added: A.1 找 ACF 的显著峰并排除 lag0（lag0 自相关恒为1，不代表周期）。
    peaks, _ = find_peaks(autocorr[1:])  # Skip lag 0
    peaks += 1  # Adjust indices since lag 0 was skipped

    # Evaluate and sort peaks by their autocorrelation values
    # paper-note-added: A.1 按自相关值从高到低排序候选周期（自相关越大代表周期性越强）。
    sorted_peaks = sorted(peaks, key=lambda x: autocorr[x], reverse=True)

    return sorted_peaks


# paper-note-added: 对应 A.1：以 ACF 为主、FFT 为兜底确定周期 P，并保证能切出至少 min_windows≈I 个长度满足 min_points 的窗口。
def compute_period(time_series, min_points=1120, min_windows=10, logger=None):
    """
    Computes the period of a time series using ACF as the primary method
    and Fourier Transform as a fallback. Ensures the period allows for at
    least `min_windows` windows and meets the `min_points` condition.

    Parameters:
        time_series (array-like): The input time series data.
        min_points (int): Minimum points required for each window.
        min_windows (int): Minimum number of windows required.

    Returns:
        period (int): The computed period for the time series.
        window_length (int): The calculated window length.
    """
    log = logger or _logger
    # Determine maximum allowable period
    # paper-note-added: A.1 由“需切出 min_windows 个长度≥min_points 的窗口”反推可接受的最大周期，等价于 P<L/2 约束的工程化上界。
    max_period = max(1, (len(time_series) - min_points) // (min_windows - 1))

    # Try to find the period using ACF
    # paper-note-added: A.1 取按自相关排序的候选周期中、第一个小于 max_period 的作为主周期 P（即“满足约束的最强周期”）。
    acf_periods = estimate_period_acf(time_series)
    period = max_period
    period_source = "Fallback to max_period"
    position = len(acf_periods)

    for idx, p in enumerate(acf_periods):
        if p < max_period:
            period = p
            period_source = "ACF"
            position = idx
            break

    # If no valid period is found using ACF, fallback to Fourier Transform
    # paper-note-added: A.1 兜底策略（论文以 ACF 为主）：ACF 无有效周期时用 FFT 频谱峰值的倒数作为周期估计。
    if period_source != "ACF":
        freqs = scipy.fftpack.fftfreq(len(time_series), d=1)
        power_spectrum = np.abs(scipy.fftpack.fft(time_series)) ** 2
        positive_freq_indices = np.where(
            (freqs > 0) & (1 / np.where(freqs != 0, freqs, np.inf) <= max_period)
        )[0]

        if positive_freq_indices.size > 0:
            peak_index = positive_freq_indices[np.argmax(power_spectrum[positive_freq_indices])]
            peak_freq = freqs[peak_index]
            period = round(1 / peak_freq)
            period_source = "FFT"

    # Calculate window length based on the determined period
    # paper-note-added: A.1 窗口长度取为周期 P 的整数倍 N·P（≥min_points），使每个窗口（实例）恰好包含整数个周期 L=N·P。
    N = max(1, (min_points + period - 1) // period)
    window_length = N * period

    # Diagnostics
    log.info("  - Max possible period: %s", max_period)
    log.info("  - First ACF periods (ranked): %s", acf_periods[: (position + 1)])
    log.info("  - Selected period: %s (Source: %s)", period, period_source)

    return period, window_length


# paper-note-added: 对应 A.1 的滑窗切分（第1步分段的产出）：用主周期 P 把长序列 L0 切成 I 个等长窗口（实例），步长 s 取为 P 的整数倍以保持相位对齐。
# paper-note-added: 三种切窗模式：minimize-overlap（论文默认，步长≈N·P，最小化重叠）、maximize-overlap（步长=P，最大化重叠多生成实例）、no-periodicity（不考虑周期均匀铺窗）。
def construct_windows(
    time_series,
    period,
    window_length=1120,
    min_windows=10,
    split="maximize-overlap",
    verbose=True,
    logger=None,
):
    """
    Constructs windows for the time series based on the computed period.
    Ensures that the overlapping points of the last window are a multiple of the period.
    """

    # Try to compute the number of windows based on minimal overlap
    computed_windows = len(time_series) // window_length

    windows = []
    start = 0
    extra_window = False
    overlap_extra_window, periods_extra_window = None, None

    if split == "no-periodicity":
        # Compute total span of data and spacing to get min_windows
        max_start = len(time_series) - window_length
        if min_windows <= 1 or max_start <= 0:
            windows.append(time_series[:window_length])
            overlap_step = 0
        else:
            overlap_step = max_start // (min_windows - 1)
            for i in range(min_windows):
                start = i * overlap_step
                end = start + window_length
                if end <= len(time_series):
                    windows.append(time_series[start:end])
                else:
                    # If the last window goes out of bounds, adjust
                    windows.append(time_series[-window_length:])
                    break
        total_coverage = len(windows[0]) + (len(windows) - 1) * overlap_step

    elif split == "maximize-overlap":
        overlap_step = period
        # Generate initial windows
        while start + window_length <= len(time_series):
            windows.append(time_series[start : start + window_length])
            start += overlap_step
        total_coverage = len(windows[0]) + (len(windows) - 1) * overlap_step

    # paper-note-added: A.1 论文主用策略：步长 s=max(1, floor((L0-L)/(I-1))) 并向下取整为 P 的整数倍，使 I 个窗口尽量铺满序列且重叠最小。
    elif split == "minimize-overlap":
        # Calculate overlap_step
        if computed_windows >= min_windows:
            number_full_periods_in_windows = window_length // period
            overlap_step = number_full_periods_in_windows * period

            # Generate initial windows
            while start + window_length <= len(time_series):
                windows.append(time_series[start : start + window_length])
                start += overlap_step
        else:
            # Calculate overlap step to get exactly min_windows
            # paper-note-added: A.1 步长公式 s≈floor((L0-L)/(I-1))，使恰好得到 min_windows≈I 个实例。
            max_start_pos = len(time_series) - window_length
            overlap_step = max(1, (max_start_pos + (min_windows - 1)) // (min_windows - 1))

            # Ensure overlap_step is a multiple of the period
            # paper-note-added: A.1 关键约束——步长 s 调整为周期 P 的最近整数倍，保证各窗口的周期相位一致。
            overlap_step = (overlap_step // period) * period

            # Generate windows with exact min_windows count
            windows = []
            start = 0
            for _ in range(min_windows):
                if start + window_length > len(
                    time_series
                ):  # Adjust last window if exceeding bounds
                    start = len(time_series) - window_length
                windows.append(time_series[start : start + window_length])
                start += overlap_step

        # Calculate total coverage and check if we can add an extra window
        total_coverage = len(windows[0]) + (len(windows) - 1) * overlap_step
        if len(time_series) - total_coverage > period:
            # Add an extra window to maximize total coverage
            extra_window = True
            end_extra_windows = (
                total_coverage + ((len(time_series) - total_coverage) // period) * period
            )
            windows.append(time_series[end_extra_windows - window_length : end_extra_windows])
            overlap_extra_window = len(windows[0]) - (end_extra_windows - total_coverage)
            periods_extra_window = overlap_extra_window // period
            total_coverage += len(windows[0]) - overlap_extra_window

    else:
        raise ValueError(
            "Unknown SPLIT parameters. Possible values: maximize-overlap, minimize-overlap, no-periodicity"
        )

    # Diagnostics
    if verbose:
        log = logger or _logger
        log.info("Period: %s", period)
        log.info("Number of windows: %s", len(windows))
        log.info("Points in each window: %s", window_length)
        log.info("Periods in a window: %s", window_length // period)
        log.info("Overlapping points: %s", window_length - overlap_step)
        log.info("Overlapping periods: %s", (window_length - overlap_step) // period)
        if extra_window:
            log.info("Extra window overlapping points: %s", overlap_extra_window)
            log.info("Extra window overlapping periods: %s", periods_extra_window)
            log.info("Total coverage: %s\n", total_coverage)

    return windows, window_length - overlap_step


# ===========================================================================
#                       EMBEDDING BLOCK
# ===========================================================================


# paper-note-added: 对应 Sec3.1 + A.2（第2步嵌入，FPC 选项）：把每个窗口投影到 k 个基函数（FPC）上得到嵌入系数 e_ij，构成 I×K 的嵌入表 E。
# paper-note-added: FPC=函数型主成分，基于协方差、按方差排序、最简洁；k 取“解释目标方差百分比的最小维数”（A.2，K>25 易致微调不稳定）。
def fpc_embed_data(preprocessed_data, embedding_dim, percentage_of_variance_explained, logger=None):
    log = logger or _logger
    n_var = len(preprocessed_data)  # Number of features

    # Compute embeddings
    data_embedded = []
    data_embedded_full = []
    fpc_basis = []
    fpc_basis_full = []
    var_explained = []
    embedding_dims = []
    full_embedding_dims = []
    sdforger_input = []

    for var in range(n_var):
        # Standardize each feature independently
        data_train_std = preprocessed_data[
            var
        ]  # Standardization already done in preprocessing step

        # Compute FPCA with dynamic embedding_dim based on percentage_of_variance
        # paper-note-added: Sec3.1 FPC 核心——对协方差矩阵 X^T X 做特征分解，特征向量即基函数 b_j，按特征值（方差）从大到小排序（FPC 有序、最简洁）。
        eigvals_var, eigenfuns = LA.eigh(data_train_std.T @ data_train_std)
        eigvals_sorted = eigvals_var[::-1]
        cumsum_eigvals = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted)

        # Determine embedding dimensions
        # paper-note-added: A.2 自动选 k——取累计方差首次达到目标百分比的最小维数（下限为2）。
        if embedding_dim == "auto":
            embedding_dim = max(
                2, np.argmax(cumsum_eigvals >= percentage_of_variance_explained) + 1
            )

        else:
            embedding_dim = embedding_dim  # Use fixed embedding dimension if provided

        log.info(
            "Latent dimension channel %s: %s",
            var,
            embedding_dim,
        )
        log.info(
            "\nVariance retained for channel %s: %.4f",
            var,
            cumsum_eigvals[embedding_dim - 1],
        )
        full_embedding_dim = max(np.argmax(cumsum_eigvals >= 0.99) + 1, embedding_dim)
        embedding_dims.append(embedding_dim)
        full_embedding_dims.append(full_embedding_dim)

        # Store eigenfunctions and truncated eigenvalues
        # paper-note-added: Sec3.1 嵌入系数 e_ij = <X_i, b_j> 的具体实现——X · 前 k 个基函数即内积投影，得到 I×k 的系数矩阵（截断到 embedding_dim）。
        data_embedded.append((data_train_std @ eigenfuns[:, -embedding_dim:]).copy())
        # paper-note-added: full 版本保留到 99% 方差（A.3 fpc-filled 用，重构时把生成系数填回高维表的低频分量）。
        data_embedded_full.append((data_train_std @ eigenfuns[:, -full_embedding_dim:]).copy())
        fpc_basis.append(eigenfuns[:, -embedding_dim:])
        fpc_basis_full.append(eigenfuns[:, -full_embedding_dim:])
        var_explained.append(cumsum_eigvals[embedding_dim - 1])

        # Prepare SDForger input
        sdforger_input.append(data_embedded[-1])

    # paper-note-added: Sec3.1 把各通道系数横向拼接成嵌入表 E，列数 K=Σk_c，即送入 LLM 的“表格化”特征（value_0..value_{K-1}）。
    sdforger_input = np.hstack(sdforger_input)

    return (
        pd.DataFrame(
            sdforger_input,
            columns=["value_" + str(k) for k in range(0, sdforger_input.shape[1])],
        ),
        embedding_dims,
        data_embedded,
        data_embedded_full,
        fpc_basis,
        fpc_basis_full,
    )


# paper-note-added: 对应 Sec3.1（第2步嵌入，FastICA 选项）：把窗口分解为 k 个统计独立成分，得到嵌入系数 e_ij，构成嵌入表 E。
# paper-note-added: ICA 成分无序、信息均匀分布，论文指出其在“生成”阶段比 FPC 更稳健；成本只取决于实例数 I 与成分数 k，与序列长度 L 无关。
def fica_embed_data(
    preprocessed_data, embedding_dim, percentage_of_variance_explained, logger=None
):
    log = logger or _logger
    n_var = len(preprocessed_data)  # Number of features

    # Compute embeddings
    fica_mixing = []
    fica_mean = []
    data_embedded = []
    embedding_dims = []
    sdforger_input = []
    var_explained = []

    for var in range(n_var):
        data_train_std = preprocessed_data[var]

        # Determine optimal k (number of components)
        # paper-note-added: A.2 自动选 k——从 k=2 起逐步增大，直到重构方差达到目标百分比；上界 max_k=30 呼应“K>25 易致微调不稳定”。
        if embedding_dim == "auto":
            k_auto = 2
            max_k = 30
            while k_auto <= max_k:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # paper-note-added: Sec3.1 FastICA 嵌入——fit_transform 得到 I×k 的独立成分系数（即内积式嵌入 e_ij）。
                    fica = FastICA(n_components=k_auto, random_state=0)
                    data_temp = fica.fit_transform(data_train_std)
                    data_reconstructed = data_temp @ fica.mixing_.T + fica.mean_

                # Compute total variance retained across all samples and features
                original_var = np.var(data_train_std, axis=1).sum()
                reconstructed_var = np.var(data_reconstructed, axis=1).sum()
                retained = reconstructed_var / original_var

                if retained >= percentage_of_variance_explained:
                    break
                k_auto += 1
            embedding_dim = k_auto
        else:
            embedding_dim = embedding_dim

        log.info(
            "Latent dimension channel %s: %s",
            var,
            embedding_dim,
        )
        # paper-note-added: Sec3.1 用最终 k 做 FastICA，data_embedded 即嵌入系数表；fica.mixing_/mean_ 为解码所需的混合矩阵与均值（第6步逆变换用）。
        fica = FastICA(n_components=embedding_dim, random_state=0)
        data_embedded.append(fica.fit_transform(data_train_std))
        data_reconstructed = data_embedded[-1] @ fica.mixing_.T + fica.mean_

        # Compute retained variance (final report)
        original_var = np.var(data_train_std, axis=1).sum()
        reconstructed_var = np.var(data_reconstructed, axis=1).sum()
        retained = reconstructed_var / original_var
        log.info(
            "\nVariance retained for channel %s: %.4f",
            var,
            retained,
        )

        embedding_dims.append(embedding_dim)
        fica_mixing.append(fica.mixing_)
        fica_mean.append(fica.mean_)
        sdforger_input.append(data_embedded[-1])
        var_explained.append(retained)

    # paper-note-added: Sec3.1 各通道独立成分系数横向拼成嵌入表 E（列数 K=Σk_c），作为 LLM 的表格化输入。
    sdforger_input = np.hstack(sdforger_input)

    return (
        pd.DataFrame(
            sdforger_input,
            columns=["value_" + str(k) for k in range(0, sdforger_input.shape[1])],
        ),
        embedding_dims,
        data_embedded,
        fica_mixing,
        fica_mean,
    )


# paper-note-added: 对应 Sec3.2.1（第3步文本编码）：把一个实例的嵌入系数向量转成文本，主用“填空(fill-in-the-middle)”模板。
# paper-note-added: 模板形如 "Input: value_pi(k) is [blank], ... [sep] Target: e_i,pi(k) [answer] ..."，把每个系数的“位置占位”与“数值答案”分开。
def embeddings_to_text(
    row: Any,
    columns: List[str],
    eos_token: str,
    permute: bool = True,
    text_template: Literal[
        "base_template", "fim_template", "fim_template_textual_encoding"
    ] = "fim_template",
    input_tokens_precision: int = 4,
):

    # Shuffle columns, if requested
    # paper-note-added: Sec3.2.1 关键——对 K 个特征做随机排列 pi（每个实例独立），以消除位置/顺序偏置，使 LLM 不依赖固定列序。
    if permute:
        random.shuffle(columns)

    decimal = input_tokens_precision

    # Prepare text based on the requested template
    # paper-note-added: base_template 为对照模板（非 FIM），直接 "col is value" 列出系数，不分占位/答案两段。
    if text_template == "base_template":
        text = ", ".join(
            [
                (
                    f"{col} is {row[col]:.{decimal}f}"
                    if isinstance(row[col], float)
                    else f"{col} is {row[col]}"
                )
                for col in columns
            ]
        )
        text = text + eos_token
    # paper-note-added: Sec3.2.1 论文主用的 FIM（fill-in-the-middle）模板：Input 段列出按 pi 排列的特征占位 [blank]，Target 段给出对应系数数值 + [answer]。
    elif text_template == "fim_template":
        # paper-note-added: Input 段——"value_pi(k) is [blank]"，只给特征名与空位，不泄露数值（供推理时作 prompt）。
        text_input = ", ".join([f"{col} is [blank]" for col in columns])
        # paper-note-added: Target 段——按相同的 pi 顺序给出嵌入系数 e_i,pi(k)（保留 decimal 位小数）后接 [answer]，即模型需自回归填出的答案。
        text_target = " ".join(
            [
                (
                    f"{row[col]:.{decimal}f} [answer]"
                    if isinstance(row[col], float)
                    else f"{row[col]} [answer]"
                )
                for col in columns
            ]
        )
        # paper-note-added: 拼成 "Input: ... [sep] Target: ..." 完整训练文本，[sep] 分隔占位与答案两段。
        text = "Input: " + text_input + " [sep] Target: " + text_target + eos_token
    # paper-note-added: fim_template_textual_encoding 为带条件(Condition)的扩展 FIM 模板：在 FIM 基础上把类别型字段作为条件前缀，支持文本条件生成。
    else:
        text_categorical = "".join(
            [
                (f"{col} is {row[col]}, " if not isinstance(row[col], float) else "")
                for col in columns
            ]
        ).strip(", ")

        text_input = "".join(
            [(f"{col} is [blank], " if isinstance(row[col], float) else "") for col in columns]
        ).strip(", ")
        text_target = "".join(
            [
                (f"{float(row[col]):.{decimal}f} [answer] " if isinstance(row[col], float) else "")
                for col in columns
            ]
        ).strip(" ")
        text = (
            "Condition: "
            + text_categorical
            + " [sep] Input: "
            + text_input
            + " [sep] Target: "
            + text_target
            + eos_token
        )

    # Return
    return text


# ===========================================================================
#                      RECOVERING BLOCK
# ===========================================================================


# paper-note-added: 对应 Sec3.2（第5步文本解码）：把 LLM 生成的文本解析回嵌入系数表（I×K），是文本编码 embeddings_to_text 的逆过程。
# paper-note-added: 按模板解析 Input/Target 两段，依据原数据列的数值/类别属性把 [answer] 处的字符串还原为系数；解析失败/缺值的样本会被丢弃（呼应 A.3 in-generation filtering）。
def convert_texts_to_tabular_data(
    text: List[str],
    original_dataset: pd.DataFrame,
    text_template: Literal[
        "base_template", "fim_template", "fim_template_textual_encoding"
    ] = "fim_template",
    logger=None,
) -> pd.DataFrame:
    """Converts the sentences back to tabular data

    Args:
        text (List[str]): List of the tabular data in text form
        original_dataset (pd.DataFrame): Original dataset
        text_template (str): Text template

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """

    feature_distribution = {
        col: get_feature_distribution(original_dataset[col].tolist())
        for col in original_dataset.columns
    }
    columns = original_dataset.columns.to_list()

    generated = []

    if text_template == "base_template":
        # Convert text to tabular data for base_template
        for t in text:
            features = t.split(",")
            td = dict.fromkeys(columns, "NaN")

            # Transform all features back to tabular data
            for f in features:
                values = f.strip().split(" is ")
                if values[0] in columns and td[values[0]] == "NaN":
                    try:
                        if len(values) >= 2:
                            if feature_distribution[values[0]][0] == "numerical":
                                i = 1
                                while i <= len(values):
                                    try:
                                        td[values[0]] = values[i]
                                        i = len(values) + 1
                                    except ValueError:
                                        i += 1
                                        continue
                            elif feature_distribution[values[0]][0] == "categorical":
                                # TODO: create categorical check
                                pass
                    except IndexError:
                        pass
            generated.append(td)
        df_gen = pd.DataFrame(generated)
        df_gen.replace("None", None, inplace=True)

    # paper-note-added: Sec3.2 FIM 模板解码：按 [sep]/[answer] 切分，先取 Input 段的特征名顺序（生成时的随机排列 pi），再把 Target 段的数值按同序对齐回各特征。
    if text_template == "fim_template":
        # Convert text to tabular data
        for t in text:
            try:
                features_input = t.split(",")
                td = dict.fromkeys(columns, "NaN")

                # paper-note-added: 拆出 Input(占位)段与 Target(系数)段——对应编码时的 "Input: ... [sep] Target: ..." 结构。
                input_part, output_part = t.split(" [sep] Target: ")
                input_values = input_part.removeprefix("Input: ")
                output_values = output_part.removesuffix(" [answer]")

                features_input = input_values.split(",")
                features_output = output_values.split(" [answer] ")
            except ValueError:
                # If text is malformed, skip it
                # paper-note-added: A.3 过滤——文本格式错误（无法解析）的生成实例直接丢弃。
                continue
            if len(features_output) >= len(features_input):
                # Transform all features back to tabular data
                # paper-note-added: Sec3.2 按 pi 顺序把第 i 个特征名与第 i 个 [answer] 数值配对，还原出该实例的嵌入系数（逆置换）。
                for i, f in enumerate(features_input):
                    col_i = f.strip().split(" is ")[0]
                    if col_i in columns:
                        try:
                            val_i = features_output[i]
                            if col_i in columns:
                                if feature_distribution[col_i][0] == "numerical":
                                    td[col_i] = float(val_i)  # Parse numerical values
                                else:
                                    td[col_i] = val_i  # Keep categorical as string
                        except ValueError:
                            continue
            generated.append(td)
        # Create DataFrame and replace placeholder "NaN" with actual NaN
        # paper-note-added: 未被成功填回的列保留为 NaN，后续按 A.3 丢弃含缺失值的实例。
        df_gen = pd.DataFrame(generated)
        df_gen.replace("NaN", None, inplace=True)

    if text_template == "fim_template_textual_encoding":
        # Convert text to tabular data
        for t in text:
            try:
                features_input = t.split(",")
                td = dict.fromkeys(columns, "NaN")

                # get data_name
                new_part_condition, new_part_values = t.split(" [sep] Input: ")
                input_textual_data = new_part_condition.removeprefix("Condition: data is ")

                input_values, output_part = new_part_values.split(" [sep] Target: ")
                # input_values = input_part.removeprefix("Input: ")
                output_values = output_part.removesuffix(" [answer]")

                features_input = input_values.split(",")
                features_output = output_values.split(" [answer] ")

            except ValueError:
                # If text is malformed, skip it
                continue

            if len(features_output) >= len(features_input):
                # Transform all features back to tabular data
                td["data"] = input_textual_data
                for i, f in enumerate(features_input):
                    col_i = f.strip().split(" is ")[0]
                    if col_i in columns:
                        try:
                            val_i = features_output[i]

                            if col_i in columns:
                                if feature_distribution[col_i][0] == "numerical":
                                    td[col_i] = float(val_i)  # Parse numerical values
                                else:
                                    td[col_i] = val_i  # Keep categorical as string
                        except ValueError:
                            (logger or _logger).warning(
                                "Skipping malformed output: %s", features_output
                            )
                            continue
            generated.append(td)
        df_gen = pd.DataFrame(generated)
        df_gen.replace("NaN", None, inplace=True)

    return df_gen


# paper-note-added: 对应 Sec3.3（第6步 FPC 解码）：把生成的嵌入系数线性重构回时间序列 x_i = Σ_j e_ij·b_j，再反标准化回原始量纲。
# paper-note-added: 先按各通道的 k_i 把拼接的生成矩阵切回各通道，再用 FPC 基函数做线性组合；embedding_type=fpc-filled 时把生成的低频系数填回采样到的高频系数以补全细节。
def fpc_transform_to_original_feature_space(
    generated_data,
    preprocessed_data,
    embedding_type,
    embedding_dims,
    data_embedded_full,
    data_embedded,
    fpc_basis,
    fpc_basis_full,
    logger=None,
):
    log = logger or _logger
    n_var = len(preprocessed_data)  # Number of features

    # Reshape generated data to original format (for multivariate case)
    log.info("Generated data shape is: %s", generated_data.shape)
    # paper-note-added: Sec3.3 解析嵌入表——把拼接的 I×K 生成系数按各通道的 k_i 切回每个通道独立的系数块（embedding 表拆分）。
    index = 0
    new_data_embedded = []
    for i in range(n_var):
        k_i = data_embedded[i].shape[1]  # Number of basis components
        new_data_embedded.append(generated_data[:, index : index + k_i])
        index += k_i
        log.info("Reshaping for var_%s which uses %s basis components", n_var, k_i)

    log.info(
        "Generated data shape has been reshaped to: %s", [arr.shape for arr in new_data_embedded]
    )

    # Retrieve to original form based on FPCA components
    new_data = []
    new_data_full = (
        [] if embedding_type == "fpc-filled" else None
    )  # Initialize for fpc-filled if required

    for var in range(n_var):
        # Retrieve the generated data back to the original feature space
        # paper-note-added: Sec3.3 核心公式 x_i = Σ_j e_ij·b_j —— (系数 @ 基函数^T) 即线性重构，再乘 std 加 mean 反标准化回原始尺度。
        reconstructed_data = preprocessed_data[var].std(axis=0) * (
            new_data_embedded[var] @ fpc_basis[var].T
        ) + preprocessed_data[var].mean(axis=0)
        new_data.append(reconstructed_data)

        # For 'fpc-filled', add the generated data to the original data in the embedded space
        # paper-note-added: fpc-filled 变体——从原始高维(99%方差)系数随机采样作底，用生成的低频系数覆盖其前 embedding_dims 个分量，保留真实高频细节后再重构（提升保真度）。
        if embedding_type == "fpc-filled":
            sampled_indices = np.random.choice(
                data_embedded_full[var].shape[0],
                size=new_data_embedded[var].shape[0],
                replace=True,
            )
            data_embedded_full_var = data_embedded_full[var][sampled_indices]
            data_embedded_full_var[:, -embedding_dims[var] :] = new_data_embedded[var]
            reconstructed_full = preprocessed_data[var].std(axis=0) * (
                data_embedded_full_var @ fpc_basis_full[var].T
            ) + preprocessed_data[var].mean(axis=0)
            new_data_full.append(reconstructed_full)

    generated_data = new_data_full if embedding_type == "fpc-filled" else new_data

    return np.stack(generated_data, axis=0)


# paper-note-added: 对应 Sec3.3（第6步 FastICA 解码）：用 ICA 的逆变换 x = e·A^T + mean（A 为混合矩阵 mixing_）把生成系数线性重构回时间序列。
def fica_transform_to_original_feature_space(
    generated_data, preprocessed_data, data_embedded, fica_mixing, fica_mean, logger=None
):
    log = logger or _logger
    n_var = len(preprocessed_data)  # Number of features

    # Reshape generated data to original format (for multivariate case)
    log.info("Generated data shape is: %s", generated_data.shape)
    # paper-note-added: Sec3.3 同 FPC——把拼接的生成系数按各通道 k_i 切回独立系数块。
    index = 0
    new_data_embedded = []
    for i in range(n_var):
        k_i = data_embedded[i].shape[1]  # Number of basis components
        new_data_embedded.append(generated_data[:, index : index + k_i])
        index += k_i
        log.info("Reshaping for var_%s which uses %s basis components", n_var, k_i)

    log.info(
        "Generated data shape has been reshaped to: %s", [arr.shape for arr in new_data_embedded]
    )

    # Retrieve to original form based on FICA components
    new_data = []
    for var in range(n_var):
        # Retrieve the generated data back to the original feature space
        # paper-note-added: Sec3.3 ICA 线性逆变换 x_i = e_i·mixing^T + mean，等价于 x_i=Σ_j e_ij·b_j（b_j 为混合矩阵的列），重构出生成的时间序列。
        reconstructed_data = np.dot(new_data_embedded[var], fica_mixing[var].T) + fica_mean[var]
        new_data.append(reconstructed_data)

    generated_data = new_data

    return np.stack(generated_data, axis=0)
