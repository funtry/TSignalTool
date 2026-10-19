"""Depth-localized period power spectra for stratigraphic signals."""

import numpy as np
from scipy.ndimage import gaussian_filter

SPECTRUM_WINDOW_CYCLES = 4.0
SPECTRUM_OVERLAP_RATIO = 0.9375
SPECTRUM_ZERO_PADDING_FACTOR = 8
SPECTRUM_PERIOD_POINTS = 180
SPECTRUM_DEPTH_POINTS = 360
SPECTRUM_SMOOTH_SIGMA = (1.6, 1.0)


def depth_period_spectrum(
    depth: np.ndarray,
    signal: np.ndarray,
    minimum_period: float,
    maximum_period: float,
    period_margin: float = 0.0,
    window_cycles: float = SPECTRUM_WINDOW_CYCLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """返回平滑的深度–周期功率图（周期、深度、相对功率）。

    滑动窗口谱会先插值到覆盖整个深度范围的均匀网格，再做高斯平滑，
    使热力图连续铺满画布；`period_margin` 用于把周期网格向两侧扩展，
    扩展部分沿用边缘谱值。`window_cycles` 控制滑动窗口包含的周期数，
    越小则深度分辨率越高、周期分辨率越低。功率逐深度归一化，便于识别
    各深度的主导周期。
    """
    depth = np.asarray(depth, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if (
        depth.size != signal.size
        or depth.size < 8
        or not np.all(np.isfinite(depth))
        or not np.all(np.isfinite(signal))
        or np.any(np.diff(depth) <= 0)
        or minimum_period <= 0
        or maximum_period <= minimum_period
    ):
        return None

    depth_step = float(np.median(np.diff(depth)))
    if depth_step <= 0:
        return None
    uniform_depth = np.arange(
        float(depth[0]),
        float(depth[-1]) + depth_step * 0.5,
        depth_step,
    )
    uniform_signal = np.interp(uniform_depth, depth, signal)
    if window_cycles <= 0:
        return None
    window_points = min(
        uniform_signal.size,
        max(16, int(round(window_cycles * maximum_period / depth_step))),
    )
    if window_points < 8:
        return None
    if window_points % 2:
        window_points -= 1

    hop = max(1, int(round(window_points * (1.0 - SPECTRUM_OVERLAP_RATIO))))
    starts = np.arange(0, uniform_signal.size - window_points + 1, hop, dtype=int)
    if starts.size == 0:
        starts = np.asarray([0], dtype=int)

    window = np.hanning(window_points)
    fft_points = SPECTRUM_ZERO_PADDING_FACTOR * window_points
    frequencies = np.fft.rfftfreq(fft_points, d=depth_step)
    frequency_mask = (frequencies > 0) & (
        frequencies >= 1.0 / maximum_period
    ) & (frequencies <= 1.0 / minimum_period)
    if not np.any(frequency_mask):
        return None

    periods = 1.0 / frequencies[frequency_mask]
    period_order = np.argsort(periods)
    periods = periods[period_order]
    power_rows = []
    centers = []
    for start in starts:
        segment = uniform_signal[start:start + window_points]
        segment = segment - float(np.mean(segment))
        spectrum = np.fft.rfft(segment * window, n=fft_points)
        power = np.abs(spectrum[frequency_mask]) ** 2
        power_rows.append(power[period_order])
        centers.append(float(uniform_depth[start + window_points // 2]))

    power_matrix = np.asarray(power_rows, dtype=float)
    row_max = np.max(power_matrix, axis=1, keepdims=True)
    power_matrix = np.divide(
        power_matrix,
        row_max,
        out=np.zeros_like(power_matrix),
        where=row_max > np.finfo(float).tiny,
    )

    margin = max(float(period_margin), 0.0)
    dense_periods = np.linspace(
        float(minimum_period) - margin,
        float(maximum_period) + margin,
        SPECTRUM_PERIOD_POINTS,
    )
    period_rows = np.asarray(
        [np.interp(dense_periods, periods, row) for row in power_matrix],
        dtype=float,
    )
    centers = np.asarray(centers, dtype=float)
    dense_depths = np.linspace(
        float(depth[0]),
        float(depth[-1]),
        SPECTRUM_DEPTH_POINTS,
    )
    dense_power = np.empty((dense_depths.size, dense_periods.size), dtype=float)
    for column in range(dense_periods.size):
        dense_power[:, column] = np.interp(
            dense_depths,
            centers,
            period_rows[:, column],
        )
    dense_power = gaussian_filter(dense_power, sigma=SPECTRUM_SMOOTH_SIGMA, mode="nearest")
    row_max = np.max(dense_power, axis=1, keepdims=True)
    dense_power = np.divide(
        dense_power,
        row_max,
        out=np.zeros_like(dense_power),
        where=row_max > np.finfo(float).tiny,
    )
    return dense_periods, dense_depths, dense_power


