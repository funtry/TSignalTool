"""旋回曲线的波峰识别、旋回个数统计与三项评价指标。"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

DEFAULT_CYCLE_COUNT_PRIOR = 13.09
PEAK_PROMINENCE_RATIO = 0.05


@dataclass(frozen=True)
class CycleMetrics:
    """旋回曲线的波峰、旋回个数与三项指标。"""

    peak_depths: np.ndarray
    peak_values: np.ndarray
    peak_numbers: np.ndarray
    cycle_count: float
    bottom_cycle_fraction: float
    top_cycle_fraction: float
    prior_count: float
    energy_index: float
    smoothness_index: float
    count_proximity: float


def analyze_cycle_curve(
    depth: np.ndarray,
    signal: np.ndarray,
    prior_count: float = DEFAULT_CYCLE_COUNT_PRIOR,
    min_cycle_period: float | None = None,
) -> CycleMetrics:
    """统计旋回曲线的波峰与旋回个数，并计算三项评价指标。

    旋回个数 = (波峰数 - 1) + 两端不足一个周期的部分；两端的小数部分按
    最近一个完整周期的深度跨度线性估算并截断在 0 到 1 之间。
    """
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if depth.size != signal.size or depth.size < 3:
        raise ValueError("旋回曲线至少需要 3 个采样点。")
    if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(signal)):
        raise ValueError("深度与旋回曲线必须全部为有限数值。")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("深度必须严格递增。")
    if prior_count <= 0:
        raise ValueError("地质先验旋回个数必须为正。")

    peak_indices = _find_peaks(depth, signal, min_cycle_period)
    peak_depths = depth[peak_indices]
    peak_values = signal[peak_indices]
    cycle_count, bottom_fraction, top_fraction = _count_cycles(depth, peak_depths)

    return CycleMetrics(
        peak_depths=peak_depths,
        peak_values=peak_values,
        peak_numbers=np.arange(peak_depths.size, 0, -1, dtype=int),
        cycle_count=cycle_count,
        bottom_cycle_fraction=bottom_fraction,
        top_cycle_fraction=top_fraction,
        prior_count=float(prior_count),
        energy_index=oscillation_energy_index(depth, signal),
        smoothness_index=regularization_smoothness_index(depth, signal),
        count_proximity=cycle_count_proximity(cycle_count, prior_count),
    )


def oscillation_energy_index(depth: np.ndarray, signal: np.ndarray) -> float:
    """振荡能量指数：旋回振荡的深度平均积分能量，越大表示提取的天文旋回分量越强。"""
    return _depth_average(np.asarray(signal, dtype=np.float64) ** 2, depth)


def regularization_smoothness_index(depth: np.ndarray, signal: np.ndarray) -> float:
    """正则平滑指数：旋回曲线二阶差分平方的深度平均，惩罚拼接处的剧烈突变。"""
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if depth.size != signal.size or depth.size < 3:
        raise ValueError("正则平滑项至少需要 3 个采样点。")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("深度必须严格递增。")

    first_difference = np.diff(signal) / np.diff(depth)
    second_difference = 2.0 * np.diff(first_difference) / (depth[2:] - depth[:-2])
    interval_centers = 0.5 * (depth[2:] + depth[:-2])
    return _depth_average(second_difference ** 2, interval_centers)


def cycle_count_proximity(cycle_count: float, prior_count: float) -> float:
    """旋回个数接近度：与地质先验旋回个数完全吻合为 1，偏差达到先验值本身为 0。"""
    if prior_count <= 0:
        raise ValueError("地质先验旋回个数必须为正。")
    deviation = abs(float(cycle_count) - float(prior_count)) / float(prior_count)
    return float(1.0 - min(1.0, deviation))


def _find_peaks(
    depth: np.ndarray,
    signal: np.ndarray,
    min_cycle_period: float | None,
) -> np.ndarray:
    distance = 1
    if min_cycle_period and min_cycle_period > 0:
        depth_step = float(np.median(np.diff(depth)))
        distance = max(1, int(round(0.5 * float(min_cycle_period) / depth_step)))
    span = float(np.ptp(signal))
    prominence = PEAK_PROMINENCE_RATIO * span if span > 0 else None
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    return peaks


def _count_cycles(depth: np.ndarray, peak_depths: np.ndarray) -> tuple[float, float, float]:
    if peak_depths.size < 2:
        return float(peak_depths.size), 0.0, 0.0

    periods = np.diff(peak_depths)
    bottom_fraction = float(np.clip((depth[-1] - peak_depths[-1]) / periods[-1], 0.0, 1.0))
    top_fraction = float(np.clip((peak_depths[0] - depth[0]) / periods[0], 0.0, 1.0))
    cycle_count = float(periods.size) + bottom_fraction + top_fraction
    return cycle_count, bottom_fraction, top_fraction


def _depth_average(values: np.ndarray, depth: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    if values.size != depth.size or values.size < 2:
        raise ValueError("数值与深度必须是等长的一维数组。")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("深度必须严格递增。")
    return float(
        np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(depth)) / (depth[-1] - depth[0])
    )
