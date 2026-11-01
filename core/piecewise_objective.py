"""分段滤波的无量纲代价损失与边界软惩罚。"""

import numpy as np

BOUNDARY_MARGIN_RATIO = 0.1
BOUNDARY_PENALTY_WEIGHT = 0.15
COST_WEIGHT_ENERGY = 0.3
COST_WEIGHT_SMOOTHNESS = 0.4
COST_WEIGHT_CYCLE_COUNT = 0.3
COST_TARGET_CYCLE_COUNT = 36.0


def combined_objective_cost(
    energy_loss: float,
    smoothness_loss: float,
    cycle_count_loss: float,
    boundary_loss: float,
) -> float:
    """计算 J = 0.3 L_E + 0.4 L_S + 0.3 L_C + 0.15 P_b。"""
    return float(
        COST_WEIGHT_ENERGY * energy_loss
        + COST_WEIGHT_SMOOTHNESS * smoothness_loss
        + COST_WEIGHT_CYCLE_COUNT * cycle_count_loss
        + BOUNDARY_PENALTY_WEIGHT * boundary_loss
    )


def _mean_squared_second_difference(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    return float(np.mean(np.diff(values, n=2) ** 2))


def normalized_objective_losses(
    raw_depth: np.ndarray,
    raw_signal: np.ndarray,
    cycle_depth: np.ndarray,
    cycle_signal: np.ndarray,
    cycle_count: float,
    target_cycle_count: float,
) -> tuple[float, float, float]:
    """返回 [0, 1] 范围内的能量、平滑度和旋回数无量纲损失。

    能量和平滑度分别以原始曲线在滤波深度网格上的对应指标为固定参考尺度。
    """
    raw_depth = np.asarray(raw_depth, dtype=float).reshape(-1)
    raw_signal = np.asarray(raw_signal, dtype=float).reshape(-1)
    cycle_depth = np.asarray(cycle_depth, dtype=float).reshape(-1)
    cycle_signal = np.asarray(cycle_signal, dtype=float).reshape(-1)
    if raw_depth.size != raw_signal.size or raw_depth.size < 2:
        raise ValueError("原始深度和信号必须是等长数组，且至少包含两个采样点。")
    if cycle_depth.size != cycle_signal.size or cycle_depth.size < 2:
        raise ValueError("滤波深度和信号必须是等长数组，且至少包含两个采样点。")
    if target_cycle_count <= 0:
        raise ValueError("目标旋回个数必须为正数。")

    reference_signal = np.interp(cycle_depth, raw_depth, raw_signal)
    centered_reference = reference_signal - float(np.mean(reference_signal))
    centered_cycle = cycle_signal - float(np.mean(cycle_signal))
    reference_energy = float(np.mean(centered_reference**2))
    cycle_energy = float(np.mean(centered_cycle**2))
    if reference_energy > np.finfo(float).tiny:
        energy_loss = 1.0 - cycle_energy / (cycle_energy + reference_energy)
    else:
        energy_loss = 0.0 if cycle_energy <= np.finfo(float).tiny else 1.0

    reference_smoothness = _mean_squared_second_difference(reference_signal)
    cycle_smoothness = _mean_squared_second_difference(cycle_signal)
    if reference_smoothness > np.finfo(float).tiny:
        smoothness_loss = cycle_smoothness / (
            cycle_smoothness + reference_smoothness
        )
    else:
        smoothness_loss = (
            0.0 if cycle_smoothness <= np.finfo(float).tiny else 1.0
        )

    cycle_count_loss = min(
        abs(float(cycle_count) - float(target_cycle_count)) / float(target_cycle_count),
        1.0,
    )
    return energy_loss, smoothness_loss, cycle_count_loss


def boundary_penalty(
    parameters: np.ndarray,
    depth_top: float,
    depth_bottom: float,
    minimum_segment_thickness: float,
    minimum_wavelength: float,
    maximum_wavelength: float,
) -> float:
    """对进入可行取值范围边缘 10% 区间的参数施加二次软惩罚。"""
    values = np.asarray(parameters, dtype=float).reshape(-1)
    if values.size < 3 or values.size % 2 == 0:
        raise ValueError("边界惩罚需要奇数个参数：分界深度数应比波长数少一个。")
    if not 0.0 < BOUNDARY_MARGIN_RATIO < 0.5:
        raise ValueError("边界缓冲比例必须位于 0 到 0.5 之间。")

    penalties = []
    segment_count = (values.size + 1) // 2
    boundary_count = segment_count - 1

    def append_interval_penalty(samples, lower, upper):
        width = upper - lower
        if width <= 0:
            return
        distance = np.minimum((samples - lower) / width, (upper - samples) / width)
        relative_distance = np.clip(distance / BOUNDARY_MARGIN_RATIO, 0.0, 1.0)
        penalties.extend(((1.0 - relative_distance) ** 2).tolist())

    span = float(depth_bottom) - float(depth_top)
    maximum_segment_thickness = span - (segment_count - 1) * float(
        minimum_segment_thickness
    )
    thickness_range = maximum_segment_thickness - float(minimum_segment_thickness)
    if thickness_range > np.finfo(float).eps * max(abs(span), 1.0):
        top_to_bottom_boundaries = np.asarray(
            [depth_top, *values[:boundary_count][::-1], depth_bottom],
            dtype=float,
        )
        thicknesses = np.diff(top_to_bottom_boundaries)
        append_interval_penalty(
            thicknesses,
            float(minimum_segment_thickness),
            maximum_segment_thickness,
        )

    append_interval_penalty(
        values[boundary_count:],
        float(minimum_wavelength),
        float(maximum_wavelength),
    )
    return float(np.mean(penalties)) if penalties else 0.0
