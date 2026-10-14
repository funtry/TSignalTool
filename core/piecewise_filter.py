"""按深度分段的带通滤波：各段独立滤波后用线性过渡权重平滑拼接。"""

import numpy as np

TRANSITION_RATIO = 0.3
MAX_TRANSITION_WIDTH = 5.0


def gaussian_bandpass(
    signal: np.ndarray,
    sample_spacing: float,
    center_frequency: float,
    relative_bandwidth: float = 0.2,
) -> np.ndarray:
    """高斯带通滤波，两端反射填充以抑制边界效应。"""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("带通滤波至少需要两个有限采样点。")
    if sample_spacing <= 0 or relative_bandwidth <= 0 or relative_bandwidth >= 1:
        raise ValueError("采样间距必须为正，相对带宽必须位于 0 到 1 之间。")

    nyquist = 0.5 / sample_spacing
    if center_frequency <= 0 or center_frequency >= nyquist:
        raise ValueError("中心频率必须为正且低于奈奎斯特频率。")

    bandwidth = relative_bandwidth * center_frequency
    padded_signal = np.pad(signal, (signal.size, signal.size), mode="reflect")
    frequencies = np.fft.rfftfreq(padded_signal.size, d=sample_spacing)
    response = (
        np.exp(-0.5 * ((frequencies - center_frequency) / bandwidth) ** 2)
        + np.exp(-0.5 * ((frequencies + center_frequency) / bandwidth) ** 2)
    )
    response /= np.max(response)
    filtered = np.fft.irfft(
        np.fft.rfft(padded_signal) * response,
        n=padded_signal.size,
    )
    return filtered[signal.size:2 * signal.size]


def transition_width(edges: np.ndarray) -> float:
    """相邻分段之间的过渡带宽度（米），取最薄分段厚度的两成。"""
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("分段边界必须严格递增。")
    return float(min(MAX_TRANSITION_WIDTH, float(np.min(np.diff(edges))) * TRANSITION_RATIO))


def piecewise_gaussian_bandpass(
    depth: np.ndarray,
    signal: np.ndarray,
    edges: np.ndarray,
    frequencies: np.ndarray,
    relative_bandwidth: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """分段滤波并按线性权重拼接，返回等间距深度网格上的旋回曲线。

    每段都先对整条曲线做一次高斯带通，再在段边界两侧一半过渡带宽度内
    用互补的线性权重交叉淡化，因此拼接处连续且不改变幅值量级。
    """
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)

    if depth.size != signal.size or depth.size < 2:
        raise ValueError("深度与信号必须是等长的一维数组。")
    if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(signal)):
        raise ValueError("深度和信号必须全部为有限数值。")
    if np.any(np.diff(depth) <= 0):
        raise ValueError("深度必须严格递增。")
    if edges.size != frequencies.size + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("分段边界必须严格递增，且比局部频率多一个。")
    if edges[0] > depth[0] or edges[-1] < depth[-1]:
        raise ValueError("分段边界必须覆盖整个测井深度范围。")
    if not 0 < relative_bandwidth < 1:
        raise ValueError("相对带宽必须位于 0 到 1 之间。")

    depth_step = float(np.median(np.diff(depth)))
    uniform_depth = np.arange(depth[0], depth[-1] + depth_step * 0.5, depth_step)
    uniform_signal = np.interp(uniform_depth, depth, signal)

    nyquist = 0.5 / depth_step
    if np.any(frequencies <= 0) or np.any(frequencies >= nyquist):
        raise ValueError("局部频率必须为正且低于奈奎斯特频率。")

    filtered_windows = [
        gaussian_bandpass(uniform_signal, depth_step, float(frequency), relative_bandwidth)
        for frequency in frequencies
    ]

    overlap = transition_width(edges)
    result = np.zeros_like(uniform_signal)
    weight_sum = np.zeros_like(uniform_signal)

    for index, filtered in enumerate(filtered_windows):
        left, right = float(edges[index]), float(edges[index + 1])
        weights = np.ones(uniform_signal.size, dtype=np.float64)
        if index == 0:
            weights[uniform_depth < left] = 0.0
        else:
            start, stop = left - overlap / 2.0, left + overlap / 2.0
            weights[uniform_depth < start] = 0.0
            mask = (uniform_depth >= start) & (uniform_depth <= stop)
            weights[mask] = (uniform_depth[mask] - start) / overlap

        if index == frequencies.size - 1:
            weights[uniform_depth > right] = 0.0
        else:
            start, stop = right - overlap / 2.0, right + overlap / 2.0
            mask = (uniform_depth >= start) & (uniform_depth <= stop)
            weights[mask] = 1.0 - (uniform_depth[mask] - start) / overlap
            weights[uniform_depth > stop] = 0.0

        result += filtered * weights
        weight_sum += weights

    return uniform_depth, np.divide(
        result,
        weight_sum,
        out=np.zeros_like(result),
        where=weight_sum > 0,
    )
