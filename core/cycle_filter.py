import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt


def extract_cycle(
    depth: np.ndarray,
    signal: np.ndarray,
    center_wavelength: float = 3.5,
    relative_bandwidth: float = 0.4,
    sampling_interval: float | None = None,
) -> np.ndarray:
    """在均匀深度网格上提取指定中心波长附近的旋回分量。"""
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if depth.ndim != 1 or signal.ndim != 1 or depth.size != signal.size:
        raise ValueError("深度和信号必须是等长的一维数组。")
    if depth.size < 12 or not np.all(np.isfinite(depth)) or not np.all(np.isfinite(signal)):
        raise ValueError("带通滤波至少需要 12 个有限数据点。")
    if center_wavelength <= 0 or not 0 < relative_bandwidth < 1:
        raise ValueError("中心波长必须为正，带宽比例必须在 0 到 1 之间。")

    depth_step = float(sampling_interval or np.median(np.diff(depth)))
    if depth_step <= 0:
        raise ValueError("深度必须严格递增。")

    uniform_depth = np.arange(depth[0], depth[-1] + depth_step * 0.5, depth_step)
    uniform_signal = np.interp(uniform_depth, depth, signal)
    detrended_signal = detrend(uniform_signal, type="linear")

    wavelength_min = center_wavelength * (1 - relative_bandwidth / 2)
    wavelength_max = center_wavelength * (1 + relative_bandwidth / 2)
    nyquist = 0.5 / depth_step
    low_frequency = 1.0 / wavelength_max
    high_frequency = min(1.0 / wavelength_min, nyquist * 0.99)
    if low_frequency >= high_frequency:
        raise ValueError("采样间距过大，无法解析该中心波长的带通范围。")

    sos = butter(
        3,
        (low_frequency, high_frequency),
        btype="bandpass",
        fs=1.0 / depth_step,
        output="sos",
    )
    filtered = sosfiltfilt(sos, detrended_signal)
    return np.interp(depth, uniform_depth, filtered)