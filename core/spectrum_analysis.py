import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.signal.windows import dpss
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2

from core.data_preprocess import min_max_normalize


def _ar1_red_noise_threshold(freq: np.ndarray, signal: np.ndarray, dx: float):
    """根据AR1红噪声模型生成随频率递减的置信阈值曲线。"""
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    if sig.size < 3:
        return np.full_like(freq, np.nan, dtype=float)

    r = np.corrcoef(sig[:-1], sig[1:])[0, 1]
    r = float(np.clip(r, -0.999, 0.999))
    if not np.isfinite(r):
        r = 0.0

    sigma2 = np.nanvar(sig)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1.0

    freq = np.asarray(freq, dtype=float)
    red_noise = sigma2 * (1.0 - r ** 2) / (1.0 - 2.0 * r * np.cos(2.0 * np.pi * freq * dx) + r ** 2)
    red_noise = np.maximum(red_noise, 1e-12)
    return red_noise


def _validate_spectrum_signal(signal: np.ndarray) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    if not np.all(np.isfinite(sig)):
        raise ValueError("频谱输入包含NaN或Inf，请先完成显式预处理。")
    return sig


def get_depth_step(depth: np.ndarray) -> float:
    """计算深度平均采样步长"""
    dx_arr = np.diff(depth)
    dx = np.mean(dx_arr)
    return dx


def depth_fft_spectrum(depth: np.ndarray, signal: np.ndarray):
    """
    深度域FFT功率谱
    return: period_m, power, freq, confidence_curves
    """
    dx = get_depth_step(depth)
    sig = _validate_spectrum_signal(signal)
    n = len(sig)
    freq = fftfreq(n, d=dx)
    fft_vals = fft(sig)
    power = np.abs(fft_vals) ** 2 / max(n, 1)

    mask = freq > 0
    freq = freq[mask]
    power = power[mask]
    power_scale = np.nanmax(power)
    if not np.isfinite(power_scale) or power_scale <= 0:
        power_scale = 1.0
    power = power / power_scale
    period_m = 1.0 / freq

    red_noise = _ar1_red_noise_threshold(freq, sig, dx)
    red_noise = np.maximum(red_noise, 1e-12)
    red_noise = red_noise / power_scale

    confidence_curves = {}
    for conf in (0.99, 0.95, 0.90):
        chi2_value = chi2.ppf(conf, df=2)
        threshold = (chi2_value / 2.0) * red_noise
        # power 已做过归一化，阈值不再二次除以 power_scale，避免量级被错误压扁
        confidence_curves[str(conf)] = threshold

    return period_m, power, freq, confidence_curves


def depth_mtm_spectrum(depth: np.ndarray, signal: np.ndarray, NW=3):
    """
    MTM多锥功率谱（地层常用，抗噪优于FFT）
    :param NW: 时间半带宽积
    return: period_m, power, freq, confidence_curves
    """
    dx = get_depth_step(depth)
    sig = _validate_spectrum_signal(signal)
    n = len(sig)
    freq = fftfreq(n, d=dx)
    k = int(2 * NW) - 1
    tapers, eig = dpss(n, NW, k, return_ratios=True)

    mtm_power = np.zeros(n, dtype=float)
    for i in range(k):
        taper_sig = sig * tapers[i, :]
        fft_tap = fft(taper_sig)
        mtm_power += eig[i] * np.abs(fft_tap) ** 2
    mtm_power /= max(np.sum(eig), 1e-12)

    mask = freq > 0
    freq = freq[mask]
    power = mtm_power[mask]
    power_scale = np.nanmax(power)
    if not np.isfinite(power_scale) or power_scale <= 0:
        power_scale = 1.0
    power = power / power_scale
    period_m = 1.0 / freq

    red_noise = _ar1_red_noise_threshold(freq, sig, dx)
    red_noise = np.maximum(red_noise, 1e-12)
    red_noise = red_noise / power_scale

    confidence_curves = {}
    for conf in (0.99, 0.95, 0.90):
        chi2_value = chi2.ppf(conf, df=2)
        threshold = (chi2_value / 2.0) * red_noise
        # power 已做过归一化，阈值不再二次除以 power_scale，避免量级被错误压扁
        confidence_curves[str(conf)] = threshold

    return period_m, power, freq, confidence_curves


def gaussian_bandpass(depth: np.ndarray, signal: np.ndarray, thick_min: float, thick_max: float):
    """
    高斯带通滤波（频域高斯窗，地层旋回厚度区间滤波）
    :param thick_min: 最小旋回厚度 m
    :param thick_max: 最大旋回厚度 m
    :return: sig_filtered
    """
    dx = get_depth_step(depth)
    n = len(signal)
    freq = fftfreq(n, d=dx)
    fft_sig = fft(signal)

    # 目标中心频率与带宽
    f_low = 1.0 / thick_max
    f_high = 1.0 / thick_min
    f_cent = (f_low + f_high) / 2.0
    f_band = f_high - f_low

    # 高斯窗
    gauss_win = np.exp(-((freq - f_cent) ** 2) / (2 * (f_band / 2) ** 2))
    # 对称负频率部分
    gauss_win = gauss_win + np.exp(-((-freq - f_cent) ** 2) / (2 * (f_band / 2) ** 2))

    fft_filtered = fft_sig * gauss_win
    sig_filtered = np.fft.ifft(fft_filtered).real
    return min_max_normalize(sig_filtered)


def taner_bandpass(depth: np.ndarray, signal: np.ndarray, thick_min: float, thick_max: float, order=4):
    """
    Taner型带通滤波（巴特沃斯带通，修复Wn越界问题）
    :param thick_min: 最小旋回厚度 m
    :param thick_max: 最大旋回厚度 m
    :param order: 滤波器阶数
    :return: sig_filtered
    """
    dx = get_depth_step(depth)
    fs = 1.0 / dx
    nyq = 0.5 * fs
    min_resolvable = 2 * dx

    # 约束最小可分辨厚度，避免频率溢出
    thick_min = max(thick_min, min_resolvable)
    if thick_min >= thick_max:
        thick_max = thick_min + 0.1

    f_low = 1.0 / thick_max
    f_high = 1.0 / thick_min

    # 归一化频率限制在(1e-6,1-1e-6)
    wn1 = np.clip(f_low / nyq, 1e-6, 1 - 1e-6)
    wn2 = np.clip(f_high / nyq, 1e-6, 1 - 1e-6)
    Wn = [wn1, wn2]

    b, a = butter(order, Wn, btype="bandpass")
    sig_filtered = filtfilt(b, a, signal)
    return sig_filtered