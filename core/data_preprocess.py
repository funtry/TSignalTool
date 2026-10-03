# data_preprocess.py
import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from scipy.signal import detrend

def normalize_minmax(signal: np.ndarray):
    """
    MinMax归一化到[-1,1]，用于旋回曲线峰谷识别
    :param signal: 一维数组，旋回曲线
    :return: 归一化后曲线 [-1,1]
    """
    s_min = np.min(signal)
    s_max = np.max(signal)
    if np.isclose(s_max - s_min, 0):
        return np.zeros_like(signal)
    return 2 * (signal - s_min) / (s_max - s_min) - 1

def min_max_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Min-Max 归一化，映射到 [0, 1]
    :param signal: 一维原始信号
    :return: 归一化后信号
    """
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    sig_min = np.nanmin(sig)
    sig_max = np.nanmax(sig)
    if np.isclose(sig_max - sig_min, 0):
        return np.zeros_like(sig)
    norm_sig = (sig - sig_min) / (sig_max - sig_min)
    return norm_sig


def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Z-score 标准化（均值0，方差1），频谱分析首选
    :param signal: 一维原始信号
    :return: 标准化信号
    """
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    mu = np.nanmean(sig)
    std = np.nanstd(sig)
    if np.isclose(std, 0):
        return np.zeros_like(sig)
    norm_sig = (sig - mu) / std
    return norm_sig


def remove_outlier_3sigma(signal: np.ndarray, depth: np.ndarray = None, fill="median"):
    """
    3σ法则剔除异常值，支持深度序列同步填充
    :param signal: 一维信号
    :param depth: 对应深度数组（可选）
    :param fill: 填充方式 median / mean / nan
    :return: clean_sig, clean_depth(若传入depth)
    """
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    mu = np.nanmean(sig)
    sigma = np.nanstd(sig)
    upper = mu + 3 * sigma
    lower = mu - 3 * sigma
    mask = (sig >= lower) & (sig <= upper)

    clean_sig = sig.copy()
    if fill == "median":
        fill_val = np.nanmedian(sig[mask])
    elif fill == "mean":
        fill_val = mu
    else:
        fill_val = np.nan
    clean_sig[~mask] = fill_val

    if depth is not None:
        dep = np.asarray(depth).squeeze()
        return clean_sig, dep
    return clean_sig


def uniform_depth_interp(depth_raw: np.ndarray, sig_raw: np.ndarray, dx: float):
    """
    不等间距深度 → 固定步长均匀插值
    :param depth_raw: 原始不等深度
    :param sig_raw: 原始测井信号
    :param dx: 目标均匀步长(m)
    :return: depth_uniform, sig_uniform
    """
    d_raw = np.asarray(depth_raw).squeeze()
    s_raw = np.asarray(sig_raw).squeeze()
    d_min, d_max = d_raw.min(), d_raw.max()
    depth_new = np.arange(d_min, d_max + dx, dx)
    interp_func = interpolate.interp1d(d_raw, s_raw, kind="linear", fill_value="extrapolate")
    sig_new = interp_func(depth_new)
    return depth_new, sig_new


def log_transform(depth_raw: np.ndarray, sig_raw: np.ndarray):
    """
    对原始测井序列做自然对数变换
    :param depth_raw: 原始深度序列
    :param sig_raw: 原始测井信号
    :return: depth_new, sig_new
    """
    d_raw = np.asarray(depth_raw, dtype=np.float64).squeeze()
    s_raw = np.asarray(sig_raw, dtype=np.float64).squeeze()
    
    eps = np.finfo(np.float64).eps
    d_safe = np.maximum(d_raw, eps)
    s_safe = np.maximum(s_raw, eps)

    depth_new = np.log(d_safe)
    sig_new = np.log(s_safe)
    return depth_new, sig_new


def median_smooth(signal: np.ndarray, win_size: int = 5):
    """中值平滑去毛刺噪声"""
    sig = np.asarray(signal).squeeze()
    return median_filter(sig, size=win_size)


def detrend_linear(signal: np.ndarray):
    """线性去趋势 scipy内置"""
    sig = np.asarray(signal).squeeze()
    return detrend(sig, type="linear")


def detrend_poly(signal: np.ndarray, order: int = 2):
    """多项式拟合去趋势"""
    sig = np.asarray(signal).squeeze()
    x = np.arange(len(sig))
    coeff = np.polyfit(x, sig, order)
    trend = np.polyval(coeff, x)
    return sig - trend


def ar1_prewhiten(signal: np.ndarray):
    """
    AR(1)预白化，压制地层红噪声，频谱分析必备前置
    :param signal: 去趋势、归一化后信号
    :return: 预白化信号
    """
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    # 一阶自相关系数
    r = np.corrcoef(sig[:-1], sig[1:])[0, 1]
    if np.isnan(r) or np.abs(r) < 1e-6:
        return sig
    # 预白化公式 s_white[i] = s[i] - r*s[i-1]
    white = sig[1:] - r * sig[:-1]
    # 首点补原信号对齐长度
    white = np.concatenate([[sig[0]], white])
    return white


def full_pipeline(
    depth_raw: np.ndarray,
    sig_raw: np.ndarray,
    dx: float = 0.1,
    norm_type: str = "zscore",
    poly_order: int = 2,
    smooth_win: int = 5,
    do_prewhiten: bool = True
):
    """
    一键完整预处理流水线（旋回提取标准流程）
    1. 异常剔除 → 2.均匀插值 →3.中值平滑 →4.多项式去趋势 →5.归一化 →6.AR1预白化
    :param norm_type: "minmax" / "zscore"
    :return: depth_uniform, sig_processed
    """
    # 1 剔除异常
    sig_clean, dep_clean = remove_outlier_3sigma(sig_raw, depth_raw)
    # 2 均匀深度插值
    depth_uni, sig_uni = uniform_depth_interp(dep_clean, sig_clean, dx)
    # 3 中值平滑降噪
    sig_smooth = median_smooth(sig_uni, win_size=smooth_win)
    # 4 多项式去趋势
    sig_detrend = detrend_poly(sig_smooth, order=poly_order)
    # 5 归一化
    if norm_type.lower() == "minmax":
        sig_norm = min_max_normalize(sig_detrend)
    else:
        sig_norm = z_score_normalize(sig_detrend)
    # 6 AR1预白化
    if do_prewhiten:
        sig_out = ar1_prewhiten(sig_norm)
    else:
        sig_out = sig_norm
    return depth_uni, sig_out


# 测试入口
if __name__ == "__main__":
    # 模拟不等间距原始数据
    depth_raw = np.sort(np.random.uniform(3000, 4170, 1200))
    sig_raw = np.sin(depth_raw / 20) + 0.002 * depth_raw + np.random.randn(len(depth_raw)) * 0.4

    # 完整预处理
    d_proc, s_proc = full_pipeline(
        depth_raw=depth_raw,
        sig_raw=sig_raw,
        dx=0.2,
        norm_type="zscore",
        do_prewhiten=True
    )
    print(f"原始点数: {len(depth_raw)}  处理后均匀点数: {len(d_proc)}")
    print(f"处理后信号均值: {np.mean(s_proc):.3f} 标准差: {np.std(s_proc):.3f}")

    # 单独测试归一化
    test_sig = np.array([1, 2, 3, 4, 5])
    print("MinMax归一化:", min_max_normalize(test_sig))
    print("Z-score标准化:", z_score_normalize(test_sig))