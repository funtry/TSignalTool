import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont
from PyQt6.QtGui import QAction, QGuiApplication, QImage
from scipy.signal.windows import dpss

def _configure_chinese_font():
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    font_path = next(
        (
            fonts_dir / name
            for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf")
            if (fonts_dir / name).is_file()
        ),
        None,
    )
    if font_path is None:
        for family in (
            "Microsoft YaHei",
            "SimHei",
            "DengXian",
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "WenQuanYi Zen Hei",
        ):
            try:
                font_path = Path(
                    findfont(
                        FontProperties(family=family),
                        fallback_to_default=False,
                    )
                )
            except ValueError:
                continue
            break

    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        font_name = FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()

# 深度范围扩大到 120 m，采样间隔减小到 0.005 m
z = np.arange(0.0, 120.0 + 0.005 / 2, 0.005)
dz = z[1] - z[0]
fs = 1.0 / dz
A = 2.0
reference_wavelength = 3.0

# 三种速率曲线：恒定、分段恒定、连续变化
rate1 = np.full_like(z, 2.0)

# 分段周期：每段 40 m，局部速率差异增大
rate2 = np.where(z <= 40.0, 0.5, np.where(z <= 80.0, 2.0, 5.0))

# 连续变化周期：局部速率从 0.5 连续变化到 5.0
rate3 = 0.5 + (5.0 - 0.5) * z / 120.0


def depth_average(values, depth):
    return np.sum(
        0.5 * (values[:-1] + values[1:]) * np.diff(depth)
    ) / (depth[-1] - depth[0])


def make_waveform(rate):
    """按局部速率换算瞬时频率，并积分得到连续相位。"""
    mean_rate = depth_average(rate, z)
    local_frequency = mean_rate / (reference_wavelength * rate)  # cycles/m

    phase_steps = (
        np.pi
        * (local_frequency[:-1] + local_frequency[1:])
        * np.diff(z)
    )
    phase = np.concatenate(([0.0], np.cumsum(phase_steps)))
    return A * np.sin(phase)


signals = [
    ("恒定周期", make_waveform(rate1)),
    ("分段变化周期", make_waveform(rate2)),
    ("连续变化周期", make_waveform(rate3)),
]


def mtm_psd(x, sampling_rate, nw=2.5, k=None):
    """等权平均 DPSS 多锥窗功率谱估计。"""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    if k is None:
        k = int(2 * nw) - 1

    tapers, eigenvalues = dpss(
        x.size,
        NW=nw,
        Kmax=k,
        sym=False,
        return_ratios=True,
    )

    spectra = []
    for taper in tapers:
        tapered = x * taper
        spectrum = np.abs(np.fft.rfft(tapered)) ** 2
        spectrum /= sampling_rate * np.sum(taper**2)

        # 转为单边功率谱
        if x.size % 2 == 0:
            spectrum[1:-1] *= 2.0
        else:
            spectrum[1:] *= 2.0

        spectra.append(spectrum)

    # 按 DPSS 特征值加权平均
    spectra = np.asarray(spectra)
    psd = np.average(spectra, axis=0, weights=eigenvalues)
    frequency = np.fft.rfftfreq(x.size, d=1.0 / sampling_rate)
    return frequency, psd


# 理论频率，用于对照谱峰/谱带
mean_rate = depth_average(rate1, z)
f_constant = mean_rate / (reference_wavelength * 2.0)

segmented_mean_rate = depth_average(rate2, z)
f_segmented = segmented_mean_rate / (
    reference_wavelength * np.array([0.5, 2.0, 5.0])
)

varying_mean_rate = depth_average(rate3, z)
f_varying = varying_mean_rate / (
    reference_wavelength * np.array([rate3[0], rate3[-1]])
)

# 绘图：左侧为曲线序列和沉积速率，右侧为 MTM 频谱
fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(16, 10),
    dpi=150,
    sharex="col",
    gridspec_kw={"width_ratios": (7, 3, 6)},
)

rate_curves = (rate1, rate2, rate3)
rate_colors = ("#0000FF", "#00AA00", "#FF0000")

for row, ((title, signal), rate, rate_color) in enumerate(
    zip(signals, rate_curves, rate_colors)
):
    sequence_ax = axes[row, 0]
    rate_ax = axes[row, 1]
    spectrum_ax = axes[row, 2]

    sequence_ax.plot(z, signal, color="black", lw=1.0)
    sequence_ax.set_title(title)
    sequence_ax.set_xlim(z[0], z[-1])
    sequence_ax.set_ylim(-2.5, 2.5)
    sequence_ax.set_ylabel("幅值")
    sequence_ax.grid(alpha=0.3)

    rate_ax.plot(z, rate, color=rate_color, lw=1.2)
    rate_ax.set_xlim(z[0], z[-1])
    rate_ax.set_ylim(0, 6)
    rate_ax.set_ylabel("沉积速率")
    rate_ax.grid(alpha=0.3)

    frequency, psd = mtm_psd(signal, fs, nw=1.5, k=2)

    # 归一化到本条曲线的谱峰，便于比较谱形
    psd_db = 10.0 * np.log10(psd / np.max(psd) + 1e-12)
    spectrum_ax.plot(frequency, psd_db, color="navy", lw=1.2)
    spectrum_ax.set_title(f"{title} - MTM频谱")
    spectrum_ax.set_ylabel("相对功率谱 (dB)")
    spectrum_ax.set_ylim(-50, 2)
    spectrum_ax.set_xlim(0, 2.0)
    spectrum_ax.grid(alpha=0.3)

axes[0, 2].axvline(
    f_constant, color="red", linestyle="--",
    label=f"理论频率 {f_constant:.3f} cycles/m",
)
axes[0, 2].legend(fontsize=8)

for f in f_segmented:
    axes[1, 2].axvline(f, color="red", linestyle="--", alpha=0.65)

axes[1, 2].text(
    0.98, 0.92,
    "红虚线：三个区段的理论频率",
    transform=axes[1, 2].transAxes,
    ha="right",
    va="top",
    fontsize=8,
)

axes[2, 2].axvspan(
    min(f_varying), max(f_varying),
    color="orange",
    alpha=0.2,
    label="连续变化的理论频率范围",
)
axes[2, 2].legend(fontsize=8)

axes[2, 0].set_xlabel("深度 (m)")
axes[2, 1].set_xlabel("深度 (m)")
axes[2, 2].set_xlabel("空间频率 (cycles/m)")

plt.tight_layout()
plt.show()