import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont
from PyQt6.QtGui import QAction, QGuiApplication, QImage


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


def gaussian_bandpass(signal, sample_spacing, center_frequency, relative_bandwidth=0.2):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError("Signal must be a one-dimensional array with at least two samples.")
    nyquist = 0.5 / sample_spacing
    if (
        sample_spacing <= 0
        or center_frequency <= 0
        or center_frequency >= nyquist
        or relative_bandwidth <= 0
    ):
        raise ValueError("Band-pass parameters must be positive and below the Nyquist frequency.")

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


def piecewise_gaussian_bandpass(signal, sample_spacing, centers, edges):
    signal = np.asarray(signal, dtype=float)
    centers = np.asarray(centers, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.size != centers.size + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Filter windows must have increasing depth boundaries.")

    filtered_windows = [
        gaussian_bandpass(signal, sample_spacing, float(center))
        for center in centers
    ]
    depth = np.arange(signal.size) * sample_spacing + edges[0]
    overlap = min(1.0, float(np.min(np.diff(edges))) * 0.2)
    result = np.zeros_like(signal)
    weight_sum = np.zeros_like(signal)

    for index, filtered in enumerate(filtered_windows):
        left, right = edges[index:index + 2]
        weights = np.ones(signal.size, dtype=float)
        if index == 0:
            weights[depth < left] = 0.0
        else:
            start, stop = left - overlap / 2.0, left + overlap / 2.0
            weights[depth < start] = 0.0
            mask = (depth >= start) & (depth <= stop)
            weights[mask] = (depth[mask] - start) / overlap

        if index == centers.size - 1:
            weights[depth > right] = 0.0
        else:
            start, stop = right - overlap / 2.0, right + overlap / 2.0
            mask = (depth >= start) & (depth <= stop)
            weights[mask] = 1.0 - (depth[mask] - start) / overlap
            weights[depth > stop] = 0.0

        result += filtered * weights
        weight_sum += weights

    return np.divide(result, weight_sum, out=np.zeros_like(result), where=weight_sum > 0)


# --------------------------
# 参数设置
# --------------------------
A = 2.0          # 统一振幅
REFERENCE_WAVELENGTH = 3.0
z_total = np.linspace(0, 30, 3000)

# ========= Case1: 恒定波长，恒定沉积速率 λ=3
def y1(z):
    return A * np.sin(2 * np.pi / REFERENCE_WAVELENGTH * z)

# ========= Case2：根据分段速率生成波形，分界深度为10和20
def y2(z_arr):
    local_rate = np.interp(z_arr, z_rate, r2)
    return rate_driven_waveform(z_arr, local_rate)

# ========= Case3：根据连续变化的速率生成波形
def y3(z):
    local_rate = np.interp(z, z_rate, r3)
    return rate_driven_waveform(z, local_rate)

# ========= 沉积速率曲线数据
z_rate = np.linspace(0,50,200)
r1 = np.full_like(z_rate, 2.0)
# Case2速率分段递增；在0–30 m显示范围内均值为2
r2 = np.zeros_like(z_rate)
r2[z_rate<=10]=1.0
r2[(z_rate>10)&(z_rate<=20)]=2.0
r2[z_rate>20]=3.0
r3 = 0.5 + (3.0 /30.0)* z_rate


def depth_average(values, depth):
    return float(
        np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(depth))
        / (depth[-1] - depth[0])
    )


common_mean_rate = depth_average(np.interp(z_total, z_rate, r1), z_total)
for rate_curve in (r2, r3):
    displayed_rate = np.interp(z_total, z_rate, rate_curve)
    rate_curve *= common_mean_rate / depth_average(displayed_rate, z_total)


def rate_driven_waveform(depth, local_rate):
    depth = np.asarray(depth, dtype=float)
    local_rate = np.asarray(local_rate, dtype=float)
    if depth.ndim != 1 or local_rate.shape != depth.shape or depth.size < 2:
        raise ValueError("Depth and rate must be matching one-dimensional arrays.")
    if np.any(local_rate <= 0) or np.any(np.diff(depth) <= 0):
        raise ValueError("Rate must be positive and depth must be strictly increasing.")

    mean_rate = depth_average(local_rate, depth)
    local_wavelength = REFERENCE_WAVELENGTH * local_rate / mean_rate
    local_frequency = 1.0 / local_wavelength
    phase_steps = (
        np.pi
        * (local_frequency[:-1] + local_frequency[1:])
        * np.diff(depth)
    )
    phase = np.concatenate(([0.0], np.cumsum(phase_steps)))
    return A * np.sin(phase)


# --------------------------
# 绘图 3行2列
# --------------------------
fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(12, 10),
    dpi=150,
    gridspec_kw={"width_ratios": (7, 3)},
)

# row 0
ax00 = axes[0,0]
ax01 = axes[0,1]
ax00.plot(z_total, y1(z_total), 'k-', lw=1.2)
#ax00.set_title("恒定速率", fontsize=9)
ax00.set_ylim(-2.5,2.5)
ax00.set_xlim(0,30)
#ax00.set_ylabel("Oscillation")
ax00.grid(alpha=0.3)

ax01.plot(z_rate, r1, "#0000FF", lw=1.5)
#ax01.set_title("沉积速率", fontsize=9)
ax01.set_xlim(0,30)
ax01.set_ylim(0,4)
ax01.grid(alpha=0.3)

# row1
ax10 = axes[1,0]
ax11 = axes[1,1]
ax10.plot(z_total, y2(z_total), 'k-', lw=1.2)
# 增加垂直虚线分割三段
ax10.axvline(x=10, linestyle='--', color='#000000', lw=1.0)
ax10.axvline(x=20, linestyle='--', color='#000000', lw=1.0)

#ax10.set_title("分段速率", fontsize=9)
ax10.set_ylim(-2.5,2.5)
ax10.set_xlim(0,30)
#ax10.set_ylabel("Oscillation")
ax10.grid(alpha=0.3)

ax11.plot(z_rate, r2, "#00FF00", lw=1.5)
ax11.set_xlim(0,30)
ax11.set_ylim(0,4)
ax11.grid(alpha=0.3)

# row2
ax20 = axes[2,0]
ax21 = axes[2,1]
ax20.plot(z_total, y3(z_total), 'k-', lw=1.2)
#ax20.set_title("变化速率", fontsize=9)
ax20.set_ylim(-2.5,2.5)
ax20.set_xlim(0,30)
#ax20.set_xlabel("Depth z")
#ax20.set_ylabel("Oscillation")
ax20.grid(alpha=0.3)

ax21.plot(z_rate, r3, "#FF0000", lw=1.5)
ax21.set_xlim(0,30)
ax21.set_ylim(0,4)
#ax21.set_xlabel("Depth z")
ax21.grid(alpha=0.3)

waveforms = (y1(z_total), y2(z_total), y3(z_total))
left_axes = (ax00, ax10, ax20)
filtered_lines = []
segmented_filtered_lines = []
varying_filtered_lines = []
reference_frequency = 1.0 / REFERENCE_WAVELENGTH
sample_spacing = float(z_total[1] - z_total[0])
segmented_rates = np.asarray((1.0, 2.0, 3.0)) * common_mean_rate / 2.0
segmented_center_frequencies = common_mean_rate / (
    REFERENCE_WAVELENGTH * segmented_rates
)
segmented_edges = np.asarray((0.0, 10.0, 20.0, 30.0))
varying_centers = np.linspace(1.0, 29.0, 15)
varying_rate = np.interp(z_total, z_rate, r3)
varying_center_frequencies = common_mean_rate / (
    REFERENCE_WAVELENGTH * np.interp(varying_centers, z_total, varying_rate)
)
varying_edges = np.linspace(0.0, 30.0, varying_centers.size + 1)
for waveform, left_ax in zip(
    waveforms,
    left_axes,
):
    filtered_waveform = gaussian_bandpass(
        waveform,
        sample_spacing=float(z_total[1] - z_total[0]),
        center_frequency=reference_frequency,
    )
    fixed_rmse = float(np.sqrt(np.mean((waveform - filtered_waveform) ** 2)))
    filtered_line, = left_ax.plot(
        z_total,
        filtered_waveform,
        color="#0000FF",
        lw=1.2,
        label=f"固定滤波 RMSE={fixed_rmse:.3f}",
    )
    filtered_lines.append(filtered_line)

    segmented_waveform = piecewise_gaussian_bandpass(
        waveform,
        sample_spacing,
        segmented_center_frequencies,
        segmented_edges,
    )
    segmented_rmse = float(np.sqrt(np.mean((waveform - segmented_waveform) ** 2)))
    segmented_line, = left_ax.plot(
        z_total,
        segmented_waveform,
        color="#00FF00",
        lw=1.2,
        visible=False,
        label=f"分段滤波 RMSE={segmented_rmse:.3f}",
    )
    segmented_filtered_lines.append(segmented_line)

    varying_waveform = piecewise_gaussian_bandpass(
        waveform,
        sample_spacing,
        varying_center_frequencies,
        varying_edges,
    )
    varying_rmse = float(np.sqrt(np.mean((waveform - varying_waveform) ** 2)))
    varying_line, = left_ax.plot(
        z_total,
        varying_waveform,
        color="#FF0000",
        lw=1.2,
        visible=False,
        label=f"变化滤波 RMSE={varying_rmse:.3f}",
    )
    varying_filtered_lines.append(varying_line)


def _refresh_filter_legends():
    for index, ax in enumerate(left_axes):
        visible_lines = [
            lines[index]
            for lines in (
                filtered_lines,
                segmented_filtered_lines,
                varying_filtered_lines,
            )
            if lines[index].get_visible()
        ]
        if visible_lines:
            ax.legend(handles=visible_lines, loc="upper right", fontsize=8)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()


_refresh_filter_legends()

for ax in axes.flat:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

plt.tight_layout()
#plt.savefig("spring_sediment_sketch.pdf", bbox_inches="tight")
#plt.savefig("spring_sediment_sketch.png", dpi=300, bbox_inches="tight")
manager = plt.get_current_fig_manager()


def _copy_current_view():
    image_data, (width, height) = manager.canvas.print_to_buffer()
    image = QImage(
        image_data,
        width,
        height,
        width * 4,
        QImage.Format.Format_RGBA8888,
    ).copy()
    QGuiApplication.clipboard().setImage(image)


copy_action = QAction("复制", manager.window)
copy_action.setToolTip("将当前绘图视图复制到剪贴板")
copy_action.triggered.connect(_copy_current_view)
manager.toolbar.addSeparator()
manager.toolbar.addAction(copy_action)


def _set_visibility(lines, visible):
    for line in lines:
        line.set_visible(visible)
    _refresh_filter_legends()
    manager.canvas.draw_idle()


filter_visibility_action = QAction("固定滤波", manager.window)
filter_visibility_action.setCheckable(True)
filter_visibility_action.setChecked(True)
filter_visibility_action.toggled.connect(
    lambda visible: _set_visibility(filtered_lines, visible)
)
filter_visibility_action.setToolTip("切换固定周期滤波结果显示")
segmented_filter_action = QAction("分段滤波", manager.window)
segmented_filter_action.setCheckable(True)
segmented_filter_action.toggled.connect(
    lambda visible: _set_visibility(segmented_filtered_lines, visible)
)
segmented_filter_action.setToolTip("切换三段式滤波结果显示")
varying_filter_action = QAction("变化滤波", manager.window)
varying_filter_action.setCheckable(True)
varying_filter_action.toggled.connect(
    lambda visible: _set_visibility(varying_filtered_lines, visible)
)
varying_filter_action.setToolTip("切换多窗口变化滤波结果显示")
manager.toolbar.addSeparator()
manager.toolbar.addAction(filter_visibility_action)
manager.toolbar.addAction(segmented_filter_action)
manager.toolbar.addAction(varying_filter_action)
manager.window.showMaximized()
plt.show()
