"""分段滤波窗口：测井曲线、分段局部周期与拼接旋回曲线三列对照。"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cycle_metrics import (
    DEFAULT_CYCLE_COUNT_PRIOR,
    analyze_cycle_curve,
    oscillation_energy_index,
)
from core.piecewise_filter import piecewise_gaussian_bandpass, transition_width
from core.well_log_loader import load_well_log

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SEGMENT_COUNT = 4
PARAMETER_NAMES = ("z1", "z2", "z3", "x1", "x2", "x3", "x4")
PARAMETER_COUNT = len(PARAMETER_NAMES)
MIN_WAVELENGTH = 5.0
MAX_WAVELENGTH = 20.0
MIN_SEGMENT_THICKNESS = 15.0
RELATIVE_BANDWIDTH = 0.2
PERIOD_AXIS_LIMITS = (3.0, 15.0)
SEGMENT_COLORS = ("#7FFF00", "#FF8C00", "#FF00FF",  "#8B4513")
RAW_LOG_COLOR = "#C8C8C8"
CYCLE_COLOR = "#FF0000"
PEAK_COLOR = "#00FF00"
OVERLAY_COLOR = "#800080"

# 手动设置预设 7 参数 (z1, z2, z3, x1, x2, x3, x4) 的位置：填写后，加载曲线与
# 「恢复默认参数」都会采用这组参数；设为 None 则按深度跨度自动生成。
#
# 下面的预设逼近这套三段划分方案：
#   深度位置 5272 / 5389 / 5456 / 5505 m（自顶界到底界）
#   沉积速率 0.16 / 0.10 / 0.17 cycles/m（自顶界到底界）
# 倒数为分段恒定波长：顶段 5272~5389 m -> 6.25 m/旋回，中段 5389~5456 m ->
# 10.00 m/旋回，底段 5456~5505 m -> 5.8824 m/旋回。
# 四段模型可精确表示该方案：把最厚的 5272~5389 m 段在 5330.5 m 处对半拆开，
# 拆出的两段沿用同一波长，因此波长与速率误差均为零（拆分点可在 5287~5374 m
# 之间任取，越界最小段厚 15 m 时结果不变）。
PRESET_PARAMETERS = (
    5456.0,  # z1 底段与中段分界（m）
    5389.0,  # z2 中段与顶段分界（m）
    5330.5,  # z3 顶段内部的拆分位置（m，可任选）
    1.0 / 0.17,  # x1 底段 5456~5505 m
    1.0 / 0.10,  # x2 中段 5389~5456 m
    1.0 / 0.16,  # x3 5330.5~5389 m
    1.0 / 0.16,  # x4 5272~5330.5 m
)


class ParameterDialog(QDialog):
    """编辑 7 个参数：内部三段分界深度 z1~z3 与四段波长 x1~x4。

    z0 为底界、z4 为顶界，直接取自测井数据的深度范围；x 为米/旋回。
    """

    def __init__(self, boundaries, wavelengths, minimum_thickness, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑分段参数（7 个）")
        self.depth_bottom = float(boundaries[0])
        self.depth_top = float(boundaries[-1])
        self.minimum_thickness = float(minimum_thickness)

        form = QFormLayout(self)
        self.depth_spins = []
        self.wavelength_spins = []
        self.depth_labels = []
        self.wavelength_labels = []

        for index in range(SEGMENT_COUNT - 1):
            depth_spin = QDoubleSpinBox()
            depth_spin.setRange(self.depth_top, self.depth_bottom)
            depth_spin.setDecimals(3)
            depth_spin.setSingleStep(0.5)
            depth_spin.setValue(float(boundaries[index + 1]))
            depth_spin.valueChanged.connect(self._refresh_labels)
            depth_label = QLabel()
            form.addRow(depth_label, depth_spin)
            self.depth_spins.append(depth_spin)
            self.depth_labels.append(depth_label)

        for index in range(SEGMENT_COUNT):
            wavelength_spin = QDoubleSpinBox()
            wavelength_spin.setRange(MIN_WAVELENGTH, MAX_WAVELENGTH)
            wavelength_spin.setDecimals(3)
            wavelength_spin.setSingleStep(0.5)
            wavelength_spin.setValue(float(wavelengths[index]))
            wavelength_spin.valueChanged.connect(self._refresh_labels)
            wavelength_label = QLabel()
            form.addRow(wavelength_label, wavelength_spin)
            self.wavelength_spins.append(wavelength_spin)
            self.wavelength_labels.append(wavelength_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._refresh_labels()

    def boundaries(self):
        """按 z0(底界) → z4(顶界) 返回五个深度位置。"""
        return [
            self.depth_bottom,
            *[spin.value() for spin in self.depth_spins],
            self.depth_top,
        ]

    def wavelengths(self):
        return [spin.value() for spin in self.wavelength_spins]

    def _thicknesses(self):
        """按 x1~x4 顺序返回各段厚度（米）。"""
        return [-value for value in np.diff(self.boundaries())]

    def _refresh_labels(self, _value=None):
        thicknesses = self._thicknesses()
        for index, label in enumerate(self.depth_labels):
            label.setText(
                f"分界深度 z{index + 1}（m，x{index + 1} 段厚 {thicknesses[index]:.2f} m）"
            )
        for index, label in enumerate(self.wavelength_labels):
            wavelength = self.wavelength_spins[index].value()
            label.setText(
                f"波长 x{index + 1}（m/旋回，段厚 {thicknesses[index]:.2f} m，"
                f"1/x = {1.0 / wavelength:.4f} cycles/m）"
            )

    def _validate(self):
        thicknesses = self._thicknesses()
        if np.any(thicknesses <= 0):
            QMessageBox.warning(self, "参数无效", "z1、z2、z3 必须自底界向顶界严格递减。")
            return
        if float(np.min(thicknesses)) + 1e-6 < self.minimum_thickness:
            QMessageBox.warning(
                self,
                "参数无效",
                f"每段持续深度不得小于 {self.minimum_thickness:g} 米。",
            )
            return
        self.accept()


class PiecewiseFilterWidget(QWidget):
    """左侧原始测井曲线，中间分段局部周期，右侧分段滤波拼接旋回曲线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分段滤波")

        self.depth = np.asarray([], dtype=float)
        self.signal = np.asarray([], dtype=float)
        self.boundaries = []
        self.wavelengths = []
        self.preset_note = ""
        self.cycle_depth = np.asarray([], dtype=float)
        self.cycle_signal = np.asarray([], dtype=float)
        self.overlay_depth = np.asarray([], dtype=float)
        self.overlay_signal = np.asarray([], dtype=float)
        self.overlay_name = ""
        self.metrics = None

        self.figure = Figure(figsize=(15, 8), dpi=100)
        self.log_axes, self.period_axes, self.cycle_axes = self.figure.subplots(1, 3)
        self.figure.subplots_adjust(
            left=0.055,
            right=0.985,
            top=0.93,
            bottom=0.09,
            wspace=0.26,
        )
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-size: 13px; font-weight: 600;")

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.status_label)

        self.canvas.mpl_connect("button_press_event", self._show_context_menu)
        self._render()

    # ---------------------------------------------------------------- 数据加载

    def load_curve_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载测井曲线",
            os.getcwd(),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if file_path:
            self.load_curve(file_path)

    def load_curve(self, file_path):
        try:
            depth, signal = load_well_log(file_path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "加载失败", str(error))
            return False

        self.depth = depth
        self.signal = signal
        self._reset_parameters()
        self._clip_overlay_curve()
        self._refresh_result()
        return True

    # ---------------------------------------------------------------- 分段参数

    def _edges(self):
        """按深度递增（顶界 z4 → 底界 z0）返回分段边界，供滤波函数使用。"""
        return list(reversed(self.boundaries))

    def _frequencies(self):
        """按 _edges 的相邻区间顺序返回局部波数（cycles/m）。"""
        return list(reversed([1.0 / value for value in self.wavelengths]))

    def _minimum_thickness(self):
        span = float(self.depth[-1] - self.depth[0])
        return min(MIN_SEGMENT_THICKNESS, span / SEGMENT_COUNT)

    def _reset_parameters(self):
        """优先采用代码中手动设置的 PRESET_PARAMETERS，否则按深度跨度自动生成。"""
        if PRESET_PARAMETERS is not None:
            try:
                self._assign_parameters(PRESET_PARAMETERS)
                self.preset_note = "（已采用代码中预设的 7 参数）"
                return
            except ValueError:
                self.preset_note = "（预设 7 参数不适用于当前深度范围，已按跨度自动生成）"
        else:
            self.preset_note = ""

        bottom = float(self.depth[-1])
        top = float(self.depth[0])
        span = bottom - top
        self.boundaries = [
            bottom - span * index / SEGMENT_COUNT for index in range(SEGMENT_COUNT + 1)
        ]
        wavelength = float(
            np.clip(span / DEFAULT_CYCLE_COUNT_PRIOR, MIN_WAVELENGTH, MAX_WAVELENGTH)
        )
        self.wavelengths = [wavelength] * SEGMENT_COUNT

    def reset_parameters(self):
        if self.depth.size == 0:
            return
        self._reset_parameters()
        self._refresh_result()

    def get_parameters(self):
        """返回 7 个自由参数 (z1, z2, z3, x1, x2, x3, x4)；z0/z4 由数据底顶界给定。"""
        return np.asarray([*self.boundaries[1:-1], *self.wavelengths], dtype=float)

    def _assign_parameters(self, parameters):
        """校验并写入 (z1, z2, z3, x1, x2, x3, x4)，不触发滤波计算。"""
        values = np.asarray(parameters, dtype=float).reshape(-1)
        if values.size != PARAMETER_COUNT:
            raise ValueError(f"需要 {PARAMETER_COUNT} 个参数：{'、'.join(PARAMETER_NAMES)}。")
        if self.depth.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("参数必须为有限数值，且需先加载测井曲线。")

        boundaries = [float(self.depth[-1]), *values[:3].tolist(), float(self.depth[0])]
        wavelengths = np.clip(values[3:], MIN_WAVELENGTH, MAX_WAVELENGTH)
        thicknesses = -np.diff(boundaries)
        if np.any(thicknesses <= 0) or float(np.min(thicknesses)) < self._minimum_thickness():
            raise ValueError(
                "z1、z2、z3 需自底界向顶界递减，且每段持续深度不小于 "
                f"{self._minimum_thickness():g} 米。"
            )

        self.boundaries = boundaries
        self.wavelengths = [float(value) for value in wavelengths]

    def set_parameters(self, parameters):
        """按 (z1, z2, z3, x1, x2, x3, x4) 顺序写入参数并重新滤波。

        超出 [MIN_WAVELENGTH, MAX_WAVELENGTH] 的波长会被夹取到区间内；
        分界深度越界或段厚小于下限则抛出 ValueError。
        """
        self._assign_parameters(parameters)
        self._refresh_result()
        return True

    def edit_parameters(self):
        if not self.wavelengths:
            return
        dialog = ParameterDialog(
            self.boundaries,
            self.wavelengths,
            self._minimum_thickness(),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.boundaries = dialog.boundaries()
        self.wavelengths = dialog.wavelengths()
        self._refresh_result()

    # ---------------------------------------------------------------- 滤波计算

    def _refresh_result(self):
        self.cycle_depth = np.asarray([], dtype=float)
        self.cycle_signal = np.asarray([], dtype=float)
        self.metrics = None
        if self.depth.size >= 2 and self.wavelengths:
            try:
                self.cycle_depth, self.cycle_signal = piecewise_gaussian_bandpass(
                    self.depth,
                    self.signal,
                    self._edges(),
                    self._frequencies(),
                    relative_bandwidth=RELATIVE_BANDWIDTH,
                )
                self.metrics = analyze_cycle_curve(
                    self.cycle_depth,
                    self.cycle_signal,
                    prior_count=DEFAULT_CYCLE_COUNT_PRIOR,
                    min_cycle_period=float(np.min(self.wavelengths)),
                )
            except ValueError as error:
                QMessageBox.warning(self, "分段滤波失败", str(error))
        self._render()

    def export_cycle_curve(self):
        if self.cycle_signal.size == 0:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出旋回曲线",
            os.path.join(os.getcwd(), "分段滤波旋回曲线.txt"),
            "文本文件 (*.txt)",
        )
        if not file_path:
            return
        try:
            np.savetxt(
                file_path,
                np.column_stack((self.cycle_depth, self.cycle_signal)),
                fmt="%.10f",
            )
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))

    # ---------------------------------------------------------------- 叠加曲线

    def overlay_curve_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "叠加曲线",
            os.getcwd(),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if file_path:
            self.overlay_curve(file_path)

    def overlay_curve(self, file_path):
        """加载两列 TXT 深度曲线，只保留与当前深度范围重叠的部分用于叠加显示。"""
        try:
            data = np.loadtxt(file_path, dtype=float, ndmin=2)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "叠加曲线失败",
                f"读取失败：{error}\n文件应为无表头的两列数值：深度、曲线值。",
            )
            return False

        if (
            data.ndim != 2
            or data.shape[0] < 2
            or data.shape[1] < 2
            or not np.all(np.isfinite(data[:, :2]))
        ):
            QMessageBox.warning(
                self,
                "叠加曲线失败",
                "文件至少需要两行有限数值，前两列为深度和曲线值。",
            )
            return False

        order = np.argsort(data[:, 0], kind="stable")
        depth, values = data[order, 0], data[order, 1]
        keep = np.concatenate(([True], np.diff(depth) > 0))
        depth, values = depth[keep], values[keep]

        within = (depth >= float(self.depth[0])) & (depth <= float(self.depth[-1]))
        if int(np.count_nonzero(within)) < 2:
            QMessageBox.warning(
                self,
                "叠加曲线失败",
                f"{Path(file_path).name} 与当前深度范围 "
                f"{self.depth[0]:g} ~ {self.depth[-1]:g} m 没有重叠。",
            )
            return False

        self.overlay_depth = depth[within]
        self.overlay_signal = values[within]
        self.overlay_name = Path(file_path).name
        self._render()
        return True

    def remove_overlay_curve(self):
        self.overlay_depth = np.asarray([], dtype=float)
        self.overlay_signal = np.asarray([], dtype=float)
        self.overlay_name = ""
        self._render()

    def _clip_overlay_curve(self):
        if not self.overlay_signal.size:
            return
        within = (self.overlay_depth >= float(self.depth[0])) & (
            self.overlay_depth <= float(self.depth[-1])
        )
        if int(np.count_nonzero(within)) >= 2:
            self.overlay_depth = self.overlay_depth[within]
            self.overlay_signal = self.overlay_signal[within]
        else:
            self.overlay_depth = np.asarray([], dtype=float)
            self.overlay_signal = np.asarray([], dtype=float)
            self.overlay_name = ""

    # ---------------------------------------------------------------- 交互

    def _show_context_menu(self, event):
        if event.button != MouseButton.RIGHT or event.inaxes is None:
            return

        if event.inaxes is self.log_axes:
            handlers = [("加载测井曲线 TXT", self.load_curve_dialog)]
        elif event.inaxes is self.period_axes:
            handlers = (
                [
                    ("编辑 7 个参数…", self.edit_parameters),
                    ("恢复默认参数", self.reset_parameters),
                ]
                if self.wavelengths
                else []
            )
        elif event.inaxes is self.cycle_axes:
            handlers = []
            if self.depth.size >= 2:
                handlers.append(("叠加曲线", self.overlay_curve_dialog))
            if self.overlay_signal.size:
                handlers.append(("移除叠加曲线", self.remove_overlay_curve))
            if self.cycle_signal.size:
                handlers.append(("导出旋回曲线 TXT", self.export_cycle_curve))
        else:
            return

        if not handlers:
            return

        menu = QMenu(self)
        actions = [(menu.addAction(label), handler) for label, handler in handlers]
        chosen = menu.exec(self.canvas.mapToGlobal(event.guiEvent.pos()))
        for action, handler in actions:
            if chosen == action:
                handler()
                return

    # ---------------------------------------------------------------- 绘制

    def _render(self):
        self._render_log_axes()
        self._render_period_axes()
        self._render_cycle_axes()
        self._update_metrics()
        self._update_status()
        self.canvas.draw_idle()

    def _prepare_axes(self, axes, title, xlabel):
        axes.clear()
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel("深度 (m)")
        axes.grid(True, alpha=0.25, linestyle="--")
        if self.depth.size >= 2:
            axes.set_ylim(float(self.depth[-1]), float(self.depth[0]))

    @staticmethod
    def _draw_placeholder(axes, text):
        axes.text(
            0.5,
            0.5,
            text,
            transform=axes.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=9,
        )

    def _render_log_axes(self):
        self._prepare_axes(self.log_axes, "测井曲线（右键加载 TXT）", "信号幅值")
        if self.depth.size == 0:
            self._draw_placeholder(self.log_axes, "右键加载两列 TXT：深度、曲线值")
            return
        self.log_axes.plot(self.signal, self.depth, color="#0000FF", linewidth=1.0)

    def _render_period_axes(self):
        axes = self.period_axes
        self._prepare_axes(axes, "分段局部周期", "局部周期 x（m/旋回）")
        axes.set_xlim(*PERIOD_AXIS_LIMITS)
        if not self.wavelengths:
            self._draw_placeholder(axes, "加载测井曲线后显示分段局部周期")
            return

        for guide in (MIN_WAVELENGTH, MAX_WAVELENGTH):
            axes.axvline(guide, color="#999999", linewidth=0.8, linestyle=":")

        overlap = transition_width(self._edges())
        axis_middle = 0.5 * (PERIOD_AXIS_LIMITS[0] + PERIOD_AXIS_LIMITS[1])
        for index, wavelength in enumerate(self.wavelengths):
            color = SEGMENT_COLORS[index % len(SEGMENT_COLORS)]
            wavelength = float(wavelength)
            top = float(self.boundaries[index + 1])
            bottom = float(self.boundaries[index])
            axes.plot(
                [wavelength, wavelength],
                [top, bottom],
                color=color,
                linewidth=2.6,
                solid_capstyle="butt",
            )
            left_aligned = wavelength <= axis_middle
            axes.annotate(
                f"x{index + 1}={wavelength:.2f} m",
                xy=(wavelength, 0.5 * (top + bottom)),
                xytext=(6 if left_aligned else -6, 0),
                textcoords="offset points",
                ha="left" if left_aligned else "right",
                color=color,
                fontsize=9,
                va="center",
            )
            if index:
                axes.plot(
                    [float(self.wavelengths[index - 1]), wavelength],
                    [bottom, bottom],
                    color="#808080",
                    linewidth=1.0,
                    linestyle="--",
                )

        transition_patch = None
        for boundary in self._edges()[1:-1]:
            transition_patch = axes.axhspan(
                boundary - overlap / 2.0,
                boundary + overlap / 2.0,
                color="#000000",
                alpha=0.08,
            )
        axes.legend(
            handles=[transition_patch],
            labels=[f"拼接过渡带 {overlap:.2f} m"],
            loc="lower left",
            fontsize=8,
            framealpha=0.85,
        )

    def _render_cycle_axes(self):
        axes = self.cycle_axes
        self._prepare_axes(axes, "分段滤波拼接旋回曲线", "信号幅值")
        if self.cycle_signal.size == 0:
            self._draw_placeholder(axes, "分段滤波后显示拼接旋回曲线")
            return
        axes.plot(
            self.signal,
            self.depth,
            color=RAW_LOG_COLOR,
            linewidth=0.8,
        )
        axes.plot(
            self.cycle_signal,
            self.cycle_depth,
            color=CYCLE_COLOR,
            linewidth=1.15,
        )
        if self.overlay_signal.size:
            axes.plot(
                self.overlay_signal,
                self.overlay_depth,
                color=OVERLAY_COLOR,
                linewidth=0.7,
            )
        self._draw_cycle_peaks(axes)
        axes.legend(handles=self._cycle_legend_handles(), loc="upper right", fontsize=8)

    def _draw_cycle_peaks(self, axes):
        """用横向虚线标出波峰深度，虚线右侧标注自下向上从 1 开始的波峰编号。"""
        metrics = self.metrics
        if metrics is None or metrics.peak_depths.size == 0:
            return

        transform = axes.get_yaxis_transform()
        for depth, number in zip(metrics.peak_depths, metrics.peak_numbers):
            axes.axhline(
                depth,
                color=PEAK_COLOR,
                linewidth=0.8,
                linestyle="--",
                alpha=0.75,
            )
            axes.text(
                0.98,
                depth,
                f"{number}",
                transform=transform,
                ha="right",
                va="center",
                fontsize=8,
                color=PEAK_COLOR,
            )
        axes.plot(
            metrics.peak_values,
            metrics.peak_depths,
            linestyle="none",
            marker="o",
            markersize=3.2,
            color=PEAK_COLOR,
        )
        axes.text(
            0.02,
            float(self.depth[0]),
            f"上端 {metrics.top_cycle_fraction:.2f} 旋回",
            transform=transform,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#808080",
        )
        axes.text(
            0.02,
            float(self.depth[-1]),
            f"下端 {metrics.bottom_cycle_fraction:.2f} 旋回",
            transform=transform,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="#808080",
        )

    def _cycle_legend_handles(self):
        handles = [
            Line2D([], [], color=RAW_LOG_COLOR, linewidth=0.8, label="原始测井曲线"),
            Line2D([], [], color=CYCLE_COLOR, linewidth=1.15, label="分段滤波拼接旋回曲线"),
        ]
        if self.overlay_signal.size:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=OVERLAY_COLOR,
                    linewidth=0.7,
                    label=f"叠加曲线（{self.overlay_name}）",
                )
            )
        if self.metrics is not None and self.metrics.peak_depths.size:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=PEAK_COLOR,
                    linewidth=0.8,
                    linestyle="--",
                    marker="o",
                    markersize=3.2,
                    label="波峰深度与编号",
                )
            )
        return handles

    def _update_metrics(self):
        if self.metrics is None:
            self.metrics_label.setText("加载测井曲线后统计波峰、旋回个数并计算三项指标。")
            return
        metrics = self.metrics
        raw_on_grid = np.interp(self.cycle_depth, self.depth, self.signal)
        raw_energy = oscillation_energy_index(self.cycle_depth, raw_on_grid)
        ratio = metrics.energy_index / raw_energy if raw_energy > 0 else 0.0
        self.metrics_label.setText(
            f"旋回个数 {metrics.cycle_count:.2f} / 先验 {metrics.prior_count:.2f}"
            f"（波峰 {metrics.peak_depths.size} 个；下端 {metrics.bottom_cycle_fraction:.2f}、"
            f"上端 {metrics.top_cycle_fraction:.2f} 个旋回）｜"
            f"振荡能量指数 {metrics.energy_index:.4f}"
            f"（占原始曲线 {ratio * 100.0:.2f}%，越大越好）｜"
            f"正则平滑指数 {metrics.smoothness_index:.6f}（越小越好）｜"
            f"旋回个数接近度 {metrics.count_proximity:.4f}（越接近 1 越好）"
        )

    def _parameter_text(self):
        boundaries = " ".join(
            f"z{index}={value:.2f}" for index, value in enumerate(self.boundaries)
        )
        wavelengths = " ".join(
            f"x{index + 1}={value:.2f}" for index, value in enumerate(self.wavelengths)
        )
        return f"7 参数：{boundaries} m（z0 底界、z4 顶界）；{wavelengths} m"

    def _update_status(self):
        if self.depth.size == 0:
            self.status_label.setText("在左侧画布右键加载两列 TXT 测井曲线（深度、曲线值）。")
            return
        messages = [
            f"深度 {self.depth[0]:g} ~ {self.depth[-1]:g} m，共 {self.depth.size} 个采样点",
        ]
        if self.wavelengths:
            overlap = transition_width(self._edges())
            messages.append(f"{len(self.wavelengths)} 段滤波，段间过渡带 ±{overlap / 2.0:.2f} m")
            messages.append(self._parameter_text() + self.preset_note)
        if self.overlay_signal.size:
            messages.append(
                f"叠加曲线 {self.overlay_name}：{self.overlay_signal.size} 点，"
                f"深度 {self.overlay_depth[0]:g} ~ {self.overlay_depth[-1]:g} m"
            )
        if self.cycle_signal.size:
            raw_on_grid = np.interp(self.cycle_depth, self.depth, self.signal)
            rmse = float(np.sqrt(np.mean((raw_on_grid - self.cycle_signal) ** 2)))
            messages.append(f"波形-滤波 RMSE = {rmse:.4f}")
        self.status_label.setText("；".join(messages) + "。")


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    widget = PiecewiseFilterWidget()
    widget.resize(1500, 820)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
