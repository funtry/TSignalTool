"""分段滤波（差分进化）窗口：相同三列布局，中间列叠加代价热力图。"""

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cycle_metrics import DEFAULT_CYCLE_COUNT_PRIOR, analyze_cycle_curve
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
# 中间列横坐标覆盖完整的波长搜索区间（左右各留 1 m 余量），
# 保证 4 个分段的竖线无论取 5 m 还是 20 m 都能完整显示。
PERIOD_AXIS_LIMITS = (MIN_WAVELENGTH - 1.0, MAX_WAVELENGTH + 1.0)
SEGMENT_COLORS = ("#FFFF00", "#FFFF00", "#FFFF00",  "#FFFF00")
RAW_LOG_COLOR = "#C8C8C8"
CYCLE_COLOR = "#FF0000"
PEAK_COLOR = "#C8C8C8"
OVERLAY_COLOR = "#0000FF"

# ---------------------------------------------------------------- 差分进化参数
# 以下参数全部写在代码内部，界面不提供输入；需要调整时直接改这里的数值。
DE_POPULATION_SIZE = 40  # 种群规模（种子数）
DE_MUTATION_FACTOR = 0.7  # 缩放因子 F（best/1 变异）
DE_CROSSOVER_RATE = 0.9  # 交叉概率 CR
DE_MAX_GENERATIONS = 400  # 最大迭代代数
DE_RANDOM_SEED = 20261017  # 随机种子；设为 None 则每次运行随机初始化
DE_STEP_INTERVAL_MS = 80  # 自动迭代时每代的间隔（毫秒）

# 约束：4 个分段中每一段的持续深度都不小于该阈值（米）。
DE_MIN_SEGMENT_THICKNESS = 15.0

# 终止条件：连续若干代最优代价的变化都小于容差时停止迭代。
DE_TOLERANCE = 1e-4  # 相邻两代最优代价之差小于该值视为「变化不大」
DE_STALL_GENERATIONS = 2  # 连续停滞的代数达到该值即终止

# 代价函数 J = w_e·J_e + w_s·J_s + w_c·J_c 的三项权重。
COST_WEIGHT_ENERGY = 0.3
COST_WEIGHT_SMOOTHNESS = 0.3
COST_WEIGHT_CYCLE_COUNT = 0.4
# 理论旋回个数 N_t：提取的旋回个数 N_c 越接近该值，J_c 越小。
COST_TARGET_CYCLE_COUNT = 32.0

# 中间列代价热力图的网格步长：行按深度、列按局部周期。
HEATMAP_DEPTH_STEP = 0.125  # 米
HEATMAP_PERIOD_STEP = 0.02  # m/旋回
# 反向 coolwarm：代价最大为蓝色，代价最小为红色。
HEATMAP_CMAP = "coolwarm_r"

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


class PiecewiseFilterDEWidget(QWidget):
    """左侧原始测井曲线，中间代价热力图+分段局部周期，右侧最优旋回曲线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分段滤波（差分进化）")

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

        self._de_rng = np.random.default_rng(DE_RANDOM_SEED)
        self._de_population = np.empty((0, PARAMETER_COUNT), dtype=float)
        self._de_costs = np.empty(0, dtype=float)
        self._de_generation = 0
        self._de_best_parameters = None
        self._de_best_cost = float("inf")
        self._de_finished = False
        self._de_stall_generations = 0

        self._heat_costs = None

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

        self.step_button = QPushButton("下一步")
        self.step_button.setToolTip(
            "迭代中点击可暂停；暂停后每点击一次执行一代，并在第三列显示当前最优解。"
        )
        self.step_button.setEnabled(False)
        self.step_button.clicked.connect(self._on_step_clicked)

        button_row = QHBoxLayout()
        button_row.addWidget(self.step_button)
        button_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(button_row)

        self.iteration_timer = QTimer(self)
        self.iteration_timer.setInterval(DE_STEP_INTERVAL_MS)
        self.iteration_timer.timeout.connect(self._iterate_once)

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
        self._restart_iteration()
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
        self._restart_iteration()

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
        self._restart_iteration()
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
        self._restart_iteration()

    # ---------------------------------------------------------------- 差分进化

    def _de_minimum_thickness(self):
        """每段必须达到的最小持续深度；跨度不足 4 倍阈值时退化为等分。"""
        span = float(self.depth[-1] - self.depth[0])
        if DE_MIN_SEGMENT_THICKNESS * SEGMENT_COUNT > span:
            return span / SEGMENT_COUNT
        return DE_MIN_SEGMENT_THICKNESS

    def _repair_parameters(self, values):
        """把任意 7 参数修复为可行解：4 段自底向顶有序、各段厚度不小于下限。

        段厚可以相等（例如数据跨度较小时退化为等分），但每段都不少于
        DE_MIN_SEGMENT_THICKNESS；波长夹取到 [MIN_WAVELENGTH, MAX_WAVELENGTH]。
        """
        top = float(self.depth[0])
        bottom = float(self.depth[-1])
        span = bottom - top
        values = np.asarray(values, dtype=float).reshape(-1)
        minimum = self._de_minimum_thickness()

        depths = np.sort(np.clip(values[:3], top, bottom))[::-1]
        thicknesses = -np.diff([bottom, *depths.tolist(), top])
        thicknesses = np.maximum(thicknesses, 0.0)

        spare = span - SEGMENT_COUNT * minimum
        if spare <= 1e-12:
            thicknesses = np.full(SEGMENT_COUNT, span / SEGMENT_COUNT)
        else:
            # 低于阈值的段先抬到阈值，剩余深度按各段的富余量成比例分配。
            extra = np.maximum(thicknesses - minimum, 0.0)
            total = float(np.sum(extra))
            if total > 1e-12:
                thicknesses = minimum + spare * extra / total
            else:
                thicknesses = np.full(SEGMENT_COUNT, span / SEGMENT_COUNT)

        rebuilt = (bottom - np.cumsum(thicknesses))[: SEGMENT_COUNT - 1]
        wavelengths = np.clip(values[3:], MIN_WAVELENGTH, MAX_WAVELENGTH)
        return np.asarray([*rebuilt.tolist(), *wavelengths.tolist()], dtype=float)

    def _solution_arrays(self, parameters):
        """7 参数 → (分段边界(底→顶), 波长, 滤波边界, 局部频率)。"""
        parameters = np.asarray(parameters, dtype=float).reshape(-1)
        boundaries = [float(self.depth[-1]), *parameters[:3].tolist(), float(self.depth[0])]
        wavelengths = [float(value) for value in parameters[3:]]
        return (
            boundaries,
            wavelengths,
            list(reversed(boundaries)),
            [1.0 / value for value in reversed(wavelengths)],
        )

    def _cost_terms(self, metrics, cycle_signal):
        """返回 (J_e, J_s, J_c) 三个分项：

        J_e = -(1/N)·Σ O²(z_i)                              平均振荡能量（取负号便于最小化）
        J_s = 1/(N-2)·Σ (O(z_{i+1}) - 2O(z_i) + O(z_{i-1}))²  二阶差分平方均值
        J_c = (N_c - N_t)²                                  旋回个数偏差平方
        """
        signal = np.asarray(cycle_signal, dtype=float)
        energy_term = -float(np.mean(signal**2)) if signal.size else 0.0
        if signal.size >= 3:
            second_difference = signal[2:] - 2.0 * signal[1:-1] + signal[:-2]
            smoothness_term = float(np.mean(second_difference**2))
        else:
            smoothness_term = 0.0
        count_term = float((metrics.cycle_count - COST_TARGET_CYCLE_COUNT) ** 2)
        return energy_term, smoothness_term, count_term

    def _weighted_cost(self, metrics, cycle_signal):
        """综合代价 J = w_e·J_e + w_s·J_s + w_c·J_c。"""
        energy_term, smoothness_term, count_term = self._cost_terms(metrics, cycle_signal)
        return (
            COST_WEIGHT_ENERGY * energy_term
            + COST_WEIGHT_SMOOTHNESS * smoothness_term
            + COST_WEIGHT_CYCLE_COUNT * count_term
        )

    def _evaluate_solution(self, parameters):
        """按 7 参数计算旋回曲线与代价；不可行时返回 None。"""
        boundaries, wavelengths, edges, frequencies = self._solution_arrays(parameters)
        try:
            cycle_depth, cycle_signal = piecewise_gaussian_bandpass(
                self.depth,
                self.signal,
                edges,
                frequencies,
                relative_bandwidth=RELATIVE_BANDWIDTH,
            )
            metrics = analyze_cycle_curve(
                cycle_depth,
                cycle_signal,
                prior_count=COST_TARGET_CYCLE_COUNT,
                min_cycle_period=float(np.min(wavelengths)),
            )
        except ValueError:
            return None
        return (
            self._weighted_cost(metrics, cycle_signal),
            cycle_depth,
            cycle_signal,
            metrics,
            boundaries,
            wavelengths,
        )

    def _evaluate_cost(self, parameters):
        result = self._evaluate_solution(parameters)
        return float("inf") if result is None else float(result[0])

    def _initialize_heatmap(self):
        """建立 M×N 代价矩阵：0.125 m 深度步长 × 0.02 m/旋回周期步长。

        初值为 inf（尚未评估，按色带最大值画成蓝色），随迭代把更小的代价写回
        对应格子（逐渐变红）。
        """
        top = float(self.depth[0])
        bottom = float(self.depth[-1])
        rows = max(1, int(np.ceil((bottom - top) / HEATMAP_DEPTH_STEP)))
        columns = max(
            1,
            int(
                np.ceil(
                    (PERIOD_AXIS_LIMITS[1] - PERIOD_AXIS_LIMITS[0]) / HEATMAP_PERIOD_STEP
                )
            ),
        )
        self._heat_costs = np.full((rows, columns), np.inf, dtype=float)

    def _record_heat(self, parameters, cost):
        """把解的代价写入其 4 段覆盖的格子；已有更小代价时保持原值。"""
        if self._heat_costs is None or not np.isfinite(cost):
            return
        top = float(self.depth[0])
        rows, columns = self._heat_costs.shape
        period_origin = float(PERIOD_AXIS_LIMITS[0])
        for index in range(SEGMENT_COUNT):
            column = int(
                round((float(parameters[3 + index]) - period_origin) / HEATMAP_PERIOD_STEP)
            )
            if not 0 <= column < columns:
                continue
            if index == SEGMENT_COUNT - 1:
                segment_top = top
            else:
                segment_top = float(parameters[index])
            if index == 0:
                segment_bottom = float(self.depth[-1])
            else:
                segment_bottom = float(parameters[index - 1])
            first = max(
                0, int(round((segment_top - top) / HEATMAP_DEPTH_STEP))
            )
            last = min(
                rows, int(round((segment_bottom - top) / HEATMAP_DEPTH_STEP))
            )
            if last <= first:
                continue
            cell = self._heat_costs[first:last, column]
            cell[cost < cell] = cost

    def _restart_iteration(self):
        """加载数据或参数变化后重建种群、清空代价矩阵并开始自动迭代。"""
        self.iteration_timer.stop()
        self._de_finished = False
        self._de_generation = 0
        self._de_stall_generations = 0
        self._de_best_parameters = None
        self._de_best_cost = float("inf")
        self._heat_costs = None
        self.step_button.setEnabled(True)
        self.step_button.setText("暂停")
        if self.depth.size < 2:
            self._de_population = np.empty((0, PARAMETER_COUNT), dtype=float)
            self._de_costs = np.empty(0, dtype=float)
            self.step_button.setText("下一步")
            self.step_button.setEnabled(False)
            return

        self._initialize_heatmap()
        self._de_rng = np.random.default_rng(DE_RANDOM_SEED)
        top = float(self.depth[0])
        bottom = float(self.depth[-1])
        minimum = self._de_minimum_thickness()
        if minimum < DE_MIN_SEGMENT_THICKNESS - 1e-9:
            print(
                f"提示：深度跨度 {bottom - top:g} m 不足 "
                f"{SEGMENT_COUNT}×{DE_MIN_SEGMENT_THICKNESS:g} m，"
                f"每段最小厚度自动降为 {minimum:.3f} m。"
            )
        population = np.empty((DE_POPULATION_SIZE, PARAMETER_COUNT), dtype=float)
        for index in range(DE_POPULATION_SIZE):
            seed = np.concatenate(
                (
                    self._de_rng.uniform(top, bottom, SEGMENT_COUNT - 1),
                    self._de_rng.uniform(MIN_WAVELENGTH, MAX_WAVELENGTH, SEGMENT_COUNT),
                )
            )
            population[index] = self._repair_parameters(seed)
        costs = np.asarray([self._evaluate_cost(row) for row in population], dtype=float)

        self._de_population = population
        self._de_costs = costs
        best = int(np.argmin(costs))
        self._de_best_parameters = population[best].copy()
        self._de_best_cost = float(costs[best])
        self._apply_best_solution()
        self._render()
        self.iteration_timer.start()

    def _evolve_generation(self):
        """执行一代 best/1/bin 差分进化，逐个体选择更优的试验向量。"""
        population = self._de_population
        costs = self._de_costs
        if population.size == 0:
            return
        best = population[int(np.argmin(costs))].copy()
        indices = np.arange(population.shape[0])
        for index in indices:
            others = indices[indices != index]
            a, b = self._de_rng.choice(others, 2, replace=False)
            mutant = best + DE_MUTATION_FACTOR * (population[a] - population[b])
            crossover = self._de_rng.random(PARAMETER_COUNT) < DE_CROSSOVER_RATE
            crossover[self._de_rng.integers(PARAMETER_COUNT)] = True
            trial = self._repair_parameters(np.where(crossover, mutant, population[index]))
            trial_cost = self._evaluate_cost(trial)
            self._record_heat(trial, trial_cost)
            if trial_cost < costs[index]:
                population[index] = trial
                costs[index] = trial_cost

        self._de_generation += 1
        best_index = int(np.argmin(costs))
        if costs[best_index] < self._de_best_cost:
            self._de_best_parameters = population[best_index].copy()
            self._de_best_cost = float(costs[best_index])
        self._record_heat(population[best_index], float(costs[best_index]))

    def _apply_best_solution(self):
        """把当前最优解写成界面上的 7 参数、旋回曲线与三项指标。"""
        if self._de_best_parameters is None or self.depth.size < 2:
            return
        result = self._evaluate_solution(self._de_best_parameters)
        if result is None:
            return
        _, cycle_depth, cycle_signal, metrics, boundaries, wavelengths = result
        self.cycle_depth = cycle_depth
        self.cycle_signal = cycle_signal
        self.metrics = metrics
        self.boundaries = boundaries
        self.wavelengths = wavelengths

    def _advance_one_generation(self):
        """进化一代并更新画面，返回是否已满足停滞终止条件。"""
        previous = self._de_best_cost
        self._evolve_generation()
        if abs(self._de_best_cost - previous) < DE_TOLERANCE:
            self._de_stall_generations += 1
        else:
            self._de_stall_generations = 0
        self._apply_best_solution()
        self._render()
        return self._de_stall_generations >= DE_STALL_GENERATIONS

    def _iterate_once(self):
        """自动迭代的一次节拍：进化一代、刷新热力图与第三列最优曲线。"""
        if self._de_population.size == 0 or self._de_finished:
            return
        if self._de_generation >= DE_MAX_GENERATIONS:
            self._finish_iteration(f"达到最大代数 {DE_MAX_GENERATIONS}")
            return
        if self._advance_one_generation():
            self._finish_iteration(
                f"最优代价连续 {self._de_stall_generations} 代变化小于 {DE_TOLERANCE:g}"
            )

    def _finish_iteration(self, reason):
        self.iteration_timer.stop()
        self._de_finished = True
        self.step_button.setText("迭代完成")
        self.step_button.setEnabled(False)
        print(f"迭代结束：{reason}，共 {self._de_generation} 代。")
        self._print_best("最终最优解")

    def _on_step_clicked(self):
        """迭代中点击暂停；暂停后每点击一次执行一代并显示当前最优解。"""
        if self.iteration_timer.isActive():
            self.iteration_timer.stop()
            self.step_button.setText("下一步")
            self._print_best("已暂停")
            return
        if self._de_population.size == 0 or self._de_finished:
            return
        if self._de_generation >= DE_MAX_GENERATIONS:
            self._finish_iteration(f"达到最大代数 {DE_MAX_GENERATIONS}")
            return
        if self._advance_one_generation():
            self._finish_iteration(
                f"最优代价连续 {self._de_stall_generations} 代变化小于 {DE_TOLERANCE:g}"
            )
            return
        self._print_best(f"第 {self._de_generation} 代")

    def _resume_iteration(self):
        """从右键菜单恢复自动迭代。"""
        if self.depth.size < 2 or self._de_finished:
            return
        if self._de_population.size == 0:
            self._restart_iteration()
            return
        self.step_button.setEnabled(True)
        self.step_button.setText("暂停")
        self.iteration_timer.start()

    def _print_best(self, prefix):
        if self._de_best_parameters is None or self.metrics is None:
            return
        values = "、".join(
            f"{name}={value:.4f}"
            for name, value in zip(PARAMETER_NAMES, self._de_best_parameters)
        )
        energy_term, smoothness_term, count_term = self._cost_terms(
            self.metrics, self.cycle_signal
        )
        print(
            f"{prefix}：J={self._de_best_cost:.4f}"
            f"（J_e={energy_term:.4f}，J_s={smoothness_term:.6f}，"
            f"J_c={count_term:.4f}）；旋回个数 {self.metrics.cycle_count:.2f}"
            f"（目标 {COST_TARGET_CYCLE_COUNT:g}）；最优 7 参数 {values}"
        )

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
                    prior_count=COST_TARGET_CYCLE_COUNT,
                    min_cycle_period=float(np.min(self.wavelengths)),
                )
            except ValueError as error:
                QMessageBox.warning(self, "分段滤波失败", str(error))
        self._render(verbose=True)

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
            handlers = []
            if self.wavelengths:
                handlers.extend(
                    [
                        ("编辑 7 个参数…", self.edit_parameters),
                        ("恢复默认参数", self.reset_parameters),
                    ]
                )
            if self._de_population.size:
                handlers.append(("继续自动迭代", self._resume_iteration))
                handlers.append(("重新初始化种群", self._restart_iteration))
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

    def _render(self, verbose=False):
        self._render_log_axes()
        self._render_period_axes()
        self._render_cycle_axes()
        if verbose:
            self._print_metrics()
            self._print_status()
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

    def _draw_heatmap(self, axes):
        """把 M×N 代价矩阵画成热力图：蓝色代表代价最大，红色代表代价最小。

        色带按已评估格子的最小/最大值自适应；尚未评估的格子取最大值端，因此
        初始化时整幅图全蓝，只有被更小代价覆盖过的格子才会逐步变红。图例隐藏。
        """
        if self._heat_costs is None or self.depth.size < 2:
            return
        evaluated = np.isfinite(self._heat_costs)
        lower = float(np.min(self._heat_costs[evaluated])) if evaluated.any() else 0.0
        upper = float(np.max(self._heat_costs[evaluated])) if evaluated.any() else 1.0
        if upper - lower < 1e-9:
            lower = upper - 1.0
        display = np.where(evaluated, self._heat_costs, upper)
        axes.imshow(
            display,
            extent=(
                PERIOD_AXIS_LIMITS[0],
                PERIOD_AXIS_LIMITS[1],
                float(self.depth[-1]),
                float(self.depth[0]),
            ),
            origin="upper",
            aspect="auto",
            cmap=HEATMAP_CMAP,
            norm=Normalize(vmin=lower, vmax=upper),
            interpolation="nearest",
            zorder=0,
        )

    def _render_period_axes(self):
        axes = self.period_axes
        title = "分段局部周期"
        if self._de_generation or self._de_best_cost < float("inf"):
            title = f"{title}（第 {self._de_generation} 代，最优代价 {self._de_best_cost:.4f}）"
        self._prepare_axes(axes, title, "局部周期 x（m/旋回）")
        axes.set_xlim(*PERIOD_AXIS_LIMITS)
        self._draw_heatmap(axes)
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
            labels=["拼接过渡带"],
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
        self._apply_cycle_axis_limits(axes)
        self._draw_cycle_peaks(axes)
        axes.legend(handles=self._cycle_legend_handles(), loc="upper right", fontsize=8)

    def _apply_cycle_axis_limits(self, axes):
        """横坐标按旋回曲线自身振幅自适应，避免被幅值更大的原始曲线压扁。"""
        lower = float(np.min(self.cycle_signal))
        upper = float(np.max(self.cycle_signal))
        span = upper - lower
        if span <= 0:
            span = max(abs(upper), 1.0) * 0.2
        padding = span * 0.05
        axes.set_xlim(lower - padding, upper + padding)

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
            Line2D([], [], color=RAW_LOG_COLOR, linewidth=0.8, label="原始曲线"),
            Line2D([], [], color=CYCLE_COLOR, linewidth=1.15, label="旋回曲线"),
        ]
        if self.overlay_signal.size:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=OVERLAY_COLOR,
                    linewidth=0.7,
                    label="叠加曲线",
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
                    label="波峰编号",
                )
            )
        return handles

    def _print_metrics(self):
        """代价分项与旋回统计输出到控制台，不再占用画布下方空间。"""
        if self.metrics is None:
            print("加载测井曲线后统计波峰、旋回个数与三项代价分项。")
            return
        metrics = self.metrics
        energy_term, smoothness_term, count_term = self._cost_terms(
            metrics, self.cycle_signal
        )
        print(
            f"旋回个数 {metrics.cycle_count:.2f}"
            f"（目标 {COST_TARGET_CYCLE_COUNT:g}，波峰 {metrics.peak_depths.size} 个）｜"
            f"J_e={energy_term:.4f}｜J_s={smoothness_term:.6f}｜J_c={count_term:.4f}"
        )

    def _parameter_text(self):
        boundaries = " ".join(
            f"z{index}={value:.2f}" for index, value in enumerate(self.boundaries)
        )
        wavelengths = " ".join(
            f"x{index + 1}={value:.2f}" for index, value in enumerate(self.wavelengths)
        )
        return f"7 参数：{boundaries} m（z0 底界、z4 顶界）；{wavelengths} m"

    def _print_status(self):
        """深度、参数、叠加曲线与 RMSE 等状态信息输出到控制台。"""
        if self.depth.size == 0:
            print("在左侧画布右键加载两列 TXT 测井曲线（深度、曲线值）。")
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
        print("；".join(messages) + "。")


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    widget = PiecewiseFilterDEWidget()
    widget.resize(1500, 820)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
