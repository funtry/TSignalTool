import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.cycle_filter import extract_cycle
from core.de_optimizer import run_de, update_heatmap
from core.well_log_loader import load_well_log
from widgets.depth_canvases import HeatmapCanvas, WellLogCanvas


class MainWindow(QMainWindow):
    WEIGHT_NAMES = ("振荡能量权重", "旋回平滑权重", "旋回计数权重")
    PARAMETER_GROUPS = (
        (
            "代价函数权重设置",
            (
                ("振荡能量权重", "小数", 0.34, 0.0, 1.0),
                ("旋回平滑权重", "小数", 0.33, 0.0, 1.0),
                ("旋回计数权重", "小数", 0.33, 0.0, 1.0),
                ("目标旋回数量", "整数", 20, 1, 10000),
            ),
        ),
        (
            "差分进化算法参数",
            (
                ("种群数量", "整数", 40, 4, 10000),
                ("缩放因子", "小数", 0.7, 0.0, 2.0),
                ("交叉概率", "小数", 0.8, 0.0, 1.0),
                ("变异策略", "选项", "最优个体差分", ("最优个体差分", "随机个体差分", "当前最优差分")),
                ("最大迭代代数", "整数", 300, 1, 100000),
                ("收敛判定窗口代数", "整数", 30, 1, 10000),
                ("代价收敛阈值", "小数", 0.000001, 0.000000001, 1.0),
                ("随机种子", "整数", 42, 0, 2147483647),
            ),
        ),
        (
            "地质模型与信号滤波参数",
            (
                ("地层分段数量", "固定选项", "4", ("4",)),
                ("单段最小地层厚度（米）", "小数", 15.0, 0.1, 10000.0),
                ("带通滤波相对带宽", "小数", 0.2, 0.01, 0.99),
                ("旋回周期下限（米/旋回）", "小数", 5.0, 0.1, 1000.0),
                ("旋回周期上限（米/旋回）", "小数", 25.0, 0.2, 1000.0),
                ("启用物理约束", "勾选", True, None, None),
                ("地层底界深度（米）", "小数", 5505.0, -100000.0, 100000.0),
                ("地层顶界深度（米）", "小数", 5272.0, -100000.0, 100000.0),
                ("测井采样间隔（米）", "小数", 0.125, 0.000001, 10000.0),
            ),
        ),
        (
            "热力图网格参数",
            (
                ("横轴起始（米/旋回）", "小数", 2.0, 0.1, 1000.0),
                ("横轴终止（米/旋回）", "小数", 8.0, 0.2, 1000.0),
                ("横轴步长", "小数", 0.1, 0.001, 100.0),
                ("深度网格步长（米）", "小数", 0.1, 0.001, 100.0),
                ("回填插值方式", "选项", "线性插值", ("线性插值", "最近邻插值", "三次插值")),
            ),
        ),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("测井旋回反演")
        self.parameter_widgets = {}
        self.depth = None
        self.gr = None
        self._build_interface()
        self.well_canvas.import_requested.connect(self.import_txt)
        self._log("界面已就绪，旋回滤波默认中心周期为三点五米每旋回。")

    def _build_interface(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        self.well_canvas = WellLogCanvas()
        self.heatmap_canvas = HeatmapCanvas()
        self.parameter_panel = self._build_parameter_panel()
        splitter.addWidget(self.well_canvas)
        splitter.addWidget(self.heatmap_canvas)
        splitter.addWidget(self.parameter_panel)
        for index, stretch in enumerate((1, 1, 3)):
            splitter.setStretchFactor(index, stretch)
        splitter.setSizes([400, 400, 1200])

    def _build_parameter_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel("龙潭组反演参数")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        for group_name, fields in self.PARAMETER_GROUPS:
            group = QGroupBox(group_name)
            grid = QGridLayout(group)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(5)
            for index, field in enumerate(fields):
                name, kind, default, *limits = field
                if kind in ("选项", "固定选项"):
                    minimum, maximum = None, limits[0]
                else:
                    minimum, maximum = limits
                row, column = divmod(index, 3)
                cell = QWidget()
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(2, 1, 2, 1)
                label = QLabel(name)
                control = self._make_control(kind, default, minimum, maximum)
                if isinstance(control, QCheckBox):
                    control.setText(name)
                    cell_layout.addWidget(control)
                else:
                    cell_layout.addWidget(label)
                    cell_layout.addWidget(control)
                self.parameter_widgets[name] = control
                grid.addWidget(cell, row, column)
            layout.addWidget(group)

        for name in self.WEIGHT_NAMES:
            self.parameter_widgets[name].editingFinished.connect(
                lambda field=name: self.normalize_weights(field)
            )

        button_row = QHBoxLayout()
        self.reset_button = QPushButton("重置默认参数")
        self.validate_button = QPushButton("参数校验")
        self.run_button = QPushButton("运行DE反演")
        self.stop_button = QPushButton("停止")
        for button in (self.reset_button, self.validate_button, self.run_button, self.stop_button):
            button_row.addWidget(button)
        layout.addLayout(button_row)
        self.reset_button.clicked.connect(self.reset_defaults)
        self.validate_button.clicked.connect(self.validate_parameters)
        self.run_button.clicked.connect(self.run_de)
        self.stop_button.clicked.connect(self.stop_de)

        layout.addWidget(QLabel("运行日志"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(110)
        layout.addWidget(self.log_output, 1)
        return panel

    @staticmethod
    def _make_control(kind, default, minimum, maximum):
        if kind in ("选项", "固定选项"):
            control = QComboBox()
            control.addItems(maximum)
            control.setCurrentText(default)
            if kind == "固定选项":
                control.setEnabled(False)
            return control
        if kind == "勾选":
            control = QCheckBox()
            control.setChecked(default)
            return control
        if kind == "整数":
            control = QSpinBox()
            control.setRange(minimum, maximum)
            control.setValue(default)
            control.setKeyboardTracking(False)
            return control
        control = QDoubleSpinBox()
        control.setDecimals(6)
        control.setRange(minimum, maximum)
        control.setValue(default)
        control.setKeyboardTracking(False)
        return control

    def _log(self, message: str) -> None:
        self.log_output.append(message)

    def import_txt(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入测井文本文件",
            str(Path.cwd()),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return
        try:
            depth, gr = load_well_log(file_path)
            values = self._parameter_values()
            center_period = (values["旋回周期下限（米/旋回）"] + values["旋回周期上限（米/旋回）"]) / 2
            cycle = extract_cycle(
                depth,
                gr,
                center_wavelength=center_period,
                relative_bandwidth=values["带通滤波相对带宽"],
                sampling_interval=values["测井采样间隔（米）"],
            )
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            self._log(f"导入失败：{exc}")
            return

        self.depth, self.gr = depth, gr
        self.well_canvas.plot_log(depth, gr, cycle)
        self.heatmap_canvas.set_depth_limits(float(depth.min()), float(depth.max()))
        self._log(f"已加载 {Path(file_path).name}，共读取 {depth.size} 个有效测井点。")
        self._log(f"已提取中心周期 {center_period:g} 米/旋回的带通旋回曲线。")

    def normalize_weights(self, changed_name=None) -> None:
        controls = [self.parameter_widgets[name] for name in self.WEIGHT_NAMES]
        values = [control.value() for control in controls]
        total = sum(values)
        if total <= 0:
            normalized = [0.333333, 0.333333, 0.333334]
        else:
            normalized = [round(value / total, 6) for value in values]
            target_index = self.WEIGHT_NAMES.index(changed_name) if changed_name in self.WEIGHT_NAMES else 0
            normalized[target_index] = round(normalized[target_index] + 1.0 - sum(normalized), 6)
        for control, value in zip(controls, normalized):
            control.setValue(value)

    def reset_defaults(self) -> None:
        for _, fields in self.PARAMETER_GROUPS:
            for name, kind, default, _, _ in fields:
                control = self.parameter_widgets[name]
                if isinstance(control, QComboBox):
                    control.setCurrentText(default)
                elif isinstance(control, QCheckBox):
                    control.setChecked(default)
                else:
                    control.setValue(default)
        self.normalize_weights()
        self._log("参数已重置为默认值。")

    def _parameter_values(self) -> dict:
        values = {}
        for name, control in self.parameter_widgets.items():
            if isinstance(control, QComboBox):
                values[name] = control.currentText()
            elif isinstance(control, QCheckBox):
                values[name] = control.isChecked()
            else:
                values[name] = control.value()
        return values

    def _validation_errors(self) -> list[str]:
        values = self._parameter_values()
        self.normalize_weights()
        values.update({name: self.parameter_widgets[name].value() for name in self.WEIGHT_NAMES})
        errors = []
        if abs(sum(values[name] for name in self.WEIGHT_NAMES) - 1.0) > 0.000001:
            errors.append("三项代价权重之和必须等于一。")
        if values["旋回周期下限（米/旋回）"] >= values["旋回周期上限（米/旋回）"]:
            errors.append("旋回周期下限必须小于上限。")
        if values["横轴起始（米/旋回）"] >= values["横轴终止（米/旋回）"]:
            errors.append("热力图横轴起始值必须小于终止值。")
        if values["地层顶界深度（米）"] >= values["地层底界深度（米）"]:
            errors.append("地层顶界深度必须小于底界深度。")
        if values["带通滤波相对带宽"] >= 1:
            errors.append("带通滤波相对带宽必须小于一。")
        if values["单段最小地层厚度（米）"] <= 0 or values["测井采样间隔（米）"] <= 0:
            errors.append("地层厚度和测井采样间隔必须大于零。")
        if values["横轴步长"] <= 0 or values["深度网格步长（米）"] <= 0:
            errors.append("热力图网格步长必须大于零。")
        if values["收敛判定窗口代数"] > values["最大迭代代数"]:
            errors.append("收敛判定窗口代数不能大于最大迭代代数。")
        return errors

    def validate_parameters(self) -> bool:
        errors = self._validation_errors()
        if errors:
            message = "\n".join(errors)
            QMessageBox.warning(self, "参数校验失败", message)
            self._log(f"参数校验失败：{'；'.join(errors)}")
            return False
        self._log("参数校验通过。")
        QMessageBox.information(self, "参数校验", "参数校验通过。")
        return True

    def run_de(self) -> None:
        errors = self._validation_errors()
        if errors:
            QMessageBox.warning(self, "参数校验失败", "\n".join(errors))
            self._log(f"参数校验失败：{'；'.join(errors)}")
            return
        self._log("已提交差分进化反演参数；核心迭代尚未接入。")
        run_de(self._parameter_values())
        update_heatmap(self.heatmap_canvas)

    def stop_de(self) -> None:
        self._log("已收到停止请求；当前没有正在运行的反演任务。")


def main() -> None:
    if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()