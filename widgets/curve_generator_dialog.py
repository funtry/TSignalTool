import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class CurveGeneratorDialog(QDialog):
    CURVE_TYPES = (
        ("constant", "常数"),
        ("linear", "线性曲线"),
        ("sine", "正弦曲线"),
        ("cosine", "余弦曲线"),
        ("normal", "正态钟形曲线"),
        ("noise", "高斯噪声"),
    )

    def __init__(self, reference_depth=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("曲线生成与导出")
        self.resize(960, 620)

        if reference_depth is not None and len(reference_depth) > 1:
            reference_depth = np.asarray(reference_depth, dtype=float)
            step = float(np.diff(reference_depth).mean())
            start = float(reference_depth[0])
            stop = float(reference_depth[-1] + step)
        else:
            start, stop, step = 600.0, 800.0, 0.125

        self.depth_start = self._float_input(start, -1e9, 1e9, 6)
        self.depth_stop = self._float_input(stop, -1e9, 1e9, 6)
        self.depth_step = self._float_input(step, 1e-9, 1e6, 6)

        self.curve_type = QComboBox()
        for key, label in self.CURVE_TYPES:
            self.curve_type.addItem(label, key)

        self.parameter_pages = QStackedWidget()
        self._build_parameter_pages()

        controls = QFormLayout()
        controls.addRow("曲线类型", self.curve_type)
        controls.addRow("深度起点", self.depth_start)
        controls.addRow("深度终点（不含）", self.depth_stop)
        controls.addRow("采样间隔", self.depth_step)
        controls.addRow(self.parameter_pages)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.canvas, 1)
        preview_layout.addWidget(self.status_label)

        content = QHBoxLayout()
        content.addLayout(controls, 0)
        content.addLayout(preview_layout, 1)

        self.export_button = QPushButton("导出 TXT")
        close_button = QPushButton("关闭")
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(content, 1)
        layout.addLayout(button_layout)

        self.curve_type.currentIndexChanged.connect(self._select_parameter_page)
        self.curve_type.currentIndexChanged.connect(self._refresh_preview)
        for control in (self.depth_start, self.depth_stop, self.depth_step):
            control.valueChanged.connect(self._refresh_preview)
        for control in self._parameter_inputs:
            control.valueChanged.connect(self._refresh_preview)
        self.export_button.clicked.connect(self._export_curve)
        close_button.clicked.connect(self.reject)

        self._refresh_preview()

    @staticmethod
    def _float_input(value, minimum, maximum, decimals=4):
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    def _build_parameter_pages(self):
        self._parameter_inputs = []
        self._parameter_controls = {}

        constant_page = QWidget()
        constant_form = QFormLayout(constant_page)
        self._add_parameter(constant_form, "level", "常数值", 0.0)
        self.parameter_pages.addWidget(constant_page)

        linear_page = QWidget()
        linear_form = QFormLayout(linear_page)
        self._add_parameter(linear_form, "slope", "斜率（每米）", 1.0)
        self._add_parameter(linear_form, "intercept", "起点偏移", 0.0)
        self.parameter_pages.addWidget(linear_page)

        periodic_page = QWidget()
        periodic_form = QFormLayout(periodic_page)
        self._add_parameter(periodic_form, "amplitude", "振幅", 1.0)
        self._add_parameter(periodic_form, "period", "周期（米）", 20.0, 1e-6, 1e6)
        self._add_parameter(periodic_form, "phase", "相位（弧度）", 0.0)
        self._add_parameter(periodic_form, "offset", "基线偏移", 0.0)
        self.parameter_pages.addWidget(periodic_page)

        normal_page = QWidget()
        normal_form = QFormLayout(normal_page)
        self._add_parameter(normal_form, "normal_scale", "幅值系数", 1.0)
        center_default = (self.depth_start.value() + self.depth_stop.value()) / 2.0
        self._add_parameter(normal_form, "center", "均值 / 中心深度", center_default, -1e9, 1e9)
        self._add_parameter(normal_form, "sigma", "标准差（米）", 5.0, 1e-6, 1e6)
        self._add_parameter(normal_form, "normal_offset", "基线偏移", 0.0)
        self.parameter_pages.addWidget(normal_page)

        noise_page = QWidget()
        noise_form = QFormLayout(noise_page)
        self._add_parameter(noise_form, "noise_mean", "均值", 0.0)
        self._add_parameter(noise_form, "noise_std", "标准差", 1.0, 0.0, 1e6)
        self.noise_seed = QSpinBox()
        self.noise_seed.setRange(0, 2147483647)
        self.noise_seed.setValue(42)
        noise_form.addRow("随机种子", self.noise_seed)
        self._parameter_inputs.append(self.noise_seed)
        self.parameter_pages.addWidget(noise_page)

    def _add_parameter(self, form, key, label, value, minimum=-1e9, maximum=1e9):
        control = self._float_input(value, minimum, maximum)
        self._parameter_controls[key] = control
        self._parameter_inputs.append(control)
        form.addRow(label, control)

    def _select_parameter_page(self, index):
        page_index = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[index]
        self.parameter_pages.setCurrentIndex(page_index)

    def _make_depth(self):
        start = self.depth_start.value()
        stop = self.depth_stop.value()
        step = self.depth_step.value()
        if stop <= start or step <= 0:
            raise ValueError("深度终点必须大于起点，采样间隔必须大于 0。")

        sample_count = int(np.ceil((stop - start) / step - 1e-12))
        if sample_count < 2:
            raise ValueError("采样点至少需要 2 个。")
        if sample_count > 2_000_000:
            raise ValueError("采样点过多，请缩小深度范围或增大采样间隔。")
        return start + step * np.arange(sample_count, dtype=float)

    def _generate_values(self, depth):
        kind = self.curve_type.currentData()
        relative_depth = depth - depth[0]
        params = self._parameter_controls

        if kind == "constant":
            return np.full(depth.shape, params["level"].value())
        if kind == "linear":
            return params["slope"].value() * relative_depth + params["intercept"].value()
        if kind in ("sine", "cosine"):
            angle = (
                2.0 * np.pi * relative_depth / params["period"].value()
                + params["phase"].value()
            )
            wave = np.sin(angle) if kind == "sine" else np.cos(angle)
            return params["amplitude"].value() * wave + params["offset"].value()
        if kind == "normal":
            sigma = params["sigma"].value()
            density = np.exp(-0.5 * ((depth - params["center"].value()) / sigma) ** 2)
            density /= sigma * np.sqrt(2.0 * np.pi)
            return (
                params["normal_scale"].value() * density
                + params["normal_offset"].value()
            )

        rng = np.random.default_rng(self.noise_seed.value())
        return rng.normal(
            loc=params["noise_mean"].value(),
            scale=params["noise_std"].value(),
            size=depth.size,
        )

    def _refresh_preview(self, *_args):
        try:
            self.depth = self._make_depth()
            self.values = self._generate_values(self.depth)
        except ValueError as error:
            self.export_button.setEnabled(False)
            self.status_label.setText(str(error))
            return

        self.export_button.setEnabled(True)
        self.axes.clear()
        self.axes.plot(self.depth, self.values, color="#0072B2", linewidth=1.1)
        self.axes.set_xlabel("Depth (m)")
        self.axes.set_ylabel("Value")
        self.axes.grid(True, alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText(f"{self.depth.size} 个采样点；TXT 将导出为深度和值两列。")

    def _export_curve(self):
        kind = self.curve_type.currentData()
        label = self.curve_type.currentText()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出曲线",
            f"{label}.txt",
            "文本文件 (*.txt)",
        )
        if not file_path:
            return

        try:
            np.savetxt(
                file_path,
                np.column_stack((self.depth, self.values)),
                fmt="%.10f",
            )
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))
            return
        self.status_label.setText(f"已导出 {self.depth.size} 个采样点：{Path(file_path).name}")


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    dialog = CurveGeneratorDialog()
    dialog.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())