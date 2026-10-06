import sys
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.figure import Figure
from PyQt6.QtCore import QByteArray, QMimeData, Qt
from PyQt6.QtGui import QImage
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
    QMenu,
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
        self.resize(1400, 900)

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

        self._loaded_curves = [None, None]
        self._loaded_three_column_data = None
        self._showing_hilbert_curve = False
        self._difference_requested = False
        self._plotted_curves = []
        self.load_curve_buttons = [
            QPushButton("加载曲线 1 TXT"),
            QPushButton("加载曲线 2 TXT"),
        ]
        self.take_difference_button = QPushButton("取差值")
        self.take_difference_button.setEnabled(False)
        self.take_difference_button.clicked.connect(self._take_difference)
        self.load_three_column_button = QPushButton("加载三列数据 TXT")
        self.load_three_column_button.clicked.connect(self._load_three_column_data)
        self.hilbert_button = QPushButton("Hilbert")
        self.hilbert_button.clicked.connect(self._show_hilbert_curve)
        self.clear_loaded_button = QPushButton("清除导入")
        import_buttons = QHBoxLayout()
        for index, button in enumerate(self.load_curve_buttons):
            button.clicked.connect(lambda _checked=False, i=index: self._load_curve(i))
            import_buttons.addWidget(button)
        import_buttons.addWidget(self.take_difference_button)
        self.clear_loaded_button.clicked.connect(self._clear_loaded_curves)
        import_buttons.addWidget(self.clear_loaded_button)
        controls.addRow("两曲线差值", import_buttons)
        direct_plot_buttons = QHBoxLayout()
        direct_plot_buttons.addWidget(self.load_three_column_button)
        direct_plot_buttons.addWidget(self.hilbert_button)
        controls.addRow("直接绘图", direct_plot_buttons)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_plot_context_menu)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.plot_font = self._find_chinese_font()

        top_layout = QHBoxLayout()
        top_layout.addLayout(controls)
        top_layout.addStretch(1)

        self.export_button = QPushButton("导出 TXT")
        close_button = QPushButton("关闭")
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top_layout)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.status_label)
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
    def _find_chinese_font():
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for font_name in (
            "Noto Sans SC (TrueType).otf",
            "Deng.ttf",
            "simhei.ttf",
            "simsun.ttc",
        ):
            font_path = fonts_dir / font_name
            if font_path.is_file():
                return FontProperties(fname=str(font_path))

        for family in ("Microsoft YaHei", "SimHei", "DengXian"):
            try:
                font_path = findfont(
                    FontProperties(family=family),
                    fallback_to_default=False,
                )
            except ValueError:
                continue
            return FontProperties(fname=font_path)
        return FontProperties()

    def _show_plot_context_menu(self, position):
        menu = QMenu(self.canvas)
        if self._plotted_curves:
            ordinals = ("第一", "第二", "第三")
            for index, curve in enumerate(self._plotted_curves[:3]):
                action = menu.addAction(f"导出{ordinals[index]}条曲线")
                action.triggered.connect(
                    lambda _checked=False, curve_index=index: self._export_plotted_curve(curve_index)
                )
            menu.addSeparator()
        copy_action = menu.addAction("复制")
        copy_action.triggered.connect(self._copy_plot_to_clipboard)
        menu.exec(self.canvas.mapToGlobal(position))

    def _export_plotted_curve(self, curve_index):
        if not 0 <= curve_index < len(self._plotted_curves):
            return
        depth, values, label = self._plotted_curves[curve_index]
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"导出{label}",
            f"{label}.txt",
            "文本文件 (*.txt)",
        )
        if not file_path:
            return
        try:
            np.savetxt(
                file_path,
                np.column_stack((depth, values)),
                fmt="%.10f",
            )
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))
            return
        self.status_label.setText(f"已导出 {depth.size} 个采样点：{Path(file_path).name}")

    def _copy_plot_to_clipboard(self):
        png_buffer = BytesIO()
        self.figure.savefig(
            png_buffer,
            format="png",
            dpi=300,
            transparent=True,
        )
        png_data = png_buffer.getvalue()
        image = QImage.fromData(png_data, "PNG")
        dots_per_meter = round(300 / 0.0254)
        image.setDotsPerMeterX(dots_per_meter)
        image.setDotsPerMeterY(dots_per_meter)

        mime_data = QMimeData()
        mime_data.setImageData(image)
        mime_data.setData("image/png", QByteArray(png_data))
        QApplication.clipboard().setMimeData(mime_data)

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
        if self._loaded_three_column_data is not None:
            self._refresh_three_column_preview()
            return
        if all(curve is not None for curve in self._loaded_curves):
            if self._difference_requested:
                self._refresh_difference_preview()
            else:
                self._refresh_loaded_curves_preview()
            return

        try:
            self.depth = self._make_depth()
            self.values = self._generate_values(self.depth)
        except ValueError as error:
            self.export_button.setEnabled(False)
            self.status_label.setText(str(error))
            return

        self.export_button.setEnabled(True)
        self.axes.clear()
        self.axes.plot(self.depth, self.values, color="#0000FF", linewidth=1.1)
        self._plotted_curves = [(self.depth.copy(), self.values.copy(), "曲线")]
        self.axes.set_xlabel("Depth (m)")
        self.axes.set_ylabel("Value")
        self.axes.grid(True, alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText(f"{self.depth.size} 个采样点；TXT 将导出为深度和值两列。")

    def _load_three_column_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载三列数据",
            "",
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            data = np.loadtxt(file_path, dtype=float, ndmin=2)
        except (OSError, ValueError):
            QMessageBox.warning(self, "数据格式不正确", "文件必须包含三列深度和数据。")
            return

        if (
            data.shape[1] != 3
            or data.shape[0] < 2
            or not np.all(np.isfinite(data))
            or np.any(np.diff(data[:, 0]) <= 0)
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件必须包含至少两行有限数值，第一列为严格递增的深度，第二、三列为数据。",
            )
            return

        self._loaded_three_column_data = (
            data[:, 0].copy(),
            data[:, 1].copy(),
            data[:, 2].copy(),
            Path(file_path).name,
        )
        self._showing_hilbert_curve = False
        self._difference_requested = False
        self._loaded_curves = [None, None]
        self.take_difference_button.setEnabled(False)
        self.take_difference_button.setEnabled(False)
        for index, button in enumerate(self.load_curve_buttons):
            button.setText(f"加载曲线 {index + 1} TXT")
        self.load_three_column_button.setText(Path(file_path).name)
        self._refresh_three_column_preview()

    def _refresh_three_column_preview(self):
        depth, first_values, second_values, file_name = self._loaded_three_column_data
        self.depth = depth
        self.values = None
        self.export_button.setEnabled(False)
        self.axes.clear()
        if self._showing_hilbert_curve:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ratio = np.divide(
                    first_values,
                    second_values,
                    out=np.full(first_values.shape, np.nan, dtype=float),
                    where=second_values != 0,
                )
            ratio[~np.isfinite(ratio)] = np.nan
            valid_count = int(np.count_nonzero(np.isfinite(ratio)))
            self.axes.plot(
                depth,
                first_values,
                color="#0000FF",
                linewidth=1.2,
                label=f"{file_name} 第2列",
            )
            self.axes.plot(
                depth,
                second_values,
                color="#FF0000",
                linewidth=0.8,
                label=f"{file_name} 第3列",
            )
            self.axes.plot(
                depth,
                ratio,
                color="#00FFFF",
                linewidth=1.5,
                label="Hilbert（第2列 / 第3列）",
            )
            self._plotted_curves = [
                (depth.copy(), first_values.copy(), "第一条曲线"),
                (depth.copy(), second_values.copy(), "第二条曲线"),
                (depth.copy(), ratio.copy(), "第三条曲线"),
            ]
            self.axes.set_xlabel("Depth (m)")
            self.axes.set_ylabel("Value", fontproperties=self.plot_font)
            self.axes.grid(True, alpha=0.25)
            self.axes.legend(prop=self.plot_font)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.status_label.setText(
                f"{file_name}；Hilbert 比值曲线有效点 {valid_count}/{depth.size}。"
            )
            return

        self.axes.plot(
            depth,
            first_values,
            color="#0000FF",
            linewidth=1.2,
            label=f"{file_name} 第2列",
        )
        self.axes.plot(
            depth,
            second_values,
            color="#FF0000",
            linewidth=0.8,
            label=f"{file_name} 第3列",
        )
        self._plotted_curves = [
            (depth.copy(), first_values.copy(), "第一条曲线"),
            (depth.copy(), second_values.copy(), "第二条曲线"),
        ]
        self.axes.set_xlabel("Depth (m)")
        self.axes.set_ylabel("Value")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(prop=self.plot_font)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText(
            f"{file_name}；按深度绘制第2列和第3列，共 {depth.size} 个采样点。"
        )

    def _show_hilbert_curve(self):
        if self._loaded_three_column_data is None:
            self.status_label.setText("请先加载三列数据 TXT。")
            return
        self._showing_hilbert_curve = True
        self._refresh_three_column_preview()

    def _refresh_loaded_curves_preview(self):
        first_depth, first_values, first_name = self._loaded_curves[0]
        second_depth, second_values, second_name = self._loaded_curves[1]
        self.depth = first_depth
        self.values = None
        self.export_button.setEnabled(False)
        self.axes.clear()
        self.axes.plot(
            first_depth,
            first_values,
            color="#0000FF",
            linewidth=1.2,
            label=first_name,
        )
        self.axes.plot(
            second_depth,
            second_values,
            color="#FF0000",
            linewidth=0.8,
            label=second_name,
        )
        self._plotted_curves = [
            (first_depth.copy(), first_values.copy(), "第一条曲线"),
            (second_depth.copy(), second_values.copy(), "第二条曲线"),
        ]
        self.axes.set_xlabel("Depth (m)")
        self.axes.set_ylabel("Value")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(prop=self.plot_font)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText("两条曲线已加载；点击“取差值”计算曲线1减曲线2。")

    def _take_difference(self):
        if not all(curve is not None for curve in self._loaded_curves):
            self.status_label.setText("请先加载两条 TXT 曲线。")
            return
        self._difference_requested = True
        self._refresh_difference_preview()

    def _load_curve(self, curve_index):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"加载曲线 {curve_index + 1}",
            "",
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            data = np.loadtxt(file_path, dtype=float)
        except (OSError, ValueError):
            QMessageBox.warning(self, "数据格式不正确", "文件必须包含两列深度和曲线值。")
            return

        if (
            data.ndim != 2
            or data.shape[1] != 2
            or data.shape[0] < 2
            or not np.all(np.isfinite(data))
            or np.any(np.diff(data[:, 0]) <= 0)
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件必须包含至少两行有限数值，深度和值各一列，且深度严格递增。",
            )
            return

        self._loaded_curves[curve_index] = (
            data[:, 0].copy(),
            data[:, 1].copy(),
            Path(file_path).name,
        )
        self._loaded_three_column_data = None
        self._showing_hilbert_curve = False
        self._difference_requested = False
        self.take_difference_button.setEnabled(
            all(curve is not None for curve in self._loaded_curves)
        )
        self.load_three_column_button.setText("加载三列数据 TXT")
        self.load_curve_buttons[curve_index].setText(Path(file_path).name)
        self._refresh_preview()

    def _clear_loaded_curves(self):
        self._loaded_curves = [None, None]
        self._loaded_three_column_data = None
        self._showing_hilbert_curve = False
        self._difference_requested = False
        self.take_difference_button.setEnabled(False)
        for index, button in enumerate(self.load_curve_buttons):
            button.setText(f"加载曲线 {index + 1} TXT")
        self.load_three_column_button.setText("加载三列数据 TXT")
        self._refresh_preview()

    def _refresh_difference_preview(self):
        (first_depth, first_values, first_name), (
            second_depth,
            second_values,
            second_name,
        ) = self._loaded_curves
        overlap_start = max(first_depth[0], second_depth[0])
        overlap_stop = min(first_depth[-1], second_depth[-1])
        mask = (first_depth >= overlap_start) & (first_depth <= overlap_stop)
        depth = first_depth[mask]
        if depth.size < 2:
            self.export_button.setEnabled(False)
            self.status_label.setText("两条曲线的重叠深度范围内采样点不足，无法计算差值。")
            return

        first_values = first_values[mask]
        aligned_second_values = np.interp(depth, second_depth, second_values)
        difference = first_values - aligned_second_values
        self.depth = depth
        self.values = difference

        self.export_button.setEnabled(True)
        self.axes.clear()
        self.axes.plot(
            first_depth,
            self._loaded_curves[0][1],
            color="#0000FF",
            linewidth=1.2,
            label=first_name,
        )
        self.axes.plot(
            second_depth,
            self._loaded_curves[1][1],
            color="#FF0000",
            linewidth=0.8,
            label=second_name,
        )
        self.axes.plot(
            depth,
            difference,
            color="#00FF00",
            linewidth=1.5,
            label="曲线B - 曲线R",
        )
        self._plotted_curves = [
            (first_depth.copy(), self._loaded_curves[0][1].copy(), "第一条曲线"),
            (second_depth.copy(), self._loaded_curves[1][1].copy(), "第二条曲线"),
            (depth.copy(), difference.copy(), "第三条曲线"),
        ]
        self.axes.set_xlabel("Depth (m)")
        self.axes.set_ylabel("Value")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(prop=self.plot_font)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.setText(
            f"差值 = 曲线1 - 曲线2；重叠区间 {depth[0]:.6g} 至 {depth[-1]:.6g}，"
            f"{depth.size} 个采样点；导出仅包含差值曲线。"
        )

    def _export_curve(self):
        difference_mode = self._difference_requested and all(
            curve is not None for curve in self._loaded_curves
        )
        label = "曲线差值" if difference_mode else self.curve_type.currentText()
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