import sys
from pathlib import Path
import os

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from widgets.signal_process_curve_canvas import SignalProcessCurveCanvas
from widgets.curve_generator_dialog import CurveGeneratorDialog
from widgets.signal_process_spectrum_canvas import SignalProcessSpectrumCanvas
from widgets.signal_process_well_log_canvas import SignalProcessWellLogCanvas


class SignalProcessWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._build_simulated_astro_signal()

        self.well_log_canvas = SignalProcessWellLogCanvas()
        self.depth_range_slider = QRangeSlider(Qt.Orientation.Vertical) 
        self.depth_range_slider.setMinimumWidth(22)
        self.depth_range_slider.setMaximumWidth(28)
        self.depth_range_slider.setInvertedAppearance(True)
        self.depth_range_slider.setInvertedControls(True)
        self.depth_range_label = QLabel()
        self.well_log_canvas.attach_depth_selection_ui(
            slider=self.depth_range_slider,
            label=self.depth_range_label,
        )
        self.spectrum_canvas = SignalProcessSpectrumCanvas()
        self.curve_canvas = SignalProcessCurveCanvas()
        self.curve_canvas.curve_file_action_requested.connect(self._handle_curve_file_action)
        self._signal_refresh_pending = False
        self.curve_generator_button = QPushButton("曲线生成器")
        self.curve_generator_button.clicked.connect(self._open_curve_generator)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.curve_generator_button, 0, Qt.AlignmentFlag.AlignRight)
        right_layout.addWidget(self.spectrum_canvas, 1)
        right_layout.addWidget(self.curve_canvas, 1)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.depth_range_label)
        left_plot_layout = QHBoxLayout()
        left_plot_layout.setContentsMargins(0, 0, 0, 0)
        left_plot_layout.addWidget(self.depth_range_slider)
        left_plot_layout.addWidget(self.well_log_canvas, 1)
        left_layout.addLayout(left_plot_layout, 1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2)

        self.well_log_canvas.set_data(self.depth, self.total_signal)
        self.curve_canvas.set_curves(self.depth, self.signal_curves)
        self.depth_range_slider.valueChanged.connect(
            self._update_signal_for_selected_depth
        )
        self._update_spectrum_for_selected_depth()

    def _update_signal_for_selected_depth(self, _value=None):
        if self._signal_refresh_pending:
            return
        self._signal_refresh_pending = True
        QTimer.singleShot(0, self._refresh_signal_for_selected_depth)

    def _refresh_signal_for_selected_depth(self):
        self._signal_refresh_pending = False
        depth_low, depth_high = self.well_log_canvas.get_current_depth_range()
        if depth_low is None or depth_high is None:
            return

        center = (depth_low + depth_high) / 2.0
        sigma = max((depth_high - depth_low) / 6.0, float(np.diff(self.depth).mean()))
        self._update_curve_arrays(center, sigma)
        self.well_log_canvas.set_data(self.depth, self.total_signal)
        self.curve_canvas.set_curves(self.depth, self.signal_curves)

        self._update_spectrum_for_selected_depth()

    def _update_spectrum_for_selected_depth(self):
        depth_subset, signal_subset = self.well_log_canvas.get_current_analysis_subset()
        if depth_subset.size >= 2:
            self.spectrum_canvas.set_signal_data(depth_subset, signal_subset)

    def _open_curve_generator(self):
        dialog = CurveGeneratorDialog(self.depth, self)
        dialog.exec()

    def _gaussian_trend(self, center, sigma):
        return (
            self.trend_amplitude
            / (sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((self.depth - center) / sigma) ** 2)
        )

    def _build_track_annotations(self, center, sigma):
        trend_annotation = (
            f"{self.trend_amplitude:g}×N({center:.2f}, {sigma ** 2:.2f})"
        )
        return (
            trend_annotation,
            "A envelope",
            None, None, None, None,
            "y=0",
        )

    def _update_curve_arrays(self, center, sigma):
        self.trend_curve = self._curve_overrides.get(
            0, self._gaussian_trend(center, sigma)
        )
        self.amplitude_curve = self._curve_overrides.get(
            1, np.exp(-0.5 * ((self.depth - center) / sigma) ** 2)
        )
        for index in range(4):
            self.components[index] = self._curve_overrides.get(
                index + 2, self.components[index]
            )
        self.noise_curve = self._curve_overrides.get(
            6, np.zeros_like(self.depth)
        )
        self.signal_curves = [
            self.trend_curve,
            self.amplitude_curve,
            *self.components,
            self.noise_curve,
        ]
        self.total_signal = (
            self.trend_curve
            + self.amplitude_curve * np.sum(self.components, axis=0)
            + self.noise_curve
        )
        generated_annotations = list(self._build_track_annotations(center, sigma))
        for index in self._curve_overrides:
            generated_annotations[index] = "Imported TXT"
        self.track_annotations = tuple(generated_annotations)

    def _handle_curve_file_action(self, curve_index, action):
        track_name = self.track_labels[curve_index]
        if action == "export":
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出曲线",
                os.path.join(os.getcwd(), f"{track_name}.txt"),
                "文本文件 (*.txt)",
            )
            if not file_path:
                return
            np.savetxt(
                file_path,
                np.column_stack((self.depth, self.signal_curves[curve_index])),
                fmt="%.10f",
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入曲线", os.getcwd(), "文本文件 (*.txt);;所有文件 (*)"
        )
        if not file_path:
            return
        try:
            imported = np.loadtxt(file_path, dtype=float)
        except (OSError, ValueError):
            QMessageBox.warning(self, "数据格式不正确", "文件必须包含两列深度和曲线值。")
            return

        if (
            imported.ndim != 2
            or imported.shape != (self.depth.size, 2)
            or not np.all(np.isfinite(imported))
            or not np.allclose(imported[:, 0], self.depth, rtol=0.0, atol=1e-8)
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件必须有两列、深度范围与当前数据一致，并按0.125 m间隔采样。",
            )
            return

        self._curve_overrides[curve_index] = imported[:, 1].copy()
        self._refresh_signal_for_selected_depth()

    def _build_simulated_astro_signal(self):
        self.depth = np.arange(600.0, 800.0, 0.125)
        self.trend_amplitude = 2000.0
        self.periods = (20.0, 5.0, 2.0, 1.0)
        self.track_labels = ("Trend", "A", "E", "e", "O", "P", "Noise")
        self.track_periods = (None, None, *self.periods, None)
        amplitudes = (20, 5, 2, 1)
        self.components = [
            amplitude * np.cos(2.0 * np.pi * (self.depth - self.depth[0]) / period)
            for period, amplitude in zip(self.periods, amplitudes)
        ]
        center = (self.depth[0] + self.depth[-1]) / 2.0
        sigma = max((self.depth[-1] - self.depth[0]) / 6.0, float(np.diff(self.depth).mean()))
        self._curve_overrides = {}
        self._update_curve_arrays(center, sigma)


def main():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    widget = SignalProcessWidget()
    widget.setWindowTitle("信号预处理与频谱分析")
    widget.resize(1400, 900)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())