import sys
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
from scipy.signal import butter, hilbert, sosfiltfilt
from statsmodels.nonparametric.smoothers_lowess import lowess

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from widgets.signal_process_widget import SignalProcessWidget


class SignalDecomposeWidget(SignalProcessWidget):
    """Decompose an imported well-log into the seven process tracks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("测井曲线分解")
        self.well_log_canvas.import_context_menu_enabled = True
        self.well_log_canvas.curve_file_action_requested.connect(
            self._handle_well_log_file_action
        )
        self.lowess_slider = self.curve_canvas.add_trend_smoothing_control()
        self.lowess_slider.valueChanged.connect(self._decompose_and_display)
        self.lowess_slider.valueChanged.connect(self._update_lowess_window_label)
        self.e_period_spinbox = self.curve_canvas.add_e_period_control(initial_cycles=13)
        self.e_period_spinbox.valueChanged.connect(self._decompose_and_display)
        self.original_depth = None
        self.original_signal = None

    def _handle_well_log_file_action(self, action):
        if action != "import":
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入测井曲线",
            str(Path.cwd()),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            data = np.loadtxt(file_path, dtype=float, ndmin=2)
        except (OSError, ValueError):
            QMessageBox.warning(self, "数据格式不正确", "文件必须包含两列深度和测井值。")
            return

        if (
            data.ndim != 2
            or data.shape[1] != 2
            or data.shape[0] < 8
            or not np.all(np.isfinite(data))
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件必须包含至少 8 行有限数值，深度和值各一列。",
            )
            return

        order = np.argsort(data[:, 0])
        raw_depth = data[order, 0]
        raw_signal = data[order, 1]
        unique_depth, unique_indices = np.unique(raw_depth, return_index=True)
        raw_signal = raw_signal[unique_indices]
        if unique_depth.size < 8:
            QMessageBox.warning(self, "数据格式不正确", "有效且不重复的深度采样点至少需要 8 个。")
            return

        dx = float(np.median(np.diff(unique_depth)))
        if not np.isfinite(dx) or dx <= 0:
            QMessageBox.warning(self, "数据格式不正确", "无法确定有效的深度采样间隔。")
            return
        sample_count = int(np.floor((unique_depth[-1] - unique_depth[0]) / dx)) + 1
        depth = unique_depth[0] + dx * np.arange(sample_count, dtype=float)
        signal = np.interp(depth, unique_depth, raw_signal)

        self.original_depth = depth
        self.original_signal = signal
        self.depth = depth.copy()
        self.well_log_canvas.selected_indices = (0, depth.size - 1)
        self.well_log_canvas.set_comparison_signal(None)
        self.well_log_canvas.set_data(depth, signal)
        self._update_lowess_window_label()
        self._decompose_and_display()

    def _decompose(self, depth, signal, smooth_percent, e_cycles):
        count = signal.size
        frac = float(np.clip(smooth_percent / 100.0, 0.01, 1.0))
        trend = lowess(signal, depth, frac=frac, it=2, return_sorted=False)
        dx = float(np.median(np.diff(depth)))
        nyquist = 0.5 / dx
        depth_span = float(depth[-1] - depth[0])
        e_period = depth_span / float(e_cycles)
        periods = e_period * np.asarray((1.0, 5.0 / 20.0, 2.0 / 20.0, 1.0 / 20.0))

        phases = 2.0 * np.pi * (depth - depth[0])[:, None] / periods[None, :]
        orbital_waves = np.sin(phases)
        ratios = np.asarray((20.0, 5.0, 2.0, 1.0))
        components = orbital_waves * (ratios / ratios.sum())[None, :]
        components = [components[:, index] for index in range(components.shape[1])]

        p_cutoff = min(1.0 / periods[-1], nyquist * 0.8)
        if count > 8 and p_cutoff > 0:
            noise_sos = butter(2, p_cutoff, btype="highpass", fs=1.0 / dx, output="sos")
            noise = sosfiltfilt(noise_sos, signal, padlen=min(9, count - 1))
        else:
            noise = np.zeros_like(signal)

        residual = signal - trend - noise
        amplitude = np.abs(hilbert(residual))

        return trend, amplitude, components, noise, periods

    def _decompose_and_display(self, *_args):
        if self.original_depth is None or self.original_signal is None:
            return

        depth = self.original_depth
        signal = self.original_signal
        trend, amplitude, components, noise, periods = self._decompose(
            depth,
            signal,
            self.lowess_slider.value(),
            self.e_period_spinbox.value(),
        )
        self.signal_curves = [trend, amplitude, *components, noise]
        oscillatory_signal = amplitude * np.sum(components, axis=0)
        self.spectral_signal = oscillatory_signal
        self.total_signal = trend + oscillatory_signal + noise
        self.track_annotations = (
            f"LOWESS {self.lowess_slider.value()}%",
            "Hilbert envelope",
            f"Sinusoid: {self.e_period_spinbox.value()} cycles",
            f"Sinusoid: {self.e_period_spinbox.value() * 4} cycles",
            f"Sinusoid: {self.e_period_spinbox.value() * 10} cycles",
            f"Sinusoid: {self.e_period_spinbox.value() * 20} cycles",
            "High-pass filtered input",
        )

        self.well_log_canvas.set_data(depth, signal)
        self.well_log_canvas.set_comparison_signal(self.total_signal)
        self.curve_canvas.set_curves(depth, self.signal_curves, self.track_annotations)
        self._update_spectrum_for_selected_depth()
        self._update_lowess_window_label()
        self.depth_range_label.setText(
            f"测井已分解：{depth[0]:.2f} ~ {depth[-1]:.2f} m；"
            f"LOWESS 平滑 {self.lowess_slider.value()}%；E={self.e_period_spinbox.value()} 个周期；"
            f"红线为七轨重组曲线。"
        )

    def _update_lowess_window_label(self, *_args):
        if self.original_depth is None or self.original_depth.size < 2:
            return
        depth_span = float(self.original_depth[-1] - self.original_depth[0])
        self.curve_canvas.set_trend_window_depth(depth_span)

    def _update_spectrum_for_selected_depth(self):
        if (
            getattr(self, "original_depth", None) is None
            or getattr(self, "spectral_signal", None) is None
        ):
            return
        start, end = self.well_log_canvas.selected_indices
        if end - start + 1 >= 2:
            self.spectrum_canvas.set_signal_data(
                self.original_depth[start:end + 1],
                self.spectral_signal[start:end + 1],
            )

    def _refresh_signal_for_selected_depth(self):
        self._signal_refresh_pending = False
        if self.original_depth is None or self.original_signal is None:
            return
        self._update_spectrum_for_selected_depth()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SignalDecomposeWidget()
    widget.resize(1400, 900)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
