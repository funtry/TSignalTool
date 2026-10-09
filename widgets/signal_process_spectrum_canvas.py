import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.ticker import LogFormatterMathtext, LogLocator, ScalarFormatter
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from scipy.fft import rfftfreq
from scipy.signal import find_peaks

from core.spectrum_analysis import depth_fft_spectrum, depth_mtm_spectrum


class SignalProcessSpectrumCanvas(QWidget):
    FFT_VIEW = "全局功率谱（FFT）"
    MTM_VIEW = "全局功率谱（MTM）"
    STFT_VIEW = "演化时频谱（STFT）"

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        super().__init__(parent)
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.12)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.method_combo = QComboBox(self)
        self.method_combo.addItems((self.FFT_VIEW, self.MTM_VIEW, self.STFT_VIEW))
        self.method_combo.setCurrentText(self.MTM_VIEW)
        self.method_combo.currentTextChanged.connect(self._render_current_result)
        self.visibility_button = QPushButton("隐藏辅助标注", self)
        self.visibility_button.setCheckable(True)
        self.visibility_button.toggled.connect(self._toggle_annotations)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.visibility_button)
        controls.addWidget(self.method_combo)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)

        self.results = {}
        self.depth = np.asarray([], dtype=float)
        self.signal = np.asarray([], dtype=float)
        self.show_annotations = True
        self.ax_top_freq = None
        self.reference_wavelengths = ()

    def set_reference_wavelengths(self, wavelengths):
        values = np.asarray(wavelengths, dtype=float).reshape(-1)
        if values.size != 4 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Exactly four positive finite reference wavelengths are required.")
        self.reference_wavelengths = tuple(float(value) for value in values)

    def set_signal_data(self, depth, signal):
        depth_arr = np.asarray(depth, dtype=float).reshape(-1)
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        if depth_arr.size < 2 or depth_arr.size != signal_arr.size:
            return
        if not np.all(np.isfinite(depth_arr)) or not np.all(np.isfinite(signal_arr)):
            return

        self.depth = depth_arr.copy()
        self.signal = signal_arr.copy()
        self.results = {}
        self._compute_all_results()
        self._render_current_result(self.method_combo.currentText())

    def _compute_all_results(self):
        try:
            period, power, _freq, confidence = depth_fft_spectrum(self.depth, self.signal)
            self.results[self.FFT_VIEW] = {
                "period": period,
                "power": power,
                "confidence": confidence,
            }
        except (ValueError, FloatingPointError):
            self.results[self.FFT_VIEW] = None

        try:
            period, power, _freq, confidence = depth_mtm_spectrum(self.depth, self.signal)
            self.results[self.MTM_VIEW] = {
                "period": period,
                "power": power,
                "confidence": confidence,
            }
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            self.results[self.MTM_VIEW] = None

        self.results[self.STFT_VIEW] = self._compute_stft_result()

    def _compute_stft_result(self):
        count = self.signal.size
        if count < 8:
            return None
        window_points = min(256, count)
        if window_points % 2:
            window_points -= 1
        overlap_ratio = 0.9375
        step = max(int(round(window_points * (1.0 - overlap_ratio))), 1)
        dx = float(np.median(np.diff(self.depth)))
        starts = np.arange(0, count - window_points + 1, step, dtype=int)
        if starts.size == 0 or dx <= 0:
            return None

        window = np.hanning(window_points)
        frequency = rfftfreq(window_points, d=dx)[1:]
        if frequency.size == 0:
            return None
        period = 1.0 / frequency
        power_rows = []
        centers = []
        for start in starts:
            segment = self.signal[start:start + window_points]
            segment = segment - np.mean(segment)
            spectrum = np.fft.rfft(segment * window)[1:]
            power_rows.append(np.abs(spectrum) ** 2 / window_points)
            centers.append(float(self.depth[start + window_points // 2]))

        power = np.asarray(power_rows, dtype=float)
        order = np.argsort(period)
        period = period[order]
        power = power[:, order]
        finite = power[np.isfinite(power)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
            if high > low:
                power = np.clip((power - low) / (high - low), 0.0, 1.0)
            else:
                power = np.zeros_like(power)
        return {
            "period": period,
            "centers_depth": np.asarray(centers, dtype=float),
            "power_matrix": power,
        }

    def _toggle_annotations(self, checked):
        self.show_annotations = not checked
        self.visibility_button.setText("显示辅助标注" if checked else "隐藏辅助标注")
        self._render_current_result(self.method_combo.currentText())

    def _render_current_result(self, method):
        if self.ax_top_freq is not None:
            self.ax_top_freq.remove()
            self.ax_top_freq = None
        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        result = self.results.get(method)
        if result is None:
            self.ax.text(0.5, 0.5, "当前深度范围数据不足", transform=self.ax.transAxes, ha="center", va="center")
            self.canvas.draw_idle()
            return

        if method == self.STFT_VIEW:
            self._render_stft(result)
        else:
            self._render_global_spectrum(method, result)
        self._format_axes()
        self.canvas.draw_idle()

    def _render_global_spectrum(self, method, result):
        period = np.asarray(result["period"], dtype=float)
        power = np.maximum(np.asarray(result["power"], dtype=float), 1e-12)
        order = np.argsort(period)
        period = period[order]
        power = power[order]

        self.ax.plot(period, power, color="#228B22", linewidth=1.7, linestyle="-")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Wavelength (m)")
        self.ax.set_ylabel("Power")
        self.ax.set_xlim(float(period[-1]) * 1.05, float(period[0]) * 0.95)

        confidence = result.get("confidence", {})
        for index, (key, curve) in enumerate(confidence.items()):
            values = np.maximum(np.asarray(curve, dtype=float)[order], 1e-12)
            if self.show_annotations:
                self.ax.plot(
                    period,
                    values,
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.85,
                    label=f"{float(key) * 100:.0f}%",
                )

        if self.show_annotations:
            for label, center in zip(("E", "e", "O", "P"), self.reference_wavelengths):
                if period[0] <= center <= period[-1]:
                    self.ax.axvspan(center * 0.9, center * 1.1, color="#888888", alpha=0.12)
                    self.ax.text(
                        center,
                        self.ax.get_ylim()[1] * 0.88,
                        f"{label}: {center:.3f} m",
                        ha="center",
                        va="top",
                        color="#555555",
                        fontsize=7,
                    )

            c99 = confidence.get("0.99")
            if c99 is not None:
                c99 = np.asarray(c99, dtype=float)[order]
                peaks, _ = find_peaks(power)
                significant = peaks[power[peaks] > c99[peaks]]
                if significant.size:
                    self.ax.scatter(period[significant], power[significant], s=22, facecolors="none", edgecolors="#333333")
            if confidence:
                self.ax.legend(loc="best", fontsize=7, ncol=3)

        self.ax_top_freq = self.ax.secondary_xaxis(
            "top", functions=(lambda value: 1.0 / np.maximum(value, 1e-12), lambda value: 1.0 / np.maximum(value, 1e-12))
        )
        self.ax_top_freq.set_xlabel("Frequency (1/m)")

    def get_e_wavelength(self):
        result = self.results.get(self.method_combo.currentText())
        if result is None or "period" not in result or "power" not in result:
            return np.nan
        period = np.asarray(result["period"], dtype=float)
        power = np.asarray(result["power"], dtype=float)
        if period.size == 0 or power.size == 0:
            return np.nan
        order = np.argsort(period)
        return float(period[order][np.argmax(power[order])])

    def _render_stft(self, result):
        period = result["period"]
        depth_centers = result["centers_depth"]
        matrix = result["power_matrix"]
        period_mesh, depth_mesh = np.meshgrid(period, depth_centers)
        self.ax.pcolormesh(period_mesh, depth_mesh, matrix, shading="auto", cmap="coolwarm")
        self.ax.set_xscale("log")
        self.ax.set_xlim(float(np.max(period)), float(np.min(period)))
        self.ax.set_ylim(float(np.max(depth_centers)), float(np.min(depth_centers)))
        self.ax.set_xlabel("Wavelength (m)")
        self.ax.set_ylabel("Depth (m)")
        self.ax_top_freq = self.ax.secondary_xaxis(
            "top", functions=(lambda value: 1.0 / np.maximum(value, 1e-12), lambda value: 1.0 / np.maximum(value, 1e-12))
        )
        self.ax_top_freq.set_xlabel("Frequency (1/m)")

    def _format_axes(self):
        self.ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
        self.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        if self.ax.get_yscale() == "log":
            self.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=5))
            self.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
        if self.ax_top_freq is not None:
            self.ax_top_freq.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
            self.ax_top_freq.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))