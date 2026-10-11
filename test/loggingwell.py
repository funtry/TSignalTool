import os
import sys
from pathlib import Path
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties, findfont
from PyQt6.QtGui import QAction, QGuiApplication, QImage
from scipy.signal.windows import dpss

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.spectrum_analysis import depth_mtm_spectrum


class LoggingWellWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("测井旋回周期分析")
        self.resize(1500, 560)

        self.figure = Figure(figsize=(15, 5), dpi=100)
        self.curve_ax, self.period_ax, self.spectrum_ax = self.figure.subplots(
            1, 3
        )
        self.raw_curve_ax = self.curve_ax.twiny()
        self.figure.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.86,
            bottom=0.13,
            wspace=0.32,
        )
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)

        self.depth = np.asarray([], dtype=float)
        self.signal = np.asarray([], dtype=float)
        self.raw_log_depth = np.asarray([], dtype=float)
        self.raw_log_signal = np.asarray([], dtype=float)
        self.raw_log_name = ""
        self.canvas.mpl_connect("button_press_event", self._show_context_menu)
        self._show_empty_state()

    def _show_empty_state(self):
        self.curve_ax.clear()
        self.raw_curve_ax.clear()
        self.curve_ax.set_title("旋回曲线（右键导入）")
        self.curve_ax.set_xlabel("旋回曲线值", color="#202020")
        self.raw_curve_ax.set_xlabel("原始测井曲线值", color="#D55E00")
        self.curve_ax.set_ylabel("深度 (m)")
        self.curve_ax.text(
            0.5,
            0.5,
            "右键导入旋回曲线或叠加原始测井曲线",
            transform=self.curve_ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

        self.period_ax.clear()
        self.period_ax.set_title("局部周期")
        self.period_ax.set_xlabel("周期 (m)")
        self.period_ax.set_ylabel("深度 (m)")
        self.period_ax.text(
            0.5,
            0.5,
            "导入曲线后根据波峰/波谷计算",
            transform=self.period_ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

        self.spectrum_ax.clear()
        self.spectrum_ax.set_title("MTM 频谱")
        self.spectrum_ax.set_xlabel("空间频率 (cycles/m)")
        self.spectrum_ax.set_ylabel("归一化功率")
        self.spectrum_ax.text(
            0.5,
            0.5,
            "导入曲线后计算",
            transform=self.spectrum_ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
        for ax in (self.curve_ax, self.period_ax, self.spectrum_ax):
            ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _show_context_menu(self, event):
        if event.button != MouseButton.RIGHT or event.inaxes not in (
            self.curve_ax,
            self.raw_curve_ax,
        ):
            return

        menu = QMenu(self)
        cycle_action = menu.addAction("导入旋回曲线...")
        raw_log_action = menu.addAction("导入原始测井曲线...")
        remove_raw_action = None
        if self.raw_log_depth.size:
            remove_raw_action = menu.addAction("移除原始测井曲线")
        chosen = menu.exec(self.canvas.mapToGlobal(event.guiEvent.pos()))
        if chosen == cycle_action:
            self._load_curve()
        elif chosen == raw_log_action:
            self._load_raw_log_curve()
        elif chosen == remove_raw_action:
            self.raw_log_depth = np.asarray([], dtype=float)
            self.raw_log_signal = np.asarray([], dtype=float)
            self.raw_log_name = ""
            self._render_results()

    def _load_raw_log_curve(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入原始测井曲线",
            os.getcwd(),
            "曲线数据 (*.txt *.csv *.dat);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            delimiter = "," if Path(file_path).suffix.lower() == ".csv" else None
            data = np.loadtxt(file_path, dtype=float, delimiter=delimiter, ndmin=2)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "无法导入测井曲线",
                f"读取失败：{error}\n文件应为无表头的两列数值：深度、曲线值。",
            )
            return

        if (
            data.ndim != 2
            or data.shape[0] < 3
            or data.shape[1] < 2
            or not np.all(np.isfinite(data[:, :2]))
            or np.any(np.diff(data[:, 0]) <= 0)
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件至少需要三行有限数值，前两列分别为严格递增的深度和曲线值。",
            )
            return

        self.raw_log_depth = data[:, 0].copy()
        self.raw_log_signal = data[:, 1].copy()
        self.raw_log_name = Path(file_path).name
        self._render_results()

    def _load_curve(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入旋回曲线",
            os.getcwd(),
            "曲线数据 (*.txt *.csv *.dat);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            delimiter = "," if Path(file_path).suffix.lower() == ".csv" else None
            data = np.loadtxt(file_path, dtype=float, delimiter=delimiter, ndmin=2)
        except (OSError, ValueError) as error:
            QMessageBox.warning(
                self,
                "无法导入曲线",
                f"读取失败：{error}\n文件应为无表头的两列数值：深度、曲线值。",
            )
            return

        if (
            data.ndim != 2
            or data.shape[0] < 3
            or data.shape[1] < 2
            or not np.all(np.isfinite(data[:, :2]))
            or np.any(np.diff(data[:, 0]) <= 0)
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件至少需要三行有限数值，前两列分别为严格递增的深度和曲线值。",
            )
            return

        self.depth = data[:, 0].copy()
        self.signal = data[:, 1].copy()
        self._render_results()

    def _local_periods(self):
        value_range = float(np.ptp(self.signal))
        if value_range <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        prominence = value_range * 0.05
        peak_indices, _ = find_peaks(self.signal, prominence=prominence)
        trough_indices, _ = find_peaks(-self.signal, prominence=prominence)

        period_depths = []
        periods = []
        for extrema in (peak_indices, trough_indices):
            if extrema.size < 2:
                continue
            intervals = np.diff(self.depth[extrema])
            period_depths.extend(
                0.5 * (self.depth[extrema[:-1]] + self.depth[extrema[1:]])
            )
            periods.extend(intervals)

        if not periods:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        order = np.argsort(period_depths)
        return (
            np.asarray(period_depths, dtype=float)[order],
            np.asarray(periods, dtype=float)[order],
        )

    def _render_results(self):
        self.curve_ax.clear()
        self.raw_curve_ax.clear()
        self.curve_ax.plot(self.signal, self.depth, color="#202020", linewidth=0.9)
        self.curve_ax.set_title("旋回曲线及原始测井曲线")
        self.curve_ax.set_xlabel("旋回曲线值", color="#202020")
        self.curve_ax.set_ylabel("深度 (m)")
        self.curve_ax.grid(True, alpha=0.3)
        if self.raw_log_depth.size:
            self.raw_curve_ax.plot(
                self.raw_log_signal,
                self.raw_log_depth,
                color="#D55E00",
                linewidth=0.9,
                alpha=0.85,
                label=self.raw_log_name,
            )
            self.raw_curve_ax.set_xlabel(
                f"原始测井曲线值：{self.raw_log_name}",
                color="#D55E00",
            )
            self.raw_curve_ax.tick_params(axis="x", colors="#D55E00")
        else:
            self.raw_curve_ax.set_xlabel("原始测井曲线值", color="#D55E00")
        self.curve_ax.set_ylim(float(self.depth[-1]), float(self.depth[0]))

        period_depths, periods = self._local_periods()
        self.period_ax.clear()
        if periods.size:
            self.period_ax.plot(
                periods,
                period_depths,
                color="#0072B2",
                marker="o",
                markersize=3,
                linewidth=1.0,
            )
            self.period_ax.invert_yaxis()
        else:
            self.period_ax.text(
                0.5,
                0.5,
                "未检测到足够的波峰/波谷",
                transform=self.period_ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
        self.period_ax.set_title("局部周期（峰-峰 / 谷-谷）")
        self.period_ax.set_xlabel("周期 (m)")
        self.period_ax.set_ylabel("深度 (m)")
        self.period_ax.grid(True, alpha=0.3)

        self.spectrum_ax.clear()
        uniform_depth = self._uniform_depth_grid()
        uniform_signal = np.interp(uniform_depth, self.depth, self.signal)
        try:
            _period, power, frequency, confidence = depth_mtm_spectrum(
                uniform_depth,
                uniform_signal,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            self.spectrum_ax.text(
                0.5,
                0.5,
                f"频谱计算失败：{error}",
                transform=self.spectrum_ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
        else:
            self.spectrum_ax.plot(
                frequency,
                power,
                color="#228B22",
                linewidth=1.3,
                label="MTM功率",
            )
            for level, threshold in confidence.items():
                self.spectrum_ax.plot(
                    frequency,
                    threshold,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.75,
                    label=f"{float(level) * 100:g}%阈值",
                )
            self.spectrum_ax.set_xlim(left=0)
            self.spectrum_ax.legend(fontsize=8)

        self.spectrum_ax.set_title("MTM 频谱")
        self.spectrum_ax.set_xlabel("空间频率 (cycles/m)")
        self.spectrum_ax.set_ylabel("归一化功率")
        self.spectrum_ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _uniform_depth_grid(self):
        spacing = float(np.median(np.diff(self.depth)))
        count = max(int(round((self.depth[-1] - self.depth[0]) / spacing)) + 1, 2)
        return np.linspace(self.depth[0], self.depth[-1], count)


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = LoggingWellWidget()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())