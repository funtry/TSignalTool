import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QMenu, QSizePolicy, QVBoxLayout, QWidget


class SignalProcessWellLogCanvas(QWidget):
    curve_file_action_requested = pyqtSignal(str)

    def __init__(self, parent=None, width=4, height=8, dpi=100):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.22, right=0.96, top=0.97, bottom=0.08)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.range_label = None
        self.range_slider = None
        self.depth = np.asarray([], dtype=float)
        self.signal = np.asarray([], dtype=float)
        self.comparison_signal = None
        self.selected_indices = (0, -1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.import_context_menu_enabled = False
        self.canvas.mpl_connect("button_press_event", self._show_context_menu)

    def _show_context_menu(self, event):
        if (
            not self.import_context_menu_enabled
            or event.button != MouseButton.RIGHT
            or event.inaxes is not self.ax
        ):
            return

        menu = QMenu(self)
        import_action = menu.addAction("导入测井曲线")
        chosen = menu.exec(self.canvas.mapToGlobal(event.guiEvent.pos()))
        if chosen == import_action:
            self.curve_file_action_requested.emit("import")

    def attach_depth_selection_ui(self, slider=None, label=None):
        if slider is not None:
            self.range_slider = slider
            slider.valueChanged.connect(self._on_range_changed)
        if label is not None:
            self.range_label = label
        self._sync_selection_controls()

    def set_data(self, depth, signal):
        depth_arr = np.asarray(depth, dtype=float).reshape(-1)
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        if depth_arr.size == 0 or depth_arr.size != signal_arr.size:
            return
        self.depth = depth_arr
        self.signal = signal_arr
        if self.selected_indices[1] < 0 or self.selected_indices[1] >= depth_arr.size:
            self.selected_indices = (0, depth_arr.size - 1)
        self._sync_selection_controls()
        self._render_curve()

    def set_comparison_signal(self, signal=None):
        if signal is None:
            self.comparison_signal = None
        else:
            comparison = np.asarray(signal, dtype=float).reshape(-1)
            self.comparison_signal = (
                comparison.copy() if comparison.size == self.depth.size else None
            )
        self._render_curve()

    def _sync_selection_controls(self):
        if self.depth.size == 0:
            if self.range_label is not None:
                self.range_label.setText("分析深度：无数据")
            return
        start, end = self.selected_indices
        start = int(np.clip(start, 0, self.depth.size - 1))
        end = int(np.clip(end, start, self.depth.size - 1))
        self.selected_indices = (start, end)
        if self.range_slider is not None:
            self.range_slider.blockSignals(True)
            self.range_slider.setRange(0, self.depth.size - 1)
            self.range_slider.setValue((start, end))
            self.range_slider.blockSignals(False)
        if self.range_label is not None:
            self.range_label.setText(
                f"分析深度：{self.depth[start]:.2f} ~ {self.depth[end]:.2f} m ({end - start + 1} 点)"
            )

    def _on_range_changed(self, value):
        if self.depth.size == 0 or not isinstance(value, (tuple, list)) or len(value) != 2:
            return
        start = int(np.clip(value[0], 0, self.depth.size - 1))
        end = int(np.clip(value[1], start, self.depth.size - 1))
        self.selected_indices = (start, end)
        self._sync_selection_controls()
        self._render_curve()

    def get_current_depth_range(self):
        if self.depth.size == 0:
            return None, None
        start, end = self.selected_indices
        return float(self.depth[start]), float(self.depth[end])

    def get_current_analysis_subset(self):
        if self.depth.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        start, end = self.selected_indices
        return self.depth[start:end + 1].copy(), self.signal[start:end + 1].copy()

    def _render_curve(self):
        self.ax.clear()
        self.ax.invert_yaxis()
        #self.ax.set_xlabel("Signal")
        #self.ax.set_ylabel("Depth (m)")
        self.ax.grid(True, alpha=0.3, linestyle="--")
        start, end = self.selected_indices
        self.ax.plot(self.signal, self.depth, color="#B8B8B8", linewidth=0.8)
        self.ax.plot(
            self.signal[start:end + 1],
            self.depth[start:end + 1],
            color="#0000FF",
            linewidth=1.15,
        )
        if self.comparison_signal is not None:
            self.ax.plot(
                self.comparison_signal,
                self.depth,
                color="#FF0000",
                linewidth=1.1,
            )
        self.ax.set_ylim(float(self.depth[-1]), float(self.depth[0]))
        self.canvas.draw_idle()