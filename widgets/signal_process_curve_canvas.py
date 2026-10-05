from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SignalProcessCurveCanvas(QWidget):
    curve_file_action_requested = pyqtSignal(int, str)

    COLORS = (
        "#D55E00", "#E69F00", "#0072B2", "#009E73",
        "#CC79A7", "#56B4E9", "#4D4D4D",
    )

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = list(self.fig.subplots(7, 1, sharex=True))
        self.lines = []
        self.track_labels = ("Trend", "A", "E", "e", "O", "P", "Noise")

        for index, ax in enumerate(self.axes):
            line, = ax.plot([], [], linewidth=1.2, color=self.COLORS[index])
            self.lines.append(line)
            ax.set_ylabel(self.track_labels[index], rotation=0, ha="right", va="center", labelpad=28)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)

        self.axes[-1].set_xlabel("Depth (m)")
        self.fig.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.10, hspace=0.08)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.e_period_control = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._show_curve_context_menu)
        self.trend_smoothing_slider = None
        self.trend_smoothing_label = None
        self.trend_window_depth_label = None

    def add_trend_smoothing_control(self, initial_value=15):
        if self.trend_smoothing_slider is not None:
            return self.trend_smoothing_slider

        self.trend_smoothing_label = QLabel(f"LOWESS 趋势平滑：{initial_value}%", self)
        self.trend_smoothing_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.trend_smoothing_slider.setRange(1, 100)
        self.trend_smoothing_slider.setValue(initial_value)
        self.trend_smoothing_slider.valueChanged.connect(
            lambda value: self.trend_smoothing_label.setText(f"LOWESS 趋势平滑：{value}%")
        )
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.trend_smoothing_label)
        control_layout.addWidget(self.trend_smoothing_slider, 1)
        self.trend_window_depth_label = QLabel("窗口深度：-- m", self)
        control_layout.addWidget(self.trend_window_depth_label)
        self.layout().insertLayout(0, control_layout)
        return self.trend_smoothing_slider

    def set_trend_window_depth(self, depth_span):
        if self.trend_smoothing_slider is None or self.trend_window_depth_label is None:
            return
        window_depth = float(depth_span) * self.trend_smoothing_slider.value() / 100.0
        self.trend_window_depth_label.setText(f"窗口深度：{window_depth:.3f} m")

    def add_e_period_control(self, initial_cycles=13):
        if self.e_period_control is not None:
            return self.e_period_control

        self.e_period_control = QWidget(self.canvas)
        control_layout = QHBoxLayout(self.e_period_control)
        control_layout.setContentsMargins(4, 1, 4, 1)
        control_layout.setSpacing(3)
        control_layout.addWidget(QLabel("E 周期数", self.e_period_control))
        spin_box = QSpinBox(self.e_period_control)
        spin_box.setRange(1, 10000)
        spin_box.setValue(initial_cycles)
        spin_box.setKeyboardTracking(False)
        control_layout.addWidget(spin_box)
        self.e_period_spinbox = spin_box
        self.canvas.mpl_connect("draw_event", self._position_e_period_control)
        self._position_e_period_control()
        self.e_period_control.show()
        return spin_box

    def _position_e_period_control(self, _event=None):
        if self.e_period_control is None:
            return
        axes_position = self.axes[2].get_position()
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        control_width = self.e_period_control.sizeHint().width()
        control_height = self.e_period_control.sizeHint().height()
        left = round(axes_position.x1 * canvas_width - control_width - 4)
        top = round((1.0 - axes_position.y1) * canvas_height + 2)
        self.e_period_control.setGeometry(left, top, control_width, control_height)
        self.e_period_control.raise_()

    def set_curves(self, depth, signals, annotations=None):
        for ax, line, signal in zip(self.axes, self.lines, signals):
            line.set_data(depth, signal)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
        self.canvas.draw_idle()
        self._position_e_period_control()

    def _show_curve_context_menu(self, event):
        if event.button != MouseButton.RIGHT or event.inaxes not in self.axes:
            return

        curve_index = self.axes.index(event.inaxes)
        menu = QMenu(self)
        import_action = menu.addAction("导入")
        export_action = menu.addAction("导出")
        chosen = menu.exec(self.canvas.mapToGlobal(event.guiEvent.pos()))
        if chosen == import_action:
            self.curve_file_action_requested.emit(curve_index, "import")
        elif chosen == export_action:
            self.curve_file_action_requested.emit(curve_index, "export")