from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
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

        self.phase_marker = self.axes[2].axvline(
            0.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.85,
            visible=False, zorder=4,
        )

        self.axes[-1].set_xlabel("Depth (m)")
        self.fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.10, hspace=0.08)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.e_wavelength_spinbox = None
        self.gaussian_filter_spinbox = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._show_curve_context_menu)
        self.trend_smoothing_slider = None
        self.trend_smoothing_spinbox = None
        self.trend_window_spinbox = None
        self.trend_control = None
        self.trend_window_control = None
        self.track_controls = {}

    def _position_control(self, control, index, _event=None):
        if control is None:
            return
        axes_position = self.axes[index].get_position()
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        control_width = control.sizeHint().width()
        control_height = control.sizeHint().height()
        left = round(axes_position.x1 * canvas_width - control_width - 4)
        top = round((1.0 - axes_position.y1) * canvas_height + 2)
        control.setGeometry(left, top, control_width, control_height)
        control.raise_()

    def _position_trend_controls(self):
        if self.trend_control is None or self.trend_window_control is None:
            return
        axes_position = self.axes[0].get_position()
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        slider_height = self.trend_control.sizeHint().height()
        input_width = self.trend_window_control.sizeHint().width()
        slider_left = round(axes_position.x0 * canvas_width + 4)
        slider_top = max(0, round((1.0 - axes_position.y1) * canvas_height - slider_height - 4))
        slider_width = max(1, round((axes_position.x1 - axes_position.x0) * canvas_width - 8))
        self.trend_control.setGeometry(slider_left, slider_top, slider_width, slider_height)
        input_left = round(axes_position.x1 * canvas_width - input_width - 4)
        input_top = round((1.0 - axes_position.y1) * canvas_height + 2)
        self.trend_window_control.setGeometry(input_left, input_top, input_width, slider_height)
        self.trend_control.raise_()
        self.trend_window_control.raise_()

    def add_trend_smoothing_control(self, initial_value=15):
        if self.trend_smoothing_slider is not None:
            return self.trend_smoothing_slider

        self.trend_control = QWidget(self.canvas)
        self.trend_control_layout = QHBoxLayout(self.trend_control)
        self.trend_control_layout.setContentsMargins(2, 1, 2, 1)
        self.trend_control_layout.setSpacing(4)

        self.trend_smoothing_slider = QSlider(Qt.Orientation.Horizontal, self.trend_control)
        self.trend_smoothing_slider.setRange(1, 100)
        self.trend_smoothing_slider.setValue(initial_value)
        self.trend_percentage_label = QLabel(f"{initial_value}%", self.trend_control)
        self.trend_percentage_label.setFixedWidth(42)
        self.trend_control_layout.addWidget(self.trend_percentage_label)
        self.trend_control_layout.addWidget(self.trend_smoothing_slider, 1)

        self.trend_window_control = QWidget(self.canvas)
        input_layout = QHBoxLayout(self.trend_window_control)
        input_layout.setContentsMargins(2, 1, 2, 1)
        input_layout.setSpacing(3)
        self.trend_window_spinbox = QDoubleSpinBox(self.trend_window_control)
        self.trend_window_spinbox.setRange(0.01, 10000.0)
        self.trend_window_spinbox.setDecimals(2)
        self.trend_window_spinbox.setSingleStep(0.1)
        self.trend_window_spinbox.setKeyboardTracking(False)
        self.trend_window_spinbox.setFixedWidth(90)
        input_layout.addWidget(self.trend_window_spinbox)

        self.trend_window_control.setStyleSheet("QWidget { background: rgba(255, 255, 255, 180); }")
        self.trend_control.setStyleSheet("QWidget { background: rgba(255, 255, 255, 180); }")

        self.trend_smoothing_slider.valueChanged.connect(self._on_trend_percentage_changed)
        self.trend_window_spinbox.valueChanged.connect(self._on_trend_window_changed)
        self.canvas.mpl_connect("draw_event", lambda event: self._position_trend_controls())
        self._position_trend_controls()
        self.trend_control.show()
        self.trend_window_control.show()
        return self.trend_smoothing_slider

    def _on_trend_percentage_changed(self, value):
        self.trend_percentage_label.setText(f"{value}%")
        if self.trend_window_spinbox is None:
            return
        self.trend_window_spinbox.blockSignals(True)
        self.trend_window_spinbox.setValue(float(value) / 100.0 * self.trend_window_depth)
        self.trend_window_spinbox.blockSignals(False)

    def _on_trend_window_changed(self, value):
        if self.trend_smoothing_slider is None or self.trend_window_depth <= 0:
            return
        percentage = max(1, min(100, int(round(value / self.trend_window_depth * 100.0))))
        self.trend_smoothing_slider.blockSignals(True)
        self.trend_smoothing_slider.setValue(percentage)
        self.trend_smoothing_slider.blockSignals(False)

    def set_trend_window_depth(self, depth_span):
        self.trend_window_depth = float(depth_span)
        if self.trend_window_spinbox is None or self.trend_smoothing_slider is None:
            return
        self.trend_window_spinbox.blockSignals(True)
        self.trend_window_spinbox.setValue(float(depth_span) * self.trend_smoothing_slider.value() / 100.0)
        self.trend_window_spinbox.blockSignals(False)

    def add_e_wavelength_control(self, initial_wavelength=6.0):
        if self.e_wavelength_spinbox is not None:
            return self.e_wavelength_spinbox

        control = QWidget(self.canvas)
        layout = QHBoxLayout(control)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)
        spin_box = QDoubleSpinBox(control)
        spin_box.setRange(0.01, 10000.0)
        spin_box.setDecimals(2)
        spin_box.setSingleStep(0.1)
        spin_box.setSuffix(" m")
        spin_box.setValue(float(initial_wavelength))
        spin_box.setKeyboardTracking(False)
        spin_box.setFixedWidth(90)
        layout.addWidget(spin_box)
        control.setStyleSheet("QWidget { background: rgba(255, 255, 255, 180); }")
        self.e_wavelength_spinbox = spin_box
        self.track_controls[2] = control
        self.canvas.mpl_connect("draw_event", lambda event: self._position_control(control, 2))
        self._position_control(control, 2)
        control.show()
        return spin_box

    def add_gaussian_filter_control(self, initial_value=1.0):
        if self.gaussian_filter_spinbox is not None:
            return self.gaussian_filter_spinbox

        control = QWidget(self.canvas)
        layout = QHBoxLayout(control)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)
        spin_box = QDoubleSpinBox(control)
        spin_box.setRange(0.1, 100.0)
        spin_box.setDecimals(2)
        spin_box.setSingleStep(0.1)
        spin_box.setValue(initial_value)
        spin_box.setKeyboardTracking(False)
        spin_box.setFixedWidth(90)
        layout.addWidget(spin_box)
        control.setStyleSheet("QWidget { background: rgba(255, 255, 255, 180); }")
        self.gaussian_filter_spinbox = spin_box
        self.track_controls[6] = control
        self.canvas.mpl_connect("draw_event", lambda event: self._position_control(control, 6))
        self._position_control(control, 6)
        control.show()
        return spin_box

    def set_curves(self, depth, signals, annotations=None):
        depth_min = float(min(depth))
        depth_max = float(max(depth))
        for ax, line, signal in zip(self.axes, self.lines, signals):
            line.set_data(depth, signal)
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=True)
        self.axes[0].set_xlim(depth_min, depth_max)
        self.canvas.draw_idle()
        if self.trend_control is not None:
            self._position_trend_controls()
        for index, control in self.track_controls.items():
            self._position_control(control, index)

    def set_phase_marker(self, depth):
        self.phase_marker.set_xdata((depth, depth))
        self.phase_marker.set_visible(True)
        self.canvas.draw_idle()

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