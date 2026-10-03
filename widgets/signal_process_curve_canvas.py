from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QMenu, QSizePolicy, QVBoxLayout, QWidget


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._show_curve_context_menu)

    def set_curves(self, depth, signals, annotations=None):
        for ax, line, signal in zip(self.axes, self.lines, signals):
            line.set_data(depth, signal)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
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