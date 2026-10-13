from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QMenu
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class DepthCanvas(FigureCanvasQTAgg):
    def __init__(self, title: str, xlabel: str):
        self.figure = Figure(tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setMinimumWidth(180)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel("深度（米）")
        self.axes.grid(True, alpha=0.2)

    def set_depth_limits(self, depth_min: float, depth_max: float) -> None:
        self.axes.set_ylim(depth_max, depth_min)
        self.draw_idle()


class WellLogCanvas(DepthCanvas):
    import_requested = pyqtSignal()

    def __init__(self):
        super().__init__("原始测井曲线与旋回曲线", "信号幅值")
        self.axes.set_title("原始测井曲线与旋回曲线")

    def plot_log(self, depth, gr, cycle) -> None:
        self.axes.clear()
        cycle_min = float(cycle.min())
        cycle_span = float(cycle.max() - cycle_min)
        gr_min = float(gr.min())
        gr_span = float(gr.max() - gr_min)
        if cycle_span > 0 and gr_span > 0:
            cycle_display = gr_min + (cycle - cycle_min) / cycle_span * gr_span
        else:
            cycle_display = gr_min + cycle * 0
        self.axes.plot(gr, depth, color="#176b87", linewidth=0.9, label="原始测井曲线")
        self.axes.plot(cycle_display, depth, color="#c25240", linewidth=1.0, label="旋回曲线（归一化）")
        self.axes.set_title("原始测井曲线与旋回曲线")
        self.axes.set_xlabel("信号幅值")
        self.axes.set_ylabel("深度（米）")
        self.axes.grid(True, alpha=0.2)
        self.axes.legend(loc="best")
        self.axes.set_ylim(float(depth.max()), float(depth.min()))
        self.draw_idle()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            import_action = menu.addAction("导入txt文件")
            selected = menu.exec(event.globalPosition().toPoint())
            if selected is import_action:
                self.import_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class HeatmapCanvas(DepthCanvas):
    def __init__(self):
        super().__init__("深度-旋回周期热力图", "旋回周期（m/cycle）")
        self.axes.set_xlabel("旋回周期（m/cycle）")