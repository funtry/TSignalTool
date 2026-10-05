import os
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from widgets.signal_process_widget import SignalProcessWidget


class SignalComposeWidget(SignalProcessWidget):
    """Compose seven imported tracks into one well-log curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._compose_grid = None
        self._imported_tracks = set()
        self.setWindowTitle("测井曲线合成")

    def _handle_curve_file_action(self, curve_index, action):
        if action == "export":
            return super()._handle_curve_file_action(curve_index, action)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"导入 {self.track_labels[curve_index]} 曲线",
            os.getcwd(),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            imported = np.loadtxt(file_path, dtype=float, ndmin=2)
        except (OSError, ValueError):
            QMessageBox.warning(self, "数据格式不正确", "文件必须包含两列深度和曲线值。")
            return

        depth = imported[:, 0]
        values = imported[:, 1] if imported.ndim == 2 and imported.shape[1] == 2 else None
        if (
            imported.ndim != 2
            or imported.shape[1] != 2
            or imported.shape[0] < 2
            or not np.all(np.isfinite(imported))
            or np.any(np.diff(depth) <= 0)
        ):
            QMessageBox.warning(
                self,
                "数据格式不正确",
                "文件必须包含至少两行有限数值，深度和值各一列，且深度严格递增。",
            )
            return

        if self._compose_grid is None or not self._same_grid(self._compose_grid, depth):
            self._curve_overrides.clear()
            self._imported_tracks.clear()
            self._compose_grid = depth.copy()
            self.depth = depth.copy()
            self.well_log_canvas.selected_indices = (0, depth.size - 1)

        self._curve_overrides[curve_index] = values.copy()
        self._imported_tracks.add(curve_index)
        self._refresh_imported_tracks()

    @staticmethod
    def _same_grid(first, second):
        return first.shape == second.shape and np.allclose(
            first,
            second,
            rtol=0.0,
            atol=1e-8,
        )

    def _refresh_imported_tracks(self):
        if self._compose_grid is None:
            return

        count = self.depth.size
        if len(self._imported_tracks) == len(self.track_labels):
            center = float((self.depth[0] + self.depth[-1]) / 2.0)
            sigma = max(float(np.ptp(self.depth) / 6.0), float(np.diff(self.depth).mean()))
            self._update_curve_arrays(center, sigma)
            self.well_log_canvas.set_comparison_signal(None)
            self.well_log_canvas.set_data(self.depth, self.total_signal)
            self.curve_canvas.set_curves(self.depth, self.signal_curves, self.track_annotations)
            self._update_spectrum_for_selected_depth()
            self.depth_range_label.setText(
                f"已导入全部 7 条轨道并完成合成；{count} 个采样点。"
            )
            return

        empty = np.zeros(count, dtype=float)
        self.signal_curves = [
            self._curve_overrides.get(index, empty) for index in range(len(self.track_labels))
        ]
        self.total_signal = empty.copy()
        self.track_annotations = tuple(
            "Imported TXT" if index in self._imported_tracks else None
            for index in range(len(self.track_labels))
        )
        self.well_log_canvas.set_comparison_signal(None)
        self.well_log_canvas.set_data(self.depth, self.total_signal)
        self.curve_canvas.set_curves(self.depth, self.signal_curves, self.track_annotations)
        self.spectrum_canvas.results = {}
        self.spectrum_canvas._render_current_result(self.spectrum_canvas.method_combo.currentText())
        self.depth_range_label.setText(
            f"已导入 {len(self._imported_tracks)}/7 条轨道；深度网格 {self.depth[0]:.6g} 至 "
            f"{self.depth[-1]:.6g}，{count} 个采样点。"
        )

    def _refresh_signal_for_selected_depth(self):
        self._signal_refresh_pending = False
        if len(self._imported_tracks) == len(self.track_labels):
            super()._refresh_signal_for_selected_depth()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = SignalComposeWidget()
    widget.resize(1400, 900)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())