import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)


class CurveJointDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("曲线拼接")
        self.resize(1200, 760)

        self._curves = []
        self._joined_depth = np.asarray([], dtype=float)
        self._joined_values = np.asarray([], dtype=float)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._update_controls)

        self.add_button = QPushButton("添加 TXT")
        self.up_button = QPushButton("上移")
        self.down_button = QPushButton("下移")
        self.remove_button = QPushButton("移除")
        self.clear_button = QPushButton("清空")
        self.export_button = QPushButton("导出拼接曲线")
        self.close_button = QPushButton("关闭")

        self.add_button.clicked.connect(self._add_curves)
        self.up_button.clicked.connect(lambda: self._move_curve(-1))
        self.down_button.clicked.connect(lambda: self._move_curve(1))
        self.remove_button.clicked.connect(self._remove_curve)
        self.clear_button.clicked.connect(self._clear_curves)
        self.export_button.clicked.connect(self._export_joined_curve)
        self.close_button.clicked.connect(self.reject)

        list_controls = QVBoxLayout()
        list_controls.addWidget(QLabel("输入曲线（从上到下为导入顺序）"))
        list_controls.addWidget(self.file_list, 1)
        for button in (
            self.add_button,
            self.up_button,
            self.down_button,
            self.remove_button,
            self.clear_button,
        ):
            list_controls.addWidget(button)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.status_label = QLabel("按深度排序；重叠区间由后导入的曲线覆盖。")
        self.status_label.setWordWrap(True)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.status_label)

        body_layout = QHBoxLayout()
        body_layout.addLayout(list_controls, 0)
        body_layout.addLayout(plot_layout, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self.export_button)
        footer.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(body_layout, 1)
        layout.addLayout(footer)

        self._render_preview()
        self._update_controls()

    def _add_curves(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "添加待拼接曲线",
            os.getcwd(),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_paths:
            return

        loaded = []
        for file_path in file_paths:
            try:
                data = np.loadtxt(file_path, dtype=float, ndmin=2)
            except (OSError, ValueError) as error:
                QMessageBox.warning(
                    self,
                    "无法导入曲线",
                    f"{Path(file_path).name} 读取失败：{error}\n"
                    "文件应为无表头的两列数值：深度和值。",
                )
                continue

            if (
                data.ndim != 2
                or data.shape[0] < 2
                or data.shape[1] < 2
                or not np.all(np.isfinite(data[:, :2]))
                or np.any(np.diff(data[:, 0]) <= 0)
            ):
                QMessageBox.warning(
                    self,
                    "数据格式不正确",
                    f"{Path(file_path).name} 必须至少有两行有限数值，"
                    "前两列为严格递增的深度和值。",
                )
                continue

            loaded.append(
                {
                    "path": file_path,
                    "depth": data[:, 0].copy(),
                    "values": data[:, 1].copy(),
                }
            )

        if not loaded:
            return

        self._curves.extend(loaded)
        for curve in loaded:
            self.file_list.addItem(
                f"{Path(curve['path']).name}  "
                f"({curve['depth'][0]:g}–{curve['depth'][-1]:g} m)"
            )
        self.file_list.setCurrentRow(self.file_list.count() - 1)
        self._refresh_joined_curve()

    def _move_curve(self, offset):
        row = self.file_list.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < len(self._curves):
            return

        curve = self._curves.pop(row)
        self._curves.insert(target, curve)
        item = self.file_list.takeItem(row)
        self.file_list.insertItem(target, item)
        self.file_list.setCurrentRow(target)
        self._refresh_joined_curve()

    def _remove_curve(self):
        row = self.file_list.currentRow()
        if row < 0:
            return
        self._curves.pop(row)
        self.file_list.takeItem(row)
        if self.file_list.count():
            self.file_list.setCurrentRow(min(row, self.file_list.count() - 1))
        self._refresh_joined_curve()

    def _clear_curves(self):
        self._curves.clear()
        self.file_list.clear()
        self._refresh_joined_curve()

    def _refresh_joined_curve(self):
        depth = np.asarray([], dtype=float)
        values = np.asarray([], dtype=float)

        for curve in self._curves:
            new_depth = curve["depth"]
            new_values = curve["values"]
            if depth.size:
                outside_new_range = (depth < new_depth[0]) | (depth > new_depth[-1])
                depth = depth[outside_new_range]
                values = values[outside_new_range]
            depth = np.concatenate((depth, new_depth))
            values = np.concatenate((values, new_values))

        if depth.size:
            order = np.argsort(depth, kind="stable")
            depth = depth[order]
            values = values[order]
            keep = np.concatenate(([True], np.diff(depth) > 0))
            depth = depth[keep]
            values = values[keep]

        self._joined_depth = depth
        self._joined_values = values
        self._render_preview()
        self._update_controls()

    def _render_preview(self):
        self.axes.clear()
        if self._joined_depth.size:
            self.axes.plot(
                self._joined_depth,
                self._joined_values,
                color="#0000FF",
                linewidth=1.1,
            )
            self.axes.set_title("拼接曲线预览")
            self.axes.set_xlabel("深度 (m)")
            self.axes.set_ylabel("曲线值")
            self.axes.grid(True, alpha=0.25)
            self.status_label.setText(
                f"{len(self._curves)} 条输入曲线；"
                f"拼接后 {self._joined_depth.size} 个采样点，"
                f"深度范围 {self._joined_depth[0]:g}–"
                f"{self._joined_depth[-1]:g} m。"
                "重叠区间由后导入曲线覆盖，空档不插值。"
            )
        else:
            self.axes.set_title("拼接曲线预览")
            self.axes.set_xlabel("深度 (m)")
            self.axes.set_ylabel("曲线值")
            self.axes.text(
                0.5,
                0.5,
                "添加两列 TXT 曲线以预览拼接结果",
                transform=self.axes.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
            self.axes.grid(True, alpha=0.25)
            self.status_label.setText("按深度排序；重叠区间由后导入的曲线覆盖。")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_controls(self, _row=None):
        row = self.file_list.currentRow()
        count = len(self._curves)
        self.up_button.setEnabled(0 < row < count)
        self.down_button.setEnabled(0 <= row < count - 1)
        self.remove_button.setEnabled(row >= 0)
        self.clear_button.setEnabled(count > 0)
        self.export_button.setEnabled(self._joined_depth.size > 0)

    def _export_joined_curve(self):
        if self._joined_depth.size == 0:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出拼接曲线",
            "拼接曲线.txt",
            "文本文件 (*.txt)",
        )
        if not file_path:
            return

        try:
            np.savetxt(
                file_path,
                np.column_stack((self._joined_depth, self._joined_values)),
                fmt="%.10f",
            )
        except OSError as error:
            QMessageBox.warning(self, "导出失败", str(error))
            return
        self.status_label.setText(
            f"已导出 {self._joined_depth.size} 个采样点：{Path(file_path).name}"
        )


def main():
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    dialog = CurveJointDialog()
    dialog.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
