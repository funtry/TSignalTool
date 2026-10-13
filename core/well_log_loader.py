from pathlib import Path

import numpy as np


def load_well_log(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取文本文件前两列为深度和 GR，并按深度升序整理。"""
    try:
        data = np.loadtxt(file_path, usecols=(0, 1), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取两列数值数据：{exc}") from exc

    data = np.atleast_2d(data)
    if data.shape[0] < 3:
        raise ValueError("测井数据至少需要 3 行有效数据。")

    valid = np.all(np.isfinite(data), axis=1)
    data = data[valid]
    if data.shape[0] < 3:
        raise ValueError("有效深度-GR 数据不足 3 行。")

    order = np.argsort(data[:, 0])
    depth, gr = data[order, 0], data[order, 1]
    depth, unique_indices = np.unique(depth, return_index=True)
    gr = gr[unique_indices]
    if depth.size < 3:
        raise ValueError("至少需要 3 个不同的深度值。")
    return depth, gr