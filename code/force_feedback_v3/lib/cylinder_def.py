"""
cylinder_def.py — 圆柱定义数据类 (v2)

用两点确定轴线 + 半径，替代旧的 (axis_char, axis_point, radius)。
"""

import numpy as np
from dataclasses import dataclass

_AXIS_DIR = {
    'X': np.array([1., 0., 0.]),
    'Y': np.array([0., 1., 0.]),
    'Z': np.array([0., 0., 1.]),
}

_AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}


@dataclass
class CylinderDef:
    """由轴线上两点 + 半径定义的圆柱。

    Fields
    ------
    p1, p2 : ndarray (3,)
        轴线上两点。p1 ≠ p2，方向为 p2 - p1。
    radius : float
        圆柱半径 (mm)。
    """
    p1: np.ndarray
    p2: np.ndarray
    radius: float

    def __post_init__(self):
        self.p1 = np.asarray(self.p1, dtype=float)
        self.p2 = np.asarray(self.p2, dtype=float)
        d = self.p2 - self.p1
        if np.linalg.norm(d) < 1e-12:
            raise ValueError("p1 and p2 must be distinct points on the axis.")

    # ---- 派生属性 ----

    @property
    def direction(self) -> np.ndarray:
        """轴线单位方向向量 (p1 → p2)。"""
        d = self.p2 - self.p1
        return d / np.linalg.norm(d)

    @property
    def nearest_axis(self) -> str:
        """最接近的坐标轴方向 'X'|'Y'|'Z'。"""
        d = self.direction
        idx = int(np.argmax(np.abs(d)))
        return ['X', 'Y', 'Z'][idx]

    @property
    def axis_point(self) -> np.ndarray:
        """轴线上一点（p1，保持与旧接口兼容）。"""
        return self.p1.copy()

    @property
    def axis_dir_vector(self) -> np.ndarray:
        """坐标轴方向的单位向量（最近坐标轴）。"""
        return _AXIS_DIR[self.nearest_axis]

    # ---- 工厂方法 ----

    @classmethod
    def from_axis_aligned(
        cls,
        axis: str,
        axis_point_like,
        radius: float,
        pts_range: tuple = None,
    ) -> "CylinderDef":
        """从轴对齐参数创建 CylinderDef。

        用于 L0 拟合输出：已知轴方向 + 轴心点 + 半径，
        用数据沿轴范围确定 p1/p2。

        Parameters
        ----------
        axis : str
            轴线方向 'X'|'Y'|'Z'。
        axis_point_like : array-like (3,)
            轴心上一点（垂直平面坐标 + 轴向均值）。
        radius : float
            半径。
        pts_range : tuple (min, max), optional
            数据沿轴向的最小/最大值。若未提供，p1/p2 = axis_point ± 1.0。

        Returns
        -------
        CylinderDef
        """
        axis_point = np.asarray(axis_point_like, dtype=float)
        ax_idx = _AXIS_IDX[axis]
        direction = _AXIS_DIR[axis]

        if pts_range is not None:
            t_min, t_max = pts_range
        else:
            t = axis_point[ax_idx]
            t_min, t_max = t - 1.0, t + 1.0

        p1 = axis_point.copy()
        p1[ax_idx] = t_min
        p2 = axis_point.copy()
        p2[ax_idx] = t_max

        return cls(p1=p1, p2=p2, radius=radius)

    @classmethod
    def from_two_points(
        cls,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float,
    ) -> "CylinderDef":
        """从任两点 + 半径直接创建（用于误差仿真等场景）。"""
        return cls(p1=p1, p2=p2, radius=radius)
