"""
cylinder_fitting.py — Layer 0：从散点拟合圆柱参数

仅依赖 numpy + cylinder_geometry.Geom
每个圆柱做投影圆拟合（3参数：圆心2D + 半径），两个圆柱分开拟合。

对外接口：
    fit_cylinders_from_points(pts, axis1, axis2)
        → (list[CylinderParams], Geom)

数据类：
    CylinderParams — 完整拟合结果（供展示）
    Geom          — 圆柱几何参数（1位小数，0采样点，供 resample_curve）
"""

import numpy as np
from dataclasses import dataclass
from cylinder_geometry import Geom


# ============================================================
# 数据类
# ============================================================

@dataclass
class CylinderParams:
    """单个圆柱的拟合结果（完整精度）。

    字段
    ----
    axis       : str            — 轴线方向 'X'/'Y'/'Z'
    axis_point : np.ndarray (3,)  — 轴线上一点坐标
    radius     : float          — 半径 (mm)
    rms        : float          — RMS 残差 (mm)
    max_err    : float          — 最大残差 (mm)
    residuals  : np.ndarray (N,)  — 各点残差
    """
    axis: str
    axis_point: np.ndarray
    radius: float
    rms: float
    max_err: float
    residuals: np.ndarray


# ============================================================
# 内部：单圆柱投影圆拟合
# ============================================================

def _fit_projection(pts: np.ndarray, axis: str) -> CylinderParams:
    """对一个圆柱做投影圆拟合（线性最小二乘，3参数）。

    将散点投影到 ┴ axis 的平面，拟合圆 (u - U0)² + (v - V0)² = r²。

    参数
    ----
    pts : ndarray (N, 3)
        散点坐标。
    axis : str
        圆柱轴线方向 'X'/'Y'/'Z'。

    返回
    ----
    CylinderParams
    """
    # 坐标映射
    idx_map = {
        'X': (0, [1, 2]),   # 投影轴 = 0，拟合平面坐标 = [1, 2] → (Y, Z)
        'Y': (1, [0, 2]),   # 投影轴 = 1，拟合平面坐标 = [0, 2] → (X, Z)
        'Z': (2, [0, 1]),   # 投影轴 = 2，拟合平面坐标 = [0, 1] → (X, Y)
    }
    ax_idx, plane_idx = idx_map[axis]
    u = pts[:, plane_idx[0]]  # 第一个平面坐标
    v = pts[:, plane_idx[1]]  # 第二个平面坐标

    # 圆方程线性化：(u - U0)² + (v - V0)² = r²
    # → u² + v² = 2U0·u + 2V0·v + (r² - U0² - V0²)
    # → Y = [u, v, 1] · [2U0, 2V0, r² - U0² - V0²]
    Y = u**2 + v**2
    A = np.column_stack([u, v, np.ones_like(u)])
    coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]

    U0 = coeffs[0] / 2
    V0 = coeffs[1] / 2
    R = np.sqrt(coeffs[2] + U0**2 + V0**2)

    # 重构三维轴心：被投影的坐标用均值填充
    axis_point = np.zeros(3)
    axis_point[plane_idx[0]] = U0
    axis_point[plane_idx[1]] = V0
    axis_point[ax_idx] = pts[:, ax_idx].mean()

    # 残差
    residuals = np.sqrt((u - U0)**2 + (v - V0)**2) - R

    return CylinderParams(
        axis=axis,
        axis_point=axis_point,
        radius=float(R),
        rms=float(np.sqrt(np.mean(residuals**2))),
        max_err=float(np.max(np.abs(residuals))),
        residuals=residuals,
    )


# ============================================================
# 对外接口
# ============================================================

def fit_cylinders_from_points(pts: np.ndarray, axis1: str, axis2: str) -> tuple:
    """从散点拟合两个圆柱，返回 CylinderParams 列表和 Geom 对象。

    Geom 参数保留 1 位小数，0 个采样点，可直接传入 resample_curve()。

    参数
    ----
    pts : ndarray (N, 3)
        散点坐标（通常为实测接触点）。
    axis1, axis2 : str
        两圆柱轴线方向，必须不同 ('X'/'Y'/'Z')。

    返回
    ----
    (list[CylinderParams], Geom)
        - params[0], params[1] — 完整拟合结果（供展示）
        - geom — 精简 Geom（1 位小数），喂给 resample_curve()
    """
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 must be different.")

    params1 = _fit_projection(pts, axis1)
    params2 = _fit_projection(pts, axis2)

    # 构建 Geom：参数保留 1 位小数
    geom = Geom(
        n_samples=0,
        sample_pts=np.empty((0, 3)),
        axis1=axis1,
        c1=tuple(round(float(x), 1) for x in params1.axis_point),
        r1=round(params1.radius, 1),
        axis2=axis2,
        c2=tuple(round(float(x), 1) for x in params2.axis_point),
        r2=round(params2.radius, 1),
    )

    return [params1, params2], geom
