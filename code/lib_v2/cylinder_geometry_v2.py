"""
cylinder_geometry_v2.py — 双圆柱相交几何计算 (v2)

用 CylinderDef(p1,p2,r) 替代旧的 (axis_char, c, r)。
支持任意轴线方向（不依赖坐标轴对齐），内部在 {u1,u2,u3} 基下求解。

对外接口:
    sample_intersection(cyl1, cyl2, n_samples, N_curve) → GeomV2
    resample_curve(geom, n_samples) → GeomV2

数据类:
    GeomV2 — 采样结果 + 两个 CylinderDef
"""

import numpy as np
from dataclasses import dataclass
from cylinder_def import CylinderDef


# ============================================================
# 数据类
# ============================================================

@dataclass
class GeomV2:
    """两圆柱相交曲线的均匀弧长采样结果 (v2).

    Fields
    ------
    n_samples  : int               — 采样点数量
    sample_pts : ndarray (N, 3)    — 均匀弧长采样点坐标 (世界坐标)
    cyl1, cyl2 : CylinderDef       — 两圆柱定义
    """
    n_samples:  int
    sample_pts: np.ndarray
    cyl1: CylinderDef
    cyl2: CylinderDef


# ============================================================
# 内部：基向量构造
# ============================================================

def _build_basis(d1, d2):
    """由两个轴线方向构造正交基 {u1, u2, u3}.

    u1 = d1 方向
    u2 = d2 正交于 u1 的分量
    u3 = u1 × u2（公共参数化方向）
    """
    u1 = d1 / np.linalg.norm(d1)
    u2_raw = d2 - np.dot(d2, u1) * u1
    norm = np.linalg.norm(u2_raw)
    if norm < 1e-12:
        raise ValueError("两圆柱轴线平行，无交线。")
    u2 = u2_raw / norm
    u3 = np.cross(u1, u2)
    return u1, u2, u3


def _to_basis(pts, u1, u2, u3):
    """世界坐标 → {u1,u2,u3} 基坐标。"""
    return np.column_stack([
        np.dot(pts, u1),
        np.dot(pts, u2),
        np.dot(pts, u3),
    ])


def _from_basis(pts_basis, u1, u2, u3):
    """{u1,u2,u3} 基坐标 → 世界坐标。"""
    return (pts_basis[:, 0:1] * u1 +
            pts_basis[:, 1:2] * u2 +
            pts_basis[:, 2:3] * u3)


# ============================================================
# 内部：分支元信息
# ============================================================

def _get_branch_meta_v2(cyl1, cyl2, N):
    """在 {u1,u2,u3} 基下计算分支元信息 (v2).

    返回
    ----
    dict:
        u1, u2, u3    : ndarray — 基向量 (世界坐标)
        p1_b, p2_b    : ndarray — 轴点在基坐标下的投影
        cosθ, sinθ    : float   — d2 在 u1-u2 平面上的方向
        curves_t      : list    — 各分支的 t (u3坐标) 值
        branch_info   : list    — (s1, s2, is_t_up)
    """
    u1, u2, u3 = _build_basis(cyl1.direction, cyl2.direction)

    # 轴点在基坐标下
    p1 = cyl1.p1
    p2 = cyl2.p1
    p1_b = np.array([np.dot(p1, u1), np.dot(p1, u2), np.dot(p1, u3)])
    p2_b = np.array([np.dot(p2, u1), np.dot(p2, u2), np.dot(p2, u3)])

    # d2 在 u1-u2 平面的分量
    cosθ = np.dot(cyl2.direction, u1)
    sinθ = np.dot(cyl2.direction, u2)

    r1, r2 = cyl1.radius, cyl2.radius

    # 有效的 u3 参数范围
    t_min = max(p1_b[2] - r1, p2_b[2] - r2)
    t_max = min(p1_b[2] + r1, p2_b[2] + r2)
    if t_min >= t_max:
        raise ValueError(f"两圆柱不相交：u3范围 [{t_min:.2f}, {t_max:.2f}] 为空。")

    t_up   = np.linspace(t_min, t_max, N) if N > 1 else np.array([t_min, t_max])
    t_down = np.linspace(t_max, t_min, N) if N > 1 else np.array([t_max, t_min])

    curves_t = [t_up, t_down, t_up, t_down]
    branch_info = [
        (+1, +1, True),      # s1=+1, s2=+1, t_up
        (+1, -1, False),     # s1=+1, s2=-1, t_down
        (-1, -1, True),      # s1=-1, s2=-1, t_up
        (-1, +1, False),     # s1=-1, s2=+1, t_down
    ]

    return {
        'u1': u1, 'u2': u2, 'u3': u3,
        'p1_b': p1_b, 'p2_b': p2_b,
        'cosθ': cosθ, 'sinθ': sinθ,
        'r1': r1, 'r2': r2,
        'curves_t': curves_t,
        'branch_info': branch_info,
    }


# ============================================================
# 内部：原始交线计算
# ============================================================

def _intersect_raw_v2(meta):
    """在 {u1,u2,u3} 基下求解 4 段分支曲线。"""
    u1, u2, u3 = meta['u1'], meta['u2'], meta['u3']
    p1_b = meta['p1_b']
    p2_b = meta['p2_b']
    cosθ, sinθ = meta['cosθ'], meta['sinθ']
    r1, r2 = meta['r1'], meta['r2']
    curves_t = meta['curves_t']
    branch_info = meta['branch_info']

    curves = []

    for bid, (s1, s2, _) in enumerate(branch_info):
        t_vals = curves_t[bid]  # u3 坐标

        # 圆柱 1：从 u3 坐标 t 求 u2 坐标 β
        # (β - p1u2)² + (t - p1u3)² = r1²
        d1_sq = r1**2 - (t_vals - p1_b[2])**2
        d1_sq = np.maximum(d1_sq, 0.0)
        beta = p1_b[1] + s1 * np.sqrt(d1_sq)

        # 圆柱 2：从 β 和 t 求 u1 坐标 α
        # (t - p2u3)² + ((α-p2u1)sinθ - (β-p2u2)cosθ)² = r2²
        d2_sq = r2**2 - (t_vals - p2_b[2])**2
        d2_sq = np.maximum(d2_sq, 0.0)
        numerator = (beta - p2_b[1]) * cosθ + s2 * np.sqrt(d2_sq)
        alpha = p2_b[0] + numerator / sinθ

        # 基坐标 → 世界坐标
        pts_world = (alpha[:, None] * u1 +
                     beta[:, None] * u2 +
                     t_vals[:, None] * u3)
        curves.append(pts_world)

    return curves


# ============================================================
# 内部：拼接 & 重采样
# ============================================================

def _build_closed_curve(curves):
    """将 4 段分支按连接顺序 [0,3,2,1] 拼接成闭合点列。"""
    branch_order = [0, 3, 2, 1]
    closed_pts = []

    for i, bid in enumerate(branch_order):
        pts = curves[bid]
        start = 0 if i == 0 else 1
        for j in range(start, len(pts)):
            closed_pts.append(pts[j])

    return np.array(closed_pts)


def _sample_uniform(closed_pts, n_samples):
    """对闭合点列做均匀弧长采样。"""
    M = len(closed_pts)

    diffs = np.diff(closed_pts, axis=0)
    chord_lens = np.linalg.norm(diffs, axis=1)
    chord_lens = np.append(chord_lens,
                           np.linalg.norm(closed_pts[0] - closed_pts[-1]))
    cum_len = np.cumsum(np.insert(chord_lens, 0, 0.0))
    total_len = cum_len[-1]

    target_lens = np.linspace(0, total_len, n_samples, endpoint=False)
    indices = [int(np.argmin(np.abs(cum_len[:M] - t))) for t in target_lens]

    return closed_pts[indices]


# ============================================================
# 对外接口
# ============================================================

def sample_intersection(
    cyl1: CylinderDef,
    cyl2: CylinderDef,
    n_samples: int = 1000,
    N_curve: int = 250,
) -> GeomV2:
    """计算两圆柱交线，均匀弧长采样。

    Parameters
    ----------
    cyl1, cyl2 : CylinderDef
        两圆柱定义。
    n_samples : int
        均匀采样点数（默认 1000）。
    N_curve : int
        每段分支内离散点数（默认 250）。

    Returns
    -------
    GeomV2
    """
    if cyl1.radius <= 0 or cyl2.radius <= 0:
        raise ValueError("圆柱半径必须 > 0。")

    meta = _get_branch_meta_v2(cyl1, cyl2, N_curve)
    curves = _intersect_raw_v2(meta)
    closed_pts = _build_closed_curve(curves)
    sample_pts = _sample_uniform(closed_pts, n_samples)

    return GeomV2(
        n_samples=n_samples,
        sample_pts=sample_pts,
        cyl1=cyl1,
        cyl2=cyl2,
    )


def resample_curve(geom: GeomV2, n_samples: int) -> GeomV2:
    """对已有 GeomV2 用新采样点数重新做均匀弧长采样。"""
    return sample_intersection(geom.cyl1, geom.cyl2, n_samples)
