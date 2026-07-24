"""
force_mechanics.py — 力分解与运动趋势计算模块 (Layer 2)

本模块提供曲线上某点的基底向量计算、力分解、运动趋势向量等力学计算。
仅依赖 numpy, 以及 cylinder_geometry 的 Geom 数据结构与内部函数.

数据类:
    Basis       — 非正交基底 {tangent, normal, vertical}
    ForceDecomp — 力分解结果 {coeffs, Ft_vec, Fn_vec, Fv_vec, error}

用法示例:
    from cylinder_geometry import sample_intersection
    from force_mechanics import compute_point_basis, decompose_force

    geom = sample_intersection('Y', (0,0,0), 10, 'Z', (27,0,0), 20)
    P = geom.sample_pts[0]
    basis = compute_point_basis(P, geom)
    decomp = decompose_force(np.array([5, -1, -8]), basis)
"""

import numpy as np
from dataclasses import dataclass
from cylinder_geometry import _get_branch_meta, _AXIS_DIR


# ============================================================
# 数据类
# ============================================================

@dataclass
class Basis:
    """曲线上一点的非正交基底 (三个单位向量).

    字段
    ----
    tangent  : ndarray (3,)  — 曲线切向量
    normal   : ndarray (3,)  — 打磨法向量 (两圆柱面径向法向的角平分线)
    vertical : ndarray (3,)  — 垂直向量 (圆柱2 表面径向法向)
    """
    tangent:  np.ndarray
    normal:   np.ndarray
    vertical: np.ndarray


@dataclass
class ForceDecomp:
    """力在非正交基 {t, n, v} 上的分解结果.

    字段
    ----
    coeffs  : ndarray (3,)  — 分解系数 [a, b, c]
    Ft_vec  : ndarray (3,)  — a * tangent
    Fn_vec  : ndarray (3,)  — b * normal
    Fv_vec  : ndarray (3,)  — c * vertical
    error   : float         — 重建误差 (应为 ~0)
    """
    coeffs:  np.ndarray
    Ft_vec:  np.ndarray
    Fn_vec:  np.ndarray
    Fv_vec:  np.ndarray
    error:   float


# ============================================================
# 解析切向量（内部）
# ============================================================

def _analytic_tangent(t, s1, s2, is_t_up, R1, R2, c1, c2, common):
    """用解析公式计算参数 t 处的单位切向量."""
    eps = 1e-12
    s1_val = np.sqrt(max(eps, R1**2 - (t - c1[common])**2))
    s2_val = np.sqrt(max(eps, R2**2 - (t - c2[common])**2))

    d_other1_dt = -s1 * (t - c1[common]) / s1_val
    d_other2_dt = -s2 * (t - c2[common]) / s2_val

    tangent = np.array([1.0, d_other2_dt, d_other1_dt])
    if not is_t_up:
        tangent = -tangent

    norm = np.linalg.norm(tangent)
    if norm > 1e-12:
        tangent /= norm
    return tangent


# ============================================================
# 基底向量计算
# ============================================================

def compute_point_basis(P, geom):
    """计算曲线上一点 P 处的非正交基底.

    内部自动定位 P 所属的分支号和局部索引, 无需调用者提供.

    参数
    ----
    P : ndarray (3,)
        采样点的空间坐标 (必须在曲线上).
    geom : Geom
        sample_intersection() 或 resample_curve() 的返回值.

    返回
    ----
    Basis
    """
    # 仅获取分支元信息 (不计算完整曲线几何)
    meta = _get_branch_meta(geom.axis1, geom.axis2,
                            geom.c1, geom.c2, geom.r1, geom.r2, N=250)
    c1, c2 = meta['c1'], meta['c2']
    common, other1, other2 = meta['common'], meta['other1'], meta['other2']
    d1 = _AXIS_DIR[geom.axis1]
    d2 = _AXIS_DIR[geom.axis2]
    R1, R2 = geom.r1, geom.r2

    # 自动定位分支: 对每个分支重建期望点, 找与 P 最近的分支
    t_val = P[common]
    best_bid, best_dist = None, float('inf')
    for bid, (s1, s2, is_t_up) in enumerate(meta['branch_info']):
        other1_val = c1[other1] + s1 * np.sqrt(max(0, R1**2 - (t_val - c1[common])**2))
        other2_val = c2[other2] + s2 * np.sqrt(max(0, R2**2 - (t_val - c2[common])**2))
        expected = np.zeros(3)
        expected[common] = t_val
        expected[other1] = other1_val
        expected[other2] = other2_val
        dist = np.linalg.norm(P - expected)
        if dist < best_dist:
            best_dist = dist
            best_bid = bid

    # 在分支内搜索最近参数索引
    local_j = np.argmin(np.abs(meta['curves_t'][best_bid] - t_val))
    s1, s2, is_t_up = meta['branch_info'][best_bid]
    t = float(meta['curves_t'][best_bid][local_j])

    # 圆柱1 / 圆柱2 径向向外法向量
    r1 = P - c1 - np.dot(P - c1, d1) * d1
    n1 = r1 / np.linalg.norm(r1)
    r2 = P - c2 - np.dot(P - c2, d2) * d2
    n2 = r2 / np.linalg.norm(r2)

    # 打磨法向量 = (n1 + n2) 归一化
    normal = n1 + n2
    normal /= np.linalg.norm(normal)

    # 解析切向量
    tangent = _analytic_tangent(t, s1, s2, is_t_up, R1, R2, c1, c2, common)

    return Basis(tangent=tangent, normal=normal, vertical=n2.copy())


# ============================================================
# 力分解
# ============================================================

def decompose_force(F, basis):
    """将外力 F 在非正交基 {t, n, v} 上分解.

    解 M * coeffs = F, 其中 M = [t | n | v].

    参数
    ----
    F : ndarray (3,)
        外力向量.
    basis : Basis
        compute_point_basis() 的返回值.

    返回
    ----
    ForceDecomp
    """
    t, n, v = basis.tangent, basis.normal, basis.vertical
    M = np.column_stack([t, n, v])
    coeffs = np.linalg.solve(M, F)
    a, b, c = coeffs

    Ft = a * t
    Fn = b * n
    Fv = c * v

    return ForceDecomp(
        coeffs=coeffs,
        Ft_vec=Ft, Fn_vec=Fn, Fv_vec=Fv,
        error=np.linalg.norm(F - (Ft + Fn + Fv)),
    )


# ============================================================
# 期望力 (由系数构造)
# ============================================================

def expected_force(coeffs, basis):
    """由给定分解系数反向构造力向量.

    公式: F = a*t + b*n + c*v

    参数
    ----
    coeffs : ndarray (3,) 或 tuple
        分解系数 [a, b, c].
    basis : Basis

    返回
    ----
    ForceDecomp
    """
    a, b, c = coeffs
    t, n, v = basis.tangent, basis.normal, basis.vertical
    Ft, Fn, Fv = a * t, b * n, c * v

    return ForceDecomp(
        coeffs=np.asarray(coeffs, dtype=float),
        Ft_vec=Ft, Fn_vec=Fn, Fv_vec=Fv,
        error=0.0,
    )


# ============================================================
# 运动趋势向量
# ============================================================

def compute_normal_motion_trend(decomp, basis, offset=8.0):
    """法向力运动趋势向量.

    方向: 与打磨法向量 n 相同.
    大小: 法向系数 + offset.

    公式: F_motion_n = (coeffs[1] + offset) * n

    参数
    ----
    decomp : ForceDecomp
    basis  : Basis
    offset : float (默认 8.0)

    返回
    ----
    ndarray (3,)
    """
    return (decomp.coeffs[1] + offset) * basis.normal


def compute_vertical_motion_trend(decomp, basis):
    """垂直力运动趋势向量.

    方向: -(t × v) = v × t
    大小: |垂直系数|.

    公式: F_motion_v = |coeffs[2]| * unit(-(t × v))

    参数
    ----
    decomp : ForceDecomp
    basis  : Basis

    返回
    ----
    ndarray (3,)
    """
    direction = -np.cross(basis.tangent, basis.vertical)
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction /= norm
    return abs(decomp.coeffs[2]) * direction
