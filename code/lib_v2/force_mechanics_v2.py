"""
force_mechanics_v2.py — 力分解与运动趋势计算 (v2)

用 contact_frame 的 {t, n, rz} 基底替代旧的 {t_a, n_平分, v}。
t = r_y × r_z（精确），n = w_y·r_y + w_z·r_z（r^(2/3) 加权），rz = Z 柱面径向。

数据类:
    Basis       — 非正交基底 {tangent, normal, vertical(=rz)}
    ForceDecomp — 力分解结果

对外接口:
    compute_point_basis(P, geom) → Basis
    decompose_force(F, basis)    → ForceDecomp
    expected_force(coeffs, basis) → ForceDecomp
    compute_normal_motion_trend(decomp, basis, offset)   → ndarray
    compute_vertical_motion_trend(decomp, basis)         → ndarray
"""

import numpy as np
from dataclasses import dataclass
from cylinder_geometry_v2 import GeomV2
from contact_frame_v2 import compute_frame


# ============================================================
# 数据类
# ============================================================

@dataclass
class Basis:
    """曲线上一点的非正交基底 {t, n, rz}。

    Fields
    ------
    tangent  : ndarray (3,) — 切向量 t = r_y × r_z
    normal   : ndarray (3,) — 法向量 n = w_y·r_y + w_z·r_z
    vertical : ndarray (3,) — Z 圆柱径向 rz（指向 Z 轴心）
    """
    tangent:  np.ndarray
    normal:   np.ndarray
    vertical: np.ndarray


@dataclass
class ForceDecomp:
    """力在非正交基 {t, n, v} 上的分解结果。

    Fields
    ------
    coeffs  : ndarray (3,)  — 分解系数 [a, b, c]
    Ft_vec  : ndarray (3,)  — a * tangent   (切向)
    Fn_vec  : ndarray (3,)  — b * normal    (法向)
    Fv_vec  : ndarray (3,)  — c * vertical  (Z径向)
    error   : float         — 重建误差 (应为 ~0)
    """
    coeffs:  np.ndarray
    Ft_vec:  np.ndarray
    Fn_vec:  np.ndarray
    Fv_vec:  np.ndarray
    error:   float


# ============================================================
# 基底向量计算
# ============================================================

def compute_point_basis(P: np.ndarray, geom: GeomV2) -> Basis:
    """计算曲线上一点 P 处的非正交基底 {t, n, rz}。

    使用 contact_frame 的加权法向量公式，无需分支定位。

    Parameters
    ----------
    P : ndarray (3,)
        采样点的空间坐标（必须在交线曲线上）。
    geom : GeomV2
        sample_intersection() 的返回值。

    Returns
    -------
    Basis
    """
    frame = compute_frame(P, geom.cyl1, geom.cyl2)
    return Basis(
        tangent=frame.tangent,
        normal=frame.normal,
        vertical=frame.radial_z,
    )


# ============================================================
# 力分解
# ============================================================

def decompose_force(F: np.ndarray, basis: Basis) -> ForceDecomp:
    """将外力 F 在非正交基 {t, n, v} 上分解。

    解 M * coeffs = F，其中 M = [t | n | v]。

    Parameters
    ----------
    F : ndarray (3,)
        外力向量。
    basis : Basis
        compute_point_basis() 的返回值。

    Returns
    -------
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
# 期望力（由系数构造）
# ============================================================

def expected_force(coeffs: np.ndarray, basis: Basis) -> ForceDecomp:
    """由给定分解系数反向构造力向量。

    F = a*t + b*n + c*v

    Parameters
    ----------
    coeffs : ndarray (3,) 或 tuple
        分解系数 [a, b, c]。
    basis : Basis

    Returns
    -------
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

def compute_normal_motion_trend(
    decomp: ForceDecomp,
    basis: Basis,
    offset: float = 8.0,
) -> np.ndarray:
    """法向力运动趋势向量。

    方向: 与法向量 n 相同。
    大小: 法向系数 + offset。

    F_motion_n = (coeffs[1] + offset) * n

    Parameters
    ----------
    decomp : ForceDecomp
    basis  : Basis
    offset : float (默认 8.0)

    Returns
    -------
    ndarray (3,)
    """
    return (decomp.coeffs[1] + offset) * basis.normal


def compute_vertical_motion_trend(
    decomp: ForceDecomp,
    basis: Basis,
) -> np.ndarray:
    """垂直力运动趋势向量。

    方向: -(t × v) = v × t
    大小: |垂直系数|。

    F_motion_v = |coeffs[2]| * unit(-(t × v))

    Parameters
    ----------
    decomp : ForceDecomp
    basis  : Basis

    Returns
    -------
    ndarray (3,)
    """
    direction = -np.cross(basis.tangent, basis.vertical)
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction /= norm
    return abs(decomp.coeffs[2]) * direction


# ============================================================
# 正交基底 {t, n, ortho} — ortho = t × n
# ============================================================

@dataclass
class BasisOrtho:
    """曲线上一点的正交基底 {t, n, ortho}。

    与非正交 Basis 的区别：第三轴改为 t × n（叉乘），保证三者两两正交。

    Fields
    ------
    tangent : ndarray (3,) — 切向量 t = r_y × r_z（不变）
    normal  : ndarray (3,) — 法向量 n = w_y·r_y + w_z·r_z（不变）
    ortho   : ndarray (3,) — t × n，正交于 tangent 和 normal
    """
    tangent: np.ndarray
    normal:  np.ndarray
    ortho:   np.ndarray


@dataclass
class ForceDecompOrtho:
    """力在正交基 {t, n, o} 上的分解结果。

    Fields
    ------
    coeffs  : ndarray (3,)  — 分解系数 [a, b, c]（点积投影）
    Ft_vec  : ndarray (3,)  — a * tangent   (切向)
    Fn_vec  : ndarray (3,)  — b * normal    (法向)
    Fo_vec  : ndarray (3,)  — c * ortho     (正交第三方向)
    error   : float         — 重建误差 (应为 ~0)
    """
    coeffs:  np.ndarray
    Ft_vec:  np.ndarray
    Fn_vec:  np.ndarray
    Fo_vec:  np.ndarray
    error:   float


def compute_point_basis_ortho(P: np.ndarray, geom: GeomV2) -> BasisOrtho:
    """计算曲线上一点 P 处的正交基底 {t, n, t×n}。

    切向量和法向量与原有非正交基底一致，第三轴 ortho = t × n。

    Parameters
    ----------
    P : ndarray (3,)
        采样点的空间坐标。
    geom : GeomV2
        sample_intersection() 的返回值。

    Returns
    -------
    BasisOrtho
    """
    frame = compute_frame(P, geom.cyl1, geom.cyl2)
    t = frame.tangent
    n = frame.normal
    o = np.cross(t, n)
    o = o / np.linalg.norm(o)
    return BasisOrtho(tangent=t, normal=n, ortho=o)


def decompose_force_ortho(F: np.ndarray, basis: BasisOrtho) -> ForceDecompOrtho:
    """将外力 F 在正交基 {t, n, o} 上分解。

    正交基直接点积投影，无需解线性方程组。

    Parameters
    ----------
    F : ndarray (3,)
        外力向量。
    basis : BasisOrtho
        compute_point_basis_ortho() 的返回值。

    Returns
    -------
    ForceDecompOrtho
    """
    t, n, o = basis.tangent, basis.normal, basis.ortho
    a = np.dot(F, t)
    b = np.dot(F, n)
    c = np.dot(F, o)

    Ft = a * t
    Fn = b * n
    Fo = c * o

    return ForceDecompOrtho(
        coeffs=np.array([a, b, c]),
        Ft_vec=Ft, Fn_vec=Fn, Fo_vec=Fo,
        error=np.linalg.norm(F - (Ft + Fn + Fo)),
    )


def expected_force_ortho(coeffs: np.ndarray, basis: BasisOrtho) -> ForceDecompOrtho:
    """由给定分解系数反向构造力向量。

    F = a*t + b*n + c*o

    Parameters
    ----------
    coeffs : ndarray (3,) 或 tuple
        分解系数 [a, b, c]。
    basis : BasisOrtho

    Returns
    -------
    ForceDecompOrtho
    """
    a, b, c = coeffs
    t, n, o = basis.tangent, basis.normal, basis.ortho
    Ft, Fn, Fo = a * t, b * n, c * o

    return ForceDecompOrtho(
        coeffs=np.asarray(coeffs, dtype=float),
        Ft_vec=Ft, Fn_vec=Fn, Fo_vec=Fo,
        error=0.0,
    )
