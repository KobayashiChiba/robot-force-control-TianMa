"""
force_mechanics_v2.py — 力分解（正交基底）

用 contact_frame 的 {t, n} + 构造 ortho = t×n，保证三者两两正交。
对外接口:
    compute_point_basis_ortho(P, geom) → BasisOrtho
    decompose_force_ortho(F, basis)    → ForceDecompOrtho
    expected_force_ortho(coeffs, basis) → ForceDecompOrtho
"""

import numpy as np
from dataclasses import dataclass
from cylinder_geometry_v2 import GeomV2
from contact_frame_v2 import compute_frame


@dataclass
class BasisOrtho:
    """正交基底 {t, n, ortho=t×n}。"""
    tangent: np.ndarray
    normal:  np.ndarray
    ortho:   np.ndarray


@dataclass
class ForceDecompOrtho:
    """力在正交基 {t, n, o} 上的分解。"""
    coeffs:  np.ndarray   # [a, b, c]
    Ft_vec:  np.ndarray   # a*t
    Fn_vec:  np.ndarray   # b*n
    Fo_vec:  np.ndarray   # c*o
    error:   float        # 重建误差 ≈0


def compute_point_basis_ortho(P: np.ndarray, geom: GeomV2) -> BasisOrtho:
    """计算正交基底 {t, n, t×n}。"""
    frame = compute_frame(P, geom.cyl1, geom.cyl2)
    t = frame.tangent
    n = frame.normal
    o = np.cross(t, n)
    o = o / np.linalg.norm(o)
    return BasisOrtho(tangent=t, normal=n, ortho=o)


def decompose_force_ortho(F: np.ndarray, basis: BasisOrtho) -> ForceDecompOrtho:
    """正交基底直接点积投影。"""
    t, n, o = basis.tangent, basis.normal, basis.ortho
    a, b, c = np.dot(F, t), np.dot(F, n), np.dot(F, o)
    Ft, Fn, Fo = a * t, b * n, c * o
    return ForceDecompOrtho(
        coeffs=np.array([a, b, c]),
        Ft_vec=Ft, Fn_vec=Fn, Fo_vec=Fo,
        error=np.linalg.norm(F - (Ft + Fn + Fo)),
    )


def expected_force_ortho(coeffs: np.ndarray, basis: BasisOrtho) -> ForceDecompOrtho:
    """由系数反向构造力向量 F = a*t + b*n + c*o。"""
    a, b, c = coeffs
    t, n, o = basis.tangent, basis.normal, basis.ortho
    Ft, Fn, Fo = a * t, b * n, c * o
    return ForceDecompOrtho(
        coeffs=np.asarray(coeffs, dtype=float),
        Ft_vec=Ft, Fn_vec=Fn, Fo_vec=Fo,
        error=0.0,
    )
