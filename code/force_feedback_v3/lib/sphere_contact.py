"""
sphere_contact.py — 球刀接触力模型

Fibonacci 球面均匀采样 → 嵌入面积 → F = kc√S，方向指向球心。

用法:
    from sphere_contact import sphere_contact_force
    F_vec, area = sphere_contact_force(pos, cyl_z, cyl_y)

参数:
    K_C=7.37 (标定), R_BALL=4.2mm, N_SPHERE=12800
"""

import numpy as np

K_C = 6.5225
R_BALL = 4.2
N_SPHERE = 12800  # ≈ 80×160，Fibonacci 均匀分布


def _fibonacci_sphere(n):
    """Fibonacci 球面均匀采样，返回 (n,3) 单位球面上的点"""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


# 预计算单位球面采样点（全局只算一次）
_UNIT_SPHERE = _fibonacci_sphere(N_SPHERE)


def _inside_cyl(pts, cyl):
    """判断点是否在圆柱内部（沿轴线投影后计算径向距离）。

    对标准圆柱（轴线∥坐标轴）和倾斜的误差圆柱均正确。
    """
    axis = cyl.p2 - cyl.p1                              # (3,)
    L = np.linalg.norm(axis)
    d = axis / L                                        # 轴线方向单位向量

    v = pts - cyl.p1                                    # (N,3) 到 p1 的向量
    t = np.clip(np.dot(v, d), 0, L)                     # (N,) 沿轴向投影
    ax_pts = cyl.p1 + t[:, None] * d                    # (N,3) 轴线上投影点
    r = np.linalg.norm(pts - ax_pts, axis=1)            # (N,) 径向距离

    return r < cyl.radius - 1e-6


def sphere_contact_force(pos, cyl_z, cyl_y):
    """球刀在正交双圆柱间的接触力。

    Args:
        pos:   球心 3D 坐标 (3,)
        cyl_z: Z 方向圆柱 CylinderDef
        cyl_y: Y 方向圆柱 CylinderDef

    Returns:
        force_3d: 力向量 (3,) N，方向指向球心
        area:     接触面积 mm²
    """
    pts = pos + R_BALL * _UNIT_SPHERE  # 缩放+平移

    in_z = _inside_cyl(pts, cyl_z)
    in_y = _inside_cyl(pts, cyl_y)
    contact = ~in_z & ~in_y

    area_per_point = 4 * np.pi * R_BALL**2 / N_SPHERE
    area = contact.sum() * area_per_point

    if area <= 0:
        return np.zeros(3), 0.0

    c_pts = pts[contact]
    norms = pos - c_pts
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    force_dir = norms.mean(axis=0)
    force_dir /= np.linalg.norm(force_dir)

    return K_C * np.sqrt(area) * force_dir, area
