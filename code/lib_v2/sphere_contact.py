"""
sphere_contact.py — 球刀接触力模型

球面采样嵌入面积 → F = kc√S，方向指向球心。
这是项目中唯一的力模型实现，其他模块统一 import 此处。

用法:
    from sphere_contact import sphere_contact_force
    F_vec, area = sphere_contact_force(pos, cyl_z, cyl_y)

参数:
    K_C=7.37 (标定), R_BALL=4.2mm, 球面精度 80×160
"""

import numpy as np

K_C = 7.37
R_BALL = 4.2
N_SPHERE_TH = 80
N_SPHERE_PH = 160


def _inside_cyl_z(pts, cyl_z):
    """球面采样点是否在 Z 圆柱内部"""
    X0, Y0 = cyl_z.p1[0], cyl_z.p1[1]
    return np.sqrt((pts[:, 0] - X0)**2 + (pts[:, 1] - Y0)**2) < cyl_z.radius - 1e-6


def _inside_cyl_y(pts, cyl_y):
    """球面采样点是否在 Y 圆柱内部"""
    X0, Z0 = cyl_y.p1[0], cyl_y.p1[2]
    return np.sqrt((pts[:, 0] - X0)**2 + (pts[:, 2] - Z0)**2) < cyl_y.radius - 1e-6


def sphere_contact_force(pos, cyl_z, cyl_y):
    """球刀在 (cyl_z, cyl_y) 正交双圆柱间的接触力。

    Args:
        pos:    球心 3D 坐标 (3,)
        cyl_z:  Z 方向圆柱 CylinderDef
        cyl_y:  Y 方向圆柱 CylinderDef

    Returns:
        force_3d: 力向量 (3,) N，方向指向球心
        area:     接触面积 mm²
    """
    th = np.linspace(0, np.pi, N_SPHERE_TH)
    ph = np.linspace(0, 2 * np.pi, N_SPHERE_PH, endpoint=False)
    Th, Ph = np.meshgrid(th, ph)

    xs = pos[0] + R_BALL * np.sin(Th) * np.cos(Ph)
    ys = pos[1] + R_BALL * np.sin(Th) * np.sin(Ph)
    zs = pos[2] + R_BALL * np.cos(Th)
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    in_z = _inside_cyl_z(pts, cyl_z)
    in_y = _inside_cyl_y(pts, cyl_y)
    contact = ~in_z & ~in_y
    area = contact.sum() * (4 * np.pi * R_BALL**2) / (N_SPHERE_TH * N_SPHERE_PH)

    if not contact.any():
        return np.zeros(3), 0.0

    c_pts = pts[contact]
    norms = pos - c_pts
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    force_dir = norms.mean(axis=0)
    force_dir /= np.linalg.norm(force_dir)

    return K_C * np.sqrt(max(0, area)) * force_dir, area
