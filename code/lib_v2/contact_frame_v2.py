"""
contact_frame_v2.py — 接触曲线局部标架计算 (v2)

用 CylinderDef 替代裸 axis_point + radius。
支持任意轴线方向，径向向量用真实 direction 投影。

对外接口:
    compute_frame(contact_pt, cyl_y, cyl_z) → ContactFrame
    compute_frames_batch(contact_pts, cyl_y, cyl_z) → dict
"""

from dataclasses import dataclass
import numpy as np
from cylinder_def import CylinderDef


@dataclass
class ContactFrame:
    """接触曲线局部标架（非正交）。

    Fields
    ------
    tangent  : ndarray (3,) — 切向量 t = r_y × r_z
    normal   : ndarray (3,) — 法向量 n = w_y·r_y + w_z·r_z (w ∝ r^(2/3))
    radial_z : ndarray (3,) — Z 圆柱径向（指向 Z 轴心）
    """
    tangent:  np.ndarray
    normal:   np.ndarray
    radial_z: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """返回 3×3 [t, n, rz]"""
        return np.column_stack([self.tangent, self.normal, self.radial_z])

    def __repr__(self):
        return (f'ContactFrame(\n'
                f'  t =({self.tangent[0]:+.4f}, {self.tangent[1]:+.4f}, {self.tangent[2]:+.4f}),\n'
                f'  n =({self.normal[0]:+.4f}, {self.normal[1]:+.4f}, {self.normal[2]:+.4f}),\n'
                f'  rz=({self.radial_z[0]:+.4f}, {self.radial_z[1]:+.4f}, {self.radial_z[2]:+.4f})\n'
                f')')


def _radial_vector(P, cyl):
    """从圆柱轴线到点 P 的径向单位向量。

    r = (P - axis_pt) 减去轴向分量，归一化。
    """
    v = np.asarray(P, dtype=float) - cyl.axis_point
    ax_proj = np.dot(v, cyl.direction) * cyl.direction
    r = v - ax_proj
    return r / np.linalg.norm(r)


def compute_frame(
    contact_pt: np.ndarray,
    cyl_y: CylinderDef,
    cyl_z: CylinderDef,
) -> ContactFrame:
    """计算接触曲线上一点的局部标架 {t, n, rz}。

    Parameters
    ----------
    contact_pt : (3,) ndarray
        接触曲线上的点坐标。
    cyl_y : CylinderDef
        Y 方向圆柱（轴线方向接近 Y 轴）。
    cyl_z : CylinderDef
        Z 方向圆柱（轴线方向接近 Z 轴）。

    Returns
    -------
    ContactFrame — {tangent, normal, radial_z}
    """
    # 径向向量
    r_y = _radial_vector(contact_pt, cyl_y)
    r_z = _radial_vector(contact_pt, cyl_z)

    # 切向量：t = r_y × r_z（精确正交于两个圆柱面法向）
    t = np.cross(r_y, r_z)
    t = t / np.linalg.norm(t)

    # 法向量：加权径向组合 n = w_y·r_y + w_z·r_z
    wy = cyl_y.radius ** (2/3)
    wz = cyl_z.radius ** (2/3)
    n = wy * r_y + wz * r_z
    n = n / np.linalg.norm(n)

    return ContactFrame(tangent=t, normal=n, radial_z=r_z)


def compute_frames_batch(
    contact_pts: np.ndarray,
    cyl_y: CylinderDef,
    cyl_z: CylinderDef,
) -> dict:
    """批量计算多个接触点的局部标架。

    Returns
    -------
    dict: {'tangents': (N,3), 'normals': (N,3), 'radial_z': (N,3)}
    """
    N = len(contact_pts)
    tangents = np.zeros((N, 3))
    normals = np.zeros((N, 3))
    radial_z = np.zeros((N, 3))

    for i in range(N):
        f = compute_frame(contact_pts[i], cyl_y, cyl_z)
        tangents[i] = f.tangent
        normals[i] = f.normal
        radial_z[i] = f.radial_z

    return {'tangents': tangents, 'normals': normals, 'radial_z': radial_z}
