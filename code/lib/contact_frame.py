"""
contact_frame.py — 接触曲线局部标架计算

从接触曲线上一点 C 和两个正交圆柱的几何参数，
计算三个向量组成的局部标架（非正交）：

  切向量 (tangent)  — 沿接触曲线切线方向（打磨进给方向）
  法向量 (normal)   — 指向球刀中心方向（力控方向）
  Z径向 (radial_z)  — Z轴圆柱径向方向（XY平面内，用于力分解）

用法:
    from contact_frame import compute_frame

    frame = compute_frame(
        contact_pt=np.array([54.5, 65.0, -31.2]),
        cyl_y_axis_pt=np.array([51.5, 65.2, -39.7]),
        cyl_z_axis_pt=np.array([72.5, 65.0, -39.8]),
    )
    t, n, rz = frame.tangent, frame.normal, frame.radial_z
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ContactFrame:
    """接触曲线局部标架（非正交）"""
    tangent:  np.ndarray   # (3,) — 切向量，沿曲线切线方向（t = ry × rz）
    normal:   np.ndarray   # (3,) — 法向量，指向球刀中心（力控方向）
    radial_z: np.ndarray   # (3,) — Z轴圆柱径向，XY平面内指向轴心

    def as_matrix(self) -> np.ndarray:
        """返回 3×3 [t, n, rz]"""
        return np.column_stack([self.tangent, self.normal, self.radial_z])

    def __repr__(self):
        return (f'ContactFrame(\n'
                f'  t =({self.tangent[0]:+.4f}, {self.tangent[1]:+.4f}, {self.tangent[2]:+.4f}),\n'
                f'  n =({self.normal[0]:+.4f}, {self.normal[1]:+.4f}, {self.normal[2]:+.4f}),\n'
                f'  rz=({self.radial_z[0]:+.4f}, {self.radial_z[1]:+.4f}, {self.radial_z[2]:+.4f})\n'
                f')')


def compute_frame(
    contact_pt: np.ndarray,
    cyl_y_axis_pt: np.ndarray,
    cyl_z_axis_pt: np.ndarray,
    cyl_y_radius: float = None,
    cyl_z_radius: float = None,
) -> ContactFrame:
    """计算接触曲线上一点的局部标架。

    Parameters
    ----------
    contact_pt : (3,) ndarray
        接触曲线上的点坐标 C = [x, y, z]
    cyl_y_axis_pt : (3,) ndarray
        Y方向圆柱的轴心上一点（axis ∥ Y），形式 [x0, *, z0]
    cyl_z_axis_pt : (3,) ndarray
        Z方向圆柱的轴心上一点（axis ∥ Z），形式 [x0, y0, *]
    cyl_y_radius : float, optional
        Y方向圆柱半径。提供后用 r^(2/3) 加权计算法向量。
    cyl_z_radius : float, optional
        Z方向圆柱半径。

    Returns
    -------
    ContactFrame — 包含 tangent, normal, radial_z 三个单位向量
    """
    C = np.asarray(contact_pt, dtype=float)
    cy = np.asarray(cyl_y_axis_pt, dtype=float)
    cz = np.asarray(cyl_z_axis_pt, dtype=float)

    # ---- Y圆柱径向（在 XZ 平面内） ----
    ry = np.array([C[0] - cy[0], 0.0, C[2] - cy[2]])
    ry = ry / np.linalg.norm(ry)

    # ---- Z圆柱径向（在 XY 平面内） ----
    rz = np.array([C[0] - cz[0], C[1] - cz[1], 0.0])
    rz = rz / np.linalg.norm(rz)

    # ---- 切向量: t = ry × rz（精确正交于两个圆柱面法向量） ----
    t = np.cross(ry, rz)
    t = t / np.linalg.norm(t)

    # ---- 法向量: n = w_y·ry + w_z·rz（加权径向组合） ----
    if cyl_y_radius is not None and cyl_z_radius is not None:
        wy = cyl_y_radius ** (2/3)
        wz = cyl_z_radius ** (2/3)
    else:
        wy = 1.0
        wz = 1.0

    n = wy * ry + wz * rz
    n = n / np.linalg.norm(n)

    # ---- Z径向: 直接用 Z 轴圆柱径向向量 ----
    return ContactFrame(tangent=t, normal=n, radial_z=rz)


# ============================================================
# 批量计算
# ============================================================

def compute_frames_batch(
    contact_pts: np.ndarray,
    cyl_y_axis_pt: np.ndarray,
    cyl_z_axis_pt: np.ndarray,
    cyl_y_radius: float = None,
    cyl_z_radius: float = None,
) -> dict:
    """批量计算多个接触点的局部标架。

    Returns
    -------
    dict: 'tangents' (N,3), 'normals' (N,3), 'radial_z' (N,3)
    """
    N = len(contact_pts)
    tangents = np.zeros((N, 3))
    normals = np.zeros((N, 3))
    radial_z = np.zeros((N, 3))

    for i in range(N):
        f = compute_frame(
            contact_pts[i],
            cyl_y_axis_pt, cyl_z_axis_pt,
            cyl_y_radius, cyl_z_radius,
        )
        tangents[i] = f.tangent
        normals[i] = f.normal
        radial_z[i] = f.radial_z

    return {'tangents': tangents, 'normals': normals, 'radial_z': radial_z}
