"""
draw_section.py — 法平面截面图

在接触曲线上取点 → 法平面（⊥ 切向量 t）→ 画与 Z柱、Y柱的交线。

横轴 = 法向量 n（加权 r^(2/3)），纵轴 = 副法向量 b = t × n。
原点 = 接触点 P_contact。

用法: python draw_section.py <比例> [--save]
"""

import sys
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from cylinder_def import CylinderDef
from contact_frame_v2 import compute_frame

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = '../data/standard_curves_v2.pkl'


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


def _section_z(phi, cyl_z, t, P0):
    """Z柱面（轴∥Z）上参数点，约束在法平面内。返回全局坐标或 None。"""
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]
    R = cyl_z.radius
    if abs(t[2]) < 1e-12:
        return None
    x = X0 + R * np.cos(phi)
    y = Y0 + R * np.sin(phi)
    h = (np.dot(t, P0) - t[0] * x - t[1] * y) / t[2]
    return np.array([x, y, h])


def _section_y(theta, cyl_y, t, P0):
    """Y柱面（轴∥Y）上参数点，约束在法平面内。返回全局坐标或 None。"""
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]
    R = cyl_y.radius
    if abs(t[1]) < 1e-12:
        return None
    x = X0 + R * np.cos(theta)
    z = Z0 + R * np.sin(theta)
    h = (np.dot(t, P0) - t[0] * x - t[2] * z) / t[1]
    return np.array([x, h, z])


def draw_section(p, save_path=None):
    data = load_data()
    geom = data['contact_geom']
    cyl_y = data['cyl_contact_y']
    cyl_z = data['cyl_contact_z']

    idx = int(round(p * (geom.n_samples - 1)))
    P0 = geom.sample_pts[idx].copy()

    # 切向量 + 法平面基底
    frame = compute_frame(P0, cyl_y, cyl_z)
    t = frame.tangent
    n = frame.normal          # 法向量（加权 r^(2/3)）→ 横轴
    b = np.cross(t, n)        # 副法向量 → 纵轴
    b = b / np.linalg.norm(b)

    # --- 截面采样 ---
    N = 2000
    phis = np.linspace(0, 2 * np.pi, N)

    def to_uv(P): return np.dot(P - P0, n), np.dot(P - P0, b)

    z_uv, y_uv = [], []
    for phi in phis:
        Pz = _section_z(phi, cyl_z, t, P0)
        if Pz is not None:
            z_uv.append(to_uv(Pz))
        Py = _section_y(phi, cyl_y, t, P0)
        if Py is not None:
            y_uv.append(to_uv(Py))

    z_uv = np.array(z_uv) if z_uv else np.empty((0, 2))
    y_uv = np.array(y_uv) if y_uv else np.empty((0, 2))

    # --- 画图 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    if len(z_uv) > 0:
        ax.plot(z_uv[:, 0], z_uv[:, 1], 'b-', linewidth=1.5, label=f'Z柱 (R={cyl_z.radius:.0f}mm)')
    if len(y_uv) > 0:
        ax.plot(y_uv[:, 0], y_uv[:, 1], 'g-', linewidth=1.5, label=f'Y柱 (R={cyl_y.radius:.0f}mm)')

    ax.plot(0, 0, 'ko', markersize=6, label='接触点 (原点)')

    # 标注
    ax.set_xlabel('n (法向量) [mm]')
    ax.set_ylabel('b = t×n (副法向量) [mm]')
    ax.set_title(f'法平面截面 — p={p:.3f}  (点 #{idx}/{geom.n_samples})')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-25, 20)
    ax.set_ylim(-20, 20)

    fig.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f'已保存: {save_path}')
    else:
        plt.show()
    plt.close(fig)

    print(f'\np={p:.3f}  |  t=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  |  Z柱截面: {len(z_uv)}点  Y柱截面: {len(y_uv)}点')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python draw_section.py <比例> [--save]')
        sys.exit(1)
    p = float(sys.argv[1])
    save_path = f'output/section_p{p:.3f}.png' if '--save' in sys.argv else None
    draw_section(p, save_path)
