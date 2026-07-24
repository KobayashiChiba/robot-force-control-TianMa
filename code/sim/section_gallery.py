"""
section_gallery.py — 0~0.25 每 0.05 取点，左右对比：3D视图 + 法平面截面

左图 (3D): 两圆柱面、接触曲线、法平面、截面线
右图 (2D): 法平面上的截面（横轴 n，纵轴 b=t×n）
        实线=金属面，虚线=重叠区域（空气）

用法: python section_gallery.py [--save]
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
PLANE_SPAN = 12.0
HALF_RANGE = 25.0
EPS = 1e-10


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


# ========== 截面采样 ==========

def _section_z_normal(phi, cyl_z, t, P0):
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]
    R = cyl_z.radius
    x = X0 + R * np.cos(phi)
    y = Y0 + R * np.sin(phi)
    h = (np.dot(t, P0) - t[0]*x - t[1]*y) / t[2]
    return np.array([x, y, h])


def _section_y_normal(theta, cyl_y, t, P0):
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]
    R = cyl_y.radius
    x = X0 + R * np.cos(theta)
    z = Z0 + R * np.sin(theta)
    h = (np.dot(t, P0) - t[0]*x - t[2]*z) / t[1]
    return np.array([x, h, z])


def _degenerate_section_z(cyl_z, t, P0, z_vals):
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]
    R = cyl_z.radius
    A, B = t[0]*R, t[1]*R
    C = np.dot(t, P0) - t[0]*X0 - t[1]*Y0
    mag = np.sqrt(A*A + B*B)
    if mag < EPS or abs(C) > mag + 1e-9:
        return None
    ratio = np.clip(C / mag, -1, 1)
    alpha = np.arctan2(A, B)
    phi1 = np.arcsin(ratio) - alpha
    phi2 = np.pi - np.arcsin(ratio) - alpha
    lines = []
    for phi in [phi1, phi2]:
        x = X0 + R * np.cos(phi)
        y = Y0 + R * np.sin(phi)
        pts = np.array([[x, y, z] for z in z_vals])
        lines.append(pts)
    return lines


def _degenerate_section_y(cyl_y, t, P0, y_vals):
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]
    R = cyl_y.radius
    A, B = t[0]*R, t[2]*R
    C = np.dot(t, P0) - t[0]*X0 - t[2]*Z0
    mag = np.sqrt(A*A + B*B)
    if mag < EPS or abs(C) > mag + 1e-9:
        return None
    ratio = np.clip(C / mag, -1, 1)
    alpha = np.arctan2(A, B)
    theta1 = np.arcsin(ratio) - alpha
    theta2 = np.pi - np.arcsin(ratio) - alpha
    lines = []
    for th in [theta1, theta2]:
        x = X0 + R * np.cos(th)
        z = Z0 + R * np.sin(th)
        pts = np.array([[x, y, z] for y in y_vals])
        lines.append(pts)
    return lines


def _inside_cyl_y(pts, cyl_y):
    """点是否在 Y柱内部（径向距离 < R_y）。Y柱轴∥Y。"""
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]
    R = cyl_y.radius
    dx = pts[:, 0] - X0
    dz = pts[:, 2] - Z0
    return np.sqrt(dx*dx + dz*dz) < R - 1e-6


def _inside_cyl_z(pts, cyl_z):
    """点是否在 Z柱内部（径向距离 < R_z）。Z柱轴∥Z。"""
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]
    R = cyl_z.radius
    dx = pts[:, 0] - X0
    dy = pts[:, 1] - Y0
    return np.sqrt(dx*dx + dy*dy) < R - 1e-6


def _split_air_mask(mask):
    """把布尔 mask 转为分段索引列表：[(start, end, is_air), ...]"""
    if mask is None or len(mask) == 0:
        return []
    segments = []
    n = len(mask)
    i = 0
    while i < n:
        val = mask[i]
        j = i
        while j < n and mask[j] == val:
            j += 1
        segments.append((i, j, val))
        i = j
    return segments


def sample_sections(P0, t, n, b, cyl_z, cyl_y, z_range, y_range):
    N = 2000
    phis = np.linspace(0, 2*np.pi, N)

    def to_uv(P): return np.dot(P - P0, n), np.dot(P - P0, b)

    result = {
        'z_uv': None, 'y_uv': None,
        'z_segs': [], 'y_segs': [],  # [(start, end, is_air)]
        'z_3d_lines': None, 'y_3d_lines': None,
    }

    # === Z柱 ===
    if abs(t[2]) > 1e-6:
        uv_list, g_list = [], []
        for phi in phis:
            Pz = _section_z_normal(phi, cyl_z, t, P0)
            if Pz is not None:
                uv_list.append(to_uv(Pz))
                g_list.append(Pz)
        if uv_list:
            result['z_uv'] = np.array(uv_list)
            g_arr = np.array(g_list)
            air = _inside_cyl_y(g_arr, cyl_y)
            result['z_segs'] = _split_air_mask(air)
    else:
        lines = _degenerate_section_z(cyl_z, t, P0, z_range)
        if lines is not None:
            result['z_3d_lines'] = lines
            # 退化线密集采样 + 空气判断
            N_dense = 100
            for i_line, line in enumerate(lines):
                ts = np.linspace(0, 1, N_dense)
                pts_3d = line[0] + ts[:, None] * (line[1] - line[0])
                uv_dense = np.array([to_uv(p) for p in pts_3d])
                air = _inside_cyl_y(pts_3d, cyl_y)
                segs = _split_air_mask(air)
                if i_line == 0:
                    result['z_uv'] = [uv_dense]
                    result['z_segs'] = [segs]
                else:
                    result['z_uv'].append(uv_dense)
                    result['z_segs'].append(segs)

    # === Y柱 ===
    if abs(t[1]) > 1e-6:
        uv_list, g_list = [], []
        for phi in phis:
            Py = _section_y_normal(phi, cyl_y, t, P0)
            if Py is not None:
                uv_list.append(to_uv(Py))
                g_list.append(Py)
        if uv_list:
            result['y_uv'] = np.array(uv_list)
            g_arr = np.array(g_list)
            air = _inside_cyl_z(g_arr, cyl_z)
            result['y_segs'] = _split_air_mask(air)
    else:
        lines = _degenerate_section_y(cyl_y, t, P0, y_range)
        if lines is not None:
            result['y_3d_lines'] = lines
            N_dense = 100
            for i_line, line in enumerate(lines):
                ts = np.linspace(0, 1, N_dense)
                pts_3d = line[0] + ts[:, None] * (line[1] - line[0])
                uv_dense = np.array([to_uv(p) for p in pts_3d])
                air = _inside_cyl_z(pts_3d, cyl_z)
                segs = _split_air_mask(air)
                if i_line == 0:
                    result['y_uv'] = [uv_dense]
                    result['y_segs'] = [segs]
                else:
                    result['y_uv'].append(uv_dense)
                    result['y_segs'].append(segs)

    return result


# ========== 绘图 ==========

def draw_3d(ax, P0, t, n, b, cyl_z, cyl_y, contact_curve, sec, center):
    z_min, z_max = center[2]-HALF_RANGE, center[2]+HALF_RANGE
    y_min, y_max = center[1]-HALF_RANGE, center[1]+HALF_RANGE

    phi = np.linspace(0, 2*np.pi, 72)
    zs = np.linspace(z_min, z_max, 20)
    Phi, Zz = np.meshgrid(phi, zs)
    ax.plot_surface(cyl_z.axis_point[0] + cyl_z.radius*np.cos(Phi),
                    cyl_z.axis_point[1] + cyl_z.radius*np.sin(Phi),
                    Zz, alpha=0.18, color='#4477AA', edgecolor='none')

    theta = np.linspace(0, 2*np.pi, 72)
    ys = np.linspace(y_min, y_max, 20)
    Theta, Yy = np.meshgrid(theta, ys)
    ax.plot_surface(cyl_y.axis_point[0] + cyl_y.radius*np.cos(Theta), Yy,
                    cyl_y.axis_point[2] + cyl_y.radius*np.sin(Theta),
                    alpha=0.18, color='#44AA44', edgecolor='none')

    uu = np.linspace(-PLANE_SPAN, PLANE_SPAN, 10)
    vv = np.linspace(-PLANE_SPAN, PLANE_SPAN, 10)
    UU, VV = np.meshgrid(uu, vv)
    pts_plane = P0 + UU[..., None]*n + VV[..., None]*b
    ax.plot_surface(pts_plane[..., 0], pts_plane[..., 1], pts_plane[..., 2],
                    alpha=0.15, color='orange', edgecolor='none')

    if sec['z_3d_lines'] is not None:
        for line in sec['z_3d_lines']:
            ax.plot(line[:, 0], line[:, 1], line[:, 2], 'b-', lw=2)
    if sec['y_3d_lines'] is not None:
        for line in sec['y_3d_lines']:
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color='lime', lw=2)

    ax.plot(contact_curve[:, 0], contact_curve[:, 1], contact_curve[:, 2],
            'k-', lw=0.8, alpha=0.6)
    ax.scatter(*P0, c='red', s=60, zorder=10)

    ax.set_xlim(center[0]-HALF_RANGE, center[0]+HALF_RANGE)
    ax.set_ylim(center[1]-HALF_RANGE, center[1]+HALF_RANGE)
    ax.set_zlim(center[2]-HALF_RANGE, center[2]+HALF_RANGE)
    ax.set_aspect('equal')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-45)


def draw_2d(ax, sec, cyl_z, cyl_y):
    ax.set_aspect('equal')

    def plot_segmented(uv_data, segs, color, label):
        """分段绘制：非空气段实线，空气段虚线。"""
        if uv_data is None:
            return
        if isinstance(uv_data, list):
            # 退化线 — 逐条画，segs 也是 list
            segs_list = segs if isinstance(segs, list) else []
            for i, line in enumerate(uv_data):
                line_segs = segs_list[i] if i < len(segs_list) else []
                if line_segs:
                    for start, end, is_air in line_segs:
                        ls = '--' if is_air else '-'
                        ax.plot(line[start:end, 0], line[start:end, 1],
                                color=color, lw=1.5, linestyle=ls,
                                label=label if i == 0 and start == 0 else None)
                        label = None
                else:
                    ax.plot(line[:, 0], line[:, 1], color=color, lw=1.5,
                            label=label if i == 0 else None)
            return
        if segs:
            for start, end, is_air in segs:
                ls = '--' if is_air else '-'
                ax.plot(uv_data[start:end, 0], uv_data[start:end, 1],
                        color=color, lw=1.5, linestyle=ls,
                        label=label if start == 0 else None)
                label = None
        else:
            ax.plot(uv_data[:, 0], uv_data[:, 1], color=color, lw=1.5, label=label)

    plot_segmented(sec['z_uv'], sec['z_segs'], 'b', f'Z柱 (R={cyl_z.radius:.0f})')
    plot_segmented(sec['y_uv'], sec['y_segs'], 'g', f'Y柱 (R={cyl_y.radius:.0f})')
    ax.plot(0, 0, 'ro', ms=5)
    ax.axhline(0, color='gray', lw=0.4, ls='--')
    ax.axvline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlabel('n (法向量)')
    ax.set_ylabel('b = t×n')
    ax.set_xlim(-20, 15); ax.set_ylim(-15, 15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc='upper right')


# ========== 主流程 ==========

def run(p_list, save_path=None):
    data = load_data()
    geom = data['contact_geom']
    cyl_y = data['cyl_contact_y']
    cyl_z = data['cyl_contact_z']
    contact_curve = geom.sample_pts

    cmin, cmax = contact_curve.min(axis=0), contact_curve.max(axis=0)
    center = (cmin + cmax) / 2

    z_range = np.array([center[2]-HALF_RANGE, center[2]+HALF_RANGE])
    y_range = np.array([center[1]-HALF_RANGE, center[1]+HALF_RANGE])

    N = len(p_list)
    fig = plt.figure(figsize=(16, 5.0*N))

    for i, p in enumerate(p_list):
        idx = int(round(p * (geom.n_samples - 1)))
        idx = max(0, min(idx, geom.n_samples - 1))
        P0 = geom.sample_pts[idx].copy()
        frame = compute_frame(P0, cyl_y, cyl_z)
        t, n = frame.tangent, frame.normal
        b = np.cross(t, n); b = b / np.linalg.norm(b)

        sec = sample_sections(P0, t, n, b, cyl_z, cyl_y, z_range, y_range)

        ax3 = fig.add_subplot(N, 2, 2*i+1, projection='3d')
        ax2 = fig.add_subplot(N, 2, 2*i+2)

        draw_3d(ax3, P0, t, n, b, cyl_z, cyl_y, contact_curve, sec, center)
        ax3.set_title(f'p={p:.2f}  — 3D', fontsize=11)

        draw_2d(ax2, sec, cyl_z, cyl_y)
        ax2.set_title(f'p={p:.2f}  — 法平面', fontsize=11)

    fig.tight_layout(pad=2.0)

    if save_path:
        import os; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'已保存: {save_path}')
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    p_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    save = '--save' in sys.argv
    run(p_vals, save_path='output/gallery_0_to_025.png' if save else None)
