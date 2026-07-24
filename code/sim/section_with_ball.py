"""
section_with_ball.py — 球刀截面图

输入球刀中心 + 半径 → 找最近接触点 → 法平面截面 + 球刀圆

用法: python section_with_ball.py <x> <y> <z> [--radius R] [--save]
  默认半径 = 4.0mm
"""

import sys
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from cylinder_def import CylinderDef
from contact_frame_v2 import compute_frame

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = '../data/standard_curves_v2.pkl'
PLANE_SPAN = 12.0
HALF_RANGE = 25.0


def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)


# ========== 截面采样（复用 gallery 逻辑） ==========

def _section_z_normal(phi, cyl_z, t, P0):
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1];
    R = cyl_z.radius
    x = X0 + R * np.cos(phi); y = Y0 + R * np.sin(phi)
    h = (np.dot(t, P0) - t[0]*x - t[1]*y) / t[2]
    return np.array([x, y, h])

def _section_y_normal(theta, cyl_y, t, P0):
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2];
    R = cyl_y.radius
    x = X0 + R * np.cos(theta); z = Z0 + R * np.sin(theta)
    h = (np.dot(t, P0) - t[0]*x - t[2]*z) / t[1]
    return np.array([x, h, z])

def _degenerate_section_z(cyl_z, t, P0, z_range):
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]; R = cyl_z.radius
    A, B = t[0]*R, t[1]*R; C = np.dot(t, P0) - t[0]*X0 - t[1]*Y0
    mag = np.sqrt(A*A + B*B)
    if mag < 1e-10 or abs(C) > mag + 1e-9: return None
    ratio = np.clip(C / mag, -1, 1); alpha = np.arctan2(A, B)
    phi1 = np.arcsin(ratio) - alpha; phi2 = np.pi - np.arcsin(ratio) - alpha
    lines = []
    for phi in [phi1, phi2]:
        x = X0 + R * np.cos(phi); y = Y0 + R * np.sin(phi)
        lines.append(np.array([[x, y, z] for z in z_range]))
    return lines

def _degenerate_section_y(cyl_y, t, P0, y_range):
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]; R = cyl_y.radius
    A, B = t[0]*R, t[2]*R; C = np.dot(t, P0) - t[0]*X0 - t[2]*Z0
    mag = np.sqrt(A*A + B*B)
    if mag < 1e-10 or abs(C) > mag + 1e-9: return None
    ratio = np.clip(C / mag, -1, 1); alpha = np.arctan2(A, B)
    theta1 = np.arcsin(ratio) - alpha; theta2 = np.pi - np.arcsin(ratio) - alpha
    lines = []
    for th in [theta1, theta2]:
        x = X0 + R * np.cos(th); z = Z0 + R * np.sin(th)
        lines.append(np.array([[x, y, z] for y in y_range]))
    return lines


def _inside_cyl_y(pts, cyl_y):
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]; R = cyl_y.radius
    return np.sqrt((pts[:,0]-X0)**2 + (pts[:,2]-Z0)**2) < R - 1e-6

def _inside_cyl_z(pts, cyl_z):
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]; R = cyl_z.radius
    return np.sqrt((pts[:,0]-X0)**2 + (pts[:,1]-Y0)**2) < R - 1e-6

def _split_air_mask(mask):
    if mask is None or len(mask)==0: return []
    segs, i, n = [], 0, len(mask)
    while i < n:
        v, j = mask[i], i
        while j < n and mask[j] == v: j += 1
        segs.append((i, j, v)); i = j
    return segs


def sample_sections(P0, t, n, b, cyl_z, cyl_y, z_range, y_range):
    N = 2000; phis = np.linspace(0, 2*np.pi, N)
    def to_uv(P): return np.dot(P-P0, n), np.dot(P-P0, b)
    result = {'z_uv':None,'y_uv':None,'z_segs':[],'y_segs':[],'z_3d':None,'y_3d':None}

    if abs(t[2]) > 1e-6:
        uv, g = [], []
        for phi in phis:
            Pz = _section_z_normal(phi, cyl_z, t, P0)
            if Pz is not None: uv.append(to_uv(Pz)); g.append(Pz)
        if uv:
            result['z_uv'] = np.array(uv)
            result['z_segs'] = _split_air_mask(_inside_cyl_y(np.array(g), cyl_y))
    else:
        lines = _degenerate_section_z(cyl_z, t, P0, z_range)
        if lines:
            result['z_3d'] = lines
            Nd = 100
            for ii, line in enumerate(lines):
                ts = np.linspace(0,1,Nd); pts = line[0]+ts[:,None]*(line[1]-line[0])
                uv_d = np.array([to_uv(p) for p in pts])
                segs = _split_air_mask(_inside_cyl_y(pts, cyl_y))
                if ii==0: result['z_uv']=[uv_d]; result['z_segs']=[segs]
                else: result['z_uv'].append(uv_d); result['z_segs'].append(segs)

    if abs(t[1]) > 1e-6:
        uv, g = [], []
        for phi in phis:
            Py = _section_y_normal(phi, cyl_y, t, P0)
            if Py is not None: uv.append(to_uv(Py)); g.append(Py)
        if uv:
            result['y_uv'] = np.array(uv)
            result['y_segs'] = _split_air_mask(_inside_cyl_z(np.array(g), cyl_z))
    else:
        lines = _degenerate_section_y(cyl_y, t, P0, y_range)
        if lines:
            result['y_3d'] = lines
            Nd = 100
            for ii, line in enumerate(lines):
                ts = np.linspace(0,1,Nd); pts = line[0]+ts[:,None]*(line[1]-line[0])
                uv_d = np.array([to_uv(p) for p in pts])
                segs = _split_air_mask(_inside_cyl_z(pts, cyl_z))
                if ii==0: result['y_uv']=[uv_d]; result['y_segs']=[segs]
                else: result['y_uv'].append(uv_d); result['y_segs'].append(segs)

    return result


# ========== 绘图 ==========

def plot_3d(ax, P0, t, n, b, cyl_z, cyl_y, contact_curve, sec, center,
            ball_center, R_ball):
    z_min, z_max = center[2]-HALF_RANGE, center[2]+HALF_RANGE
    y_min, y_max = center[1]-HALF_RANGE, center[1]+HALF_RANGE

    # 圆柱面
    phi_ = np.linspace(0, 2*np.pi, 72); zs_ = np.linspace(z_min,z_max,20)
    Phi, Zz = np.meshgrid(phi_, zs_)
    ax.plot_surface(cyl_z.axis_point[0]+cyl_z.radius*np.cos(Phi),
                    cyl_z.axis_point[1]+cyl_z.radius*np.sin(Phi),
                    Zz, alpha=0.18, color='#4477AA', edgecolor='none')
    theta_ = np.linspace(0, 2*np.pi, 72); ys_ = np.linspace(y_min,y_max,20)
    Theta, Yy = np.meshgrid(theta_, ys_)
    ax.plot_surface(cyl_y.axis_point[0]+cyl_y.radius*np.cos(Theta), Yy,
                    cyl_y.axis_point[2]+cyl_y.radius*np.sin(Theta),
                    alpha=0.18, color='#44AA44', edgecolor='none')

    # 球刀球（线框）
    u_sp = np.linspace(0, 2*np.pi, 24); v_sp = np.linspace(0, np.pi, 12)
    xs = ball_center[0] + R_ball * np.outer(np.cos(u_sp), np.sin(v_sp))
    ys = ball_center[1] + R_ball * np.outer(np.sin(u_sp), np.sin(v_sp))
    zs = ball_center[2] + R_ball * np.outer(np.ones_like(u_sp), np.cos(v_sp))
    ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.3, lw=0.3)

    # 法平面
    uu = np.linspace(-PLANE_SPAN, PLANE_SPAN, 10); vv = np.linspace(-PLANE_SPAN, PLANE_SPAN, 10)
    UU, VV = np.meshgrid(uu, vv)
    pts_plane = P0 + UU[...,None]*n + VV[...,None]*b
    ax.plot_surface(pts_plane[...,0], pts_plane[...,1], pts_plane[...,2],
                    alpha=0.15, color='orange', edgecolor='none')

    # 截面线(3D)
    if sec['z_3d']:
        for line in sec['z_3d']: ax.plot(line[:,0],line[:,1],line[:,2],'b-',lw=2)
    if sec['y_3d']:
        for line in sec['y_3d']: ax.plot(line[:,0],line[:,1],line[:,2],color='lime',lw=2)

    ax.plot(contact_curve[:,0], contact_curve[:,1], contact_curve[:,2],
            'k-', lw=0.8, alpha=0.6)
    ax.scatter(*P0, c='black', s=50, zorder=10, label='接触点')
    ax.scatter(*ball_center, c='red', s=60, zorder=10, label='球刀中心')

    ax.set_xlim(center[0]-HALF_RANGE, center[0]+HALF_RANGE)
    ax.set_ylim(center[1]-HALF_RANGE, center[1]+HALF_RANGE)
    ax.set_zlim(center[2]-HALF_RANGE, center[2]+HALF_RANGE)
    ax.set_aspect('equal')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=8)
    ax.view_init(elev=20, azim=-45)


def _ball_contact_segments(ball_uv, ball_r, P_contact, n_vec, b, cyl_z, cyl_y, N=360):
    """球刀圆上接触段分析。

    在法平面上采样球刀圆 → 映射回 3D → 判断在哪个柱面内部。
    柱面内部 = 空腔（空气），柱面外部 = 金属。

    - 不在 Z柱内 AND 不在 Y柱内 → 金属工件 → 接触
    - 在 Z柱内 AND 在 Y柱内 → 孔洞（空气）
    - 只在一柱内 → 空腔

    返回 [(a_start, a_end, 'contact'), ...]
    """
    alphas = np.linspace(0, 2*np.pi, N, endpoint=False)
    uv_x = ball_uv[0] + ball_r * np.cos(alphas)
    uv_y = ball_uv[1] + ball_r * np.sin(alphas)

    pts_3d = P_contact + uv_x[:,None]*n_vec + uv_y[:,None]*b

    in_z = _inside_cyl_z(pts_3d, cyl_z)
    in_y = _inside_cyl_y(pts_3d, cyl_y)

    # 不在任何柱面内部 = 在金属工件中
    in_metal = ~in_z & ~in_y

    segs = []
    i = 0
    while i < N:
        if not in_metal[i]:
            i += 1
            continue
        j = i
        while j < N and in_metal[j]:
            j += 1
        a1 = alphas[i]
        a2 = alphas[j-1] + (2*np.pi/N)
        segs.append((a1, a2, 'contact'))
        i = j

    # 合并首尾跨越 2π
    if len(segs) >= 2:
        fst, lst = segs[0], segs[-1]
        if abs(fst[0]) < 1e-6 and abs(lst[1] - 2*np.pi) < 1e-6:
            segs = [(lst[0] - 2*np.pi, fst[1], 'contact')] + segs[1:-1]

    return segs


def plot_2d(ax, sec, cyl_z, cyl_y, ball_center_uv, ball_radius_uv,
            P_contact, n_vec, b, N_ball=360):
    ax.set_aspect('equal')

    def plot_segmented(uv_data, segs, color, label):
        if uv_data is None: return
        if isinstance(uv_data, list):
            sl = segs if isinstance(segs,list) else []
            for i, line in enumerate(uv_data):
                ls = sl[i] if i < len(sl) else []
                if ls:
                    for s,e,a in ls:
                        ls_ = '--' if a else '-'
                        ax.plot(line[s:e,0], line[s:e,1], color=color, lw=1.5, linestyle=ls_,
                                label=label if i==0 and s==0 else None)
                        label = None
                else:
                    ax.plot(line[:,0],line[:,1],color=color,lw=1.5,label=label if i==0 else None)
            return
        if segs:
            for s,e,a in segs:
                ls_ = '--' if a else '-'
                ax.plot(uv_data[s:e,0], uv_data[s:e,1], color=color, lw=1.5, linestyle=ls_,
                        label=label if s==0 else None)
                label = None
        else:
            ax.plot(uv_data[:,0], uv_data[:,1], color=color, lw=1.5, label=label)

    plot_segmented(sec['z_uv'], sec['z_segs'], 'b', f'Z柱 (R={cyl_z.radius:.0f})')
    plot_segmented(sec['y_uv'], sec['y_segs'], 'g', f'Y柱 (R={cyl_y.radius:.0f})')

    # === 球刀圆 + 接触段高亮 ===
    if ball_radius_uv > 0.01:
        # 薄红圆（整体轮廓）
        ball_circle = Circle(ball_center_uv, ball_radius_uv, fill=False,
                             color='red', linewidth=1.5, alpha=0.5)
        ax.add_patch(ball_circle)
        ax.plot(*ball_center_uv, 'r+', ms=10, mew=2)

        # 接触段分析
        segs = _ball_contact_segments(
            ball_center_uv, ball_radius_uv,
            P_contact, n_vec, b, cyl_z, cyl_y, N=N_ball)

        for (a1, a2, _which) in segs:
            th = np.linspace(a1, a2, max(20, int(abs(a2-a1)/(2*np.pi)*N_ball)))
            x = ball_center_uv[0] + ball_radius_uv * np.cos(th)
            y = ball_center_uv[1] + ball_radius_uv * np.sin(th)
            ax.plot(x, y, color='#FF4500', lw=5, label='接触段' if len(segs)==1 or (_which==segs[0][2]) else None)

    # 接触点（原点）
    ax.plot(0, 0, 'ko', ms=6, label='接触点')

    ax.axhline(0, color='gray', lw=0.4, ls='--')
    ax.axvline(0, color='gray', lw=0.4, ls='--')
    ax.set_xlabel('n (法向量)'); ax.set_ylabel('b = t×n')
    ax.set_xlim(-20, 15); ax.set_ylim(-15, 15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc='upper right')


# ========== 主函数 ==========

def draw(ball_center, R_ball=4.0, save_path=None):
    """输入球刀中心坐标和半径，画 3D+2D 截面图。

    Parameters
    ----------
    ball_center : (3,) array-like
        球刀球心的 3D 坐标 (mm)。
    R_ball : float
        球刀半径 (mm)，默认 4.0。
    save_path : str or None
    """
    ball_center = np.asarray(ball_center, dtype=float)
    data = load_data()
    contact_curve = data['contact_geom'].sample_pts
    cyl_y = data['cyl_contact_y']
    cyl_z = data['cyl_contact_z']

    # 找最近接触点
    dists = np.linalg.norm(contact_curve - ball_center, axis=1)
    idx = np.argmin(dists)
    P_contact = contact_curve[idx].copy()
    d_min = dists[idx]

    # 法平面基底
    frame = compute_frame(P_contact, cyl_y, cyl_z)
    t = frame.tangent
    n_vec = frame.normal
    b = np.cross(t, n_vec); b = b / np.linalg.norm(b)

    # 球刀球 → 法平面投影
    d_plane = np.dot(ball_center - P_contact, t)   # 球心到法平面距离
    ball_proj = ball_center - d_plane * t            # 球心在法平面上的投影
    ball_uv = np.array([np.dot(ball_proj - P_contact, n_vec),
                        np.dot(ball_proj - P_contact, b)])
    r_circle = np.sqrt(max(0, R_ball**2 - d_plane**2))  # 法平面上截面圆半径

    # 截面
    cmin, cmax = contact_curve.min(axis=0), contact_curve.max(axis=0)
    center_3d = (cmin + cmax) / 2
    z_rng = np.array([center_3d[2]-HALF_RANGE, center_3d[2]+HALF_RANGE])
    y_rng = np.array([center_3d[1]-HALF_RANGE, center_3d[1]+HALF_RANGE])

    sec = sample_sections(P_contact, t, n_vec, b, cyl_z, cyl_y, z_rng, y_rng)

    # 画图
    fig = plt.figure(figsize=(16, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    plot_3d(ax3, P_contact, t, n_vec, b, cyl_z, cyl_y, contact_curve, sec,
            center_3d, ball_center, R_ball)
    ax3.set_title(f'3D — 接触点 #{idx}', fontsize=12)

    plot_2d(ax2, sec, cyl_z, cyl_y, ball_uv, r_circle,
            P_contact, n_vec, b)
    ax2.set_title(f'法平面截面 — 球刀圆 R={r_circle:.2f}mm', fontsize=12)

    fig.tight_layout()

    print(f'球刀中心: ({ball_center[0]:.2f}, {ball_center[1]:.2f}, {ball_center[2]:.2f})')
    print(f'最近接触点 #{idx}: ({P_contact[0]:.2f}, {P_contact[1]:.2f}, {P_contact[2]:.2f})')
    print(f'距离: {d_min:.3f} mm')
    print(f'法平面距离: {d_plane:.3f} mm')
    print(f'球刀圆半径: {r_circle:.3f} mm')
    print(f'球刀圆心 uv: ({ball_uv[0]:.2f}, {ball_uv[1]:.2f})')

    if save_path:
        import os; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'已保存: {save_path}')
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('用法: python section_with_ball.py <x> <y> <z> [--radius R] [--save]')
        sys.exit(1)

    x, y, z = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    R = 4.0
    if '--radius' in sys.argv:
        ri = sys.argv.index('--radius')
        R = float(sys.argv[ri + 1])

    save = '--save' in sys.argv
    sp = f'output/ball_{x:.1f}_{y:.1f}_{z:.1f}.png' if save else None
    draw([x, y, z], R, sp)
