"""
force_profile.py — 球刀沿中心曲线跑一圈，计算接触力变化

用法: python force_profile.py [--save]
"""

import sys
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from contact_frame_v2 import compute_frame

plt.rcParams['font.family'] = 'Microsoft YaHei'

plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = '../data/force_model.pkl'
K_C = 6.80  # N/mm² (全接触面积, 标定均值=8N)
R_BALL = 4.2
N_SPHERE_TH = 80
N_SPHERE_PH = 160

def load_data():
    with open(DATA_PATH, 'rb') as f:
        return pickle.load(f)

def _inside_cyl_z(pts, cyl_z):
    X0, Y0 = cyl_z.axis_point[0], cyl_z.axis_point[1]; R = cyl_z.radius
    return np.sqrt((pts[:,0]-X0)**2 + (pts[:,1]-Y0)**2) < R - 1e-6

def _inside_cyl_y(pts, cyl_y):
    X0, Z0 = cyl_y.axis_point[0], cyl_y.axis_point[2]; R = cyl_y.radius
    return np.sqrt((pts[:,0]-X0)**2 + (pts[:,2]-Z0)**2) < R - 1e-6

def sphere_contact_force(ball_center, v_dir, cyl_z, cyl_y):
    """球面采样 → 全部接触面积 → 力。

    Returns: force_3d, area_total
    """
    th = np.linspace(0, np.pi, N_SPHERE_TH)
    ph = np.linspace(0, 2*np.pi, N_SPHERE_PH, endpoint=False)
    Th, Ph = np.meshgrid(th, ph)
    xs = ball_center[0] + R_BALL * np.sin(Th) * np.cos(Ph)
    ys = ball_center[1] + R_BALL * np.sin(Th) * np.sin(Ph)
    zs = ball_center[2] + R_BALL * np.cos(Th)
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    in_z = _inside_cyl_z(pts, cyl_z)
    in_y = _inside_cyl_y(pts, cyl_y)
    contact = ~in_z & ~in_y
    area_total = contact.sum() * (4*np.pi*R_BALL**2) / (N_SPHERE_TH*N_SPHERE_PH)

    if not contact.any():
        return np.zeros(3), area_total

    # 接触力方向 = 所有接触点平均法向（指向球心）
    c_pts = pts[contact]
    normals = ball_center - c_pts
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    force_dir = normals.mean(axis=0)
    force_dir /= np.linalg.norm(force_dir)

    force_mag = K_C * area_total
    return force_mag * force_dir, area_total

def run(save_path=None):
    data = load_data()
    ball_centers = data['ball_center_500']        # (500, 3) 傅里叶拟合
    contact_geom = data['contact_geom']           # 2000点
    cyl_y, cyl_z = data['cyl_contact_y'], data['cyl_contact_z']

    N = len(ball_centers)
    forces = np.zeros((N, 3))
    areas = np.zeros(N)
    Ft, Fn, Fo = np.zeros(N), np.zeros(N), np.zeros(N)
    Ft_f, Fn_f, Fo_f = np.zeros(N), np.zeros(N), np.zeros(N)  # 摩擦
    Ft_tot, Fn_tot, Fo_tot = np.zeros(N), np.zeros(N), np.zeros(N)  # 总力(含噪声)
    progress = np.linspace(0, 1, N)
    MU = 0.2
    NOISE_SIGMA = 0.5  # N
    np.random.seed(42)

    for i in range(N):
        ball_center = ball_centers[i]
        dists = np.linalg.norm(contact_geom.sample_pts - ball_center, axis=1)
        idx = np.argmin(dists)
        P_contact = contact_geom.sample_pts[idx]

        v_diff = ball_centers[1]-ball_centers[0] if i==0 else ball_centers[i]-ball_centers[i-1]
        v_dir = v_diff / np.linalg.norm(v_diff)

        f_vec, at = sphere_contact_force(ball_center, v_dir, cyl_z, cyl_y)
        forces[i] = f_vec
        areas[i] = at

        # 正交分解
        frame = compute_frame(P_contact, cyl_y, cyl_z)
        t = frame.tangent; nb = frame.normal; ob = np.cross(t, nb); ob /= np.linalg.norm(ob)
        Ft[i] = np.dot(f_vec, t); Fn[i] = np.dot(f_vec, nb); Fo[i] = np.dot(f_vec, ob)

        # 摩擦力
        f_fric = -MU * abs(Fn[i]) * v_dir
        Ft_f[i] = np.dot(f_fric, t); Fn_f[i] = np.dot(f_fric, nb); Fo_f[i] = np.dot(f_fric, ob)

        # 噪声
        noise = np.random.randn(3) * NOISE_SIGMA
        f_total = f_vec + f_fric + noise
        Ft_tot[i] = np.dot(f_total, t); Fn_tot[i] = np.dot(f_total, nb); Fo_tot[i] = np.dot(f_total, ob)

        if i % 50 == 0:
            print(f'  {i}/{N}  |F|={np.linalg.norm(f_vec):.2f}N  Ft_f={Ft_f[i]:+.2f}')

    # --- 画图 ---
    F_mag = np.linalg.norm(forces, axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    ax = axes[0]
    ax.plot(progress, F_mag, 'b-', lw=1)
    ax.plot(progress, np.sqrt(Ft_tot**2+Fn_tot**2+Fo_tot**2), 'r-', lw=0.6, alpha=0.6, label='+摩擦+噪声')
    ax.set_ylabel('|F| (N)'); ax.axhline(8, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(progress, Ft, ':', color='gray', lw=0.8, label='Ft接触')
    ax.plot(progress, Fn, ':', color='gray', lw=0.8, label='Fn接触')
    ax.plot(progress, Fo, ':', color='gray', lw=0.8)
    ax.plot(progress, Ft_tot, 'b', lw=1, label='Ft 总')
    ax.plot(progress, Fn_tot, 'g', lw=1, label='Fn 总')
    ax.plot(progress, Fo_tot, 'orange', lw=1, label='Fo 总')
    ax.set_ylabel('正交分解 (N)'); ax.axhline(0, color='gray', lw=0.3)
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(progress, Ft_f, 'b', lw=1, label='Ft 摩擦')
    ax.plot(progress, Fn_f, 'g', lw=1, label='Fn 摩擦')
    ax.plot(progress, Fo_f, 'orange', lw=1, label='Fo 摩擦')
    ax.set_ylabel('摩擦力 (N)'); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(progress, areas*1000, 'orange', lw=1.2)
    ax.set_xlabel('曲线进度'); ax.set_ylabel('接触面积 (x0.001 mm²)'); ax.grid(alpha=0.3)

    fig.suptitle('球刀沿标准中心曲线 — 接触力变化', fontsize=13)
    fig.tight_layout()

    if save_path:
        import os; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'已保存: {save_path}')
    else:
        plt.show()
    plt.close(fig)

    print(f'\n接触力: 均值 {F_mag.mean():.2f}N  std {F_mag.std():.2f}N  min {F_mag.min():.2f}N  max {F_mag.max():.2f}N')
    print(f'接触面积: 均值 {areas.mean()*1000:.2f}×10⁻³ mm²')

if __name__ == '__main__':
    save = '--save' in sys.argv
    run(save_path='output/force_profile.png' if save else None)
