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
K_C = 13.4  # N/mm² (标定: R=4.2mm, 傅里叶球刀曲线, 均值=8N)
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
    """球面采样 → 前方接触面积 → 力。

    Returns: force_3d, area_front, area_total
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
        return np.zeros(3), 0.0, area_total

    # 前方接触
    dot_v = np.dot(pts - ball_center, v_dir)
    front = contact & (dot_v > 0)
    area_front = front.sum() * (4*np.pi*R_BALL**2) / (N_SPHERE_TH*N_SPHERE_PH)

    if not front.any():
        return np.zeros(3), area_front, area_total

    # 接触力方向 = 前方接触点平均法向（指向球心）
    c_pts = pts[front]
    normals = ball_center - c_pts
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    force_dir = normals.mean(axis=0)
    force_dir /= np.linalg.norm(force_dir)

    force_mag = K_C * area_front
    return force_mag * force_dir, area_front, area_total


def run(save_path=None):
    data = load_data()
    ball_centers = data['ball_center_500']        # (500, 3) 傅里叶拟合
    contact_geom = data['contact_geom']           # 2000点
    cyl_y, cyl_z = data['cyl_contact_y'], data['cyl_contact_z']

    N = len(ball_centers)
    forces = np.zeros((N, 3))
    areas_front = np.zeros(N)
    areas_total = np.zeros(N)
    progress = np.linspace(0, 1, N)

    for i in range(N):
        ball_center = ball_centers[i]

        # 最近接触点
        dists = np.linalg.norm(contact_geom.sample_pts - ball_center, axis=1)
        idx = np.argmin(dists)

        # 运动方向 = 前后点差分
        if i > 0:
            v_diff = ball_centers[i] - ball_centers[i-1]
        else:
            v_diff = ball_centers[1] - ball_centers[0]
        v_dir = v_diff / np.linalg.norm(v_diff)

        f_vec, af, at = sphere_contact_force(ball_center, v_dir, cyl_z, cyl_y)
        forces[i] = f_vec
        areas_front[i] = af
        areas_total[i] = at

        if i % 10 == 0:
            print(f'  {i}/{N}  p={progress[i]:.3f}  |F|={np.linalg.norm(f_vec):.2f}N')

    # --- 画图 ---
    F_mag = np.linalg.norm(forces, axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(progress, F_mag, 'b-', lw=1.2)
    ax.set_ylabel('接触力 |F| (N)')
    ax.axhline(8, color='gray', ls='--', lw=0.8, label='标定基准 8N')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(progress, areas_front*1000, 'orange', lw=1.2, label='前方接触')
    ax.plot(progress, areas_total*1000, 'gray', lw=0.8, alpha=0.5, label='全部接触')
    ax.set_ylabel('接触面积 (x0.001 mm²)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(progress, forces[:,0], label='F_x', lw=1)
    ax.plot(progress, forces[:,1], label='F_y', lw=1)
    ax.plot(progress, forces[:,2], label='F_z', lw=1)
    ax.set_xlabel('曲线进度')
    ax.set_ylabel('力分量 (N)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

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
    print(f'前方面积: 均值 {areas_front.mean()*1000:.4f}×10⁻³ mm²')


if __name__ == '__main__':
    save = '--save' in sys.argv
    run(save_path='output/force_profile.png' if save else None)
