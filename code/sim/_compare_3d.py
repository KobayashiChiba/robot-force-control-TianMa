"""
3D轨迹对比: 无扰动 / 只摩擦 / 只噪声 / 全开
"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT = 0.005
N_STEPS = 3000
rng = np.random.RandomState(42)

cy0, cz0 = load_standard_cylinders()
geom0 = sample_intersection(cy0, cz0, n_samples=500)
pts0 = geom0.sample_pts
with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    ball_ref = pickle.load(f)['ball_center_500']


def run(label, mu=0.0, sigma=0.0):
    ctrl = ForceController(cy0, cz0)
    pos = ctrl.ball_ref[0]
    traj, flog = [], []
    v_prev = np.zeros(3)
    for step in range(N_STEPS):
        s_cur = step / (N_STEPS - 1)
        F_raw, _ = sphere_contact_force(pos, cz0, cy0)
        P_ct = ctrl._nearest_contact(pos)
        basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
        Fn = np.dot(F_raw, basis.normal)
        v_norm = np.linalg.norm(v_prev)
        F_fric = mu * abs(Fn) * (-v_prev / v_norm) if v_norm > 1e-6 else np.zeros(3)
        F_noise = rng.randn(3) * sigma
        F_vec = F_raw + F_fric + F_noise
        v_3d = ctrl.step(F_vec, s_cur, pos, N_STEPS, DT)
        pos = pos + v_3d * DT
        v_prev = v_3d.copy()
        traj.append(pos.copy())
        flog.append(np.linalg.norm(F_vec))
    return np.array(traj), np.array(flog)


print("V5 3D轨迹对比")
print("=" * 30)
rng = np.random.RandomState(42); t0, f0 = run("无扰动")
rng = np.random.RandomState(42); t1, f1 = run("只摩擦", mu=0.2)
rng = np.random.RandomState(42); t2, f2 = run("只噪声", sigma=0.5)
rng = np.random.RandomState(42); t3, f3 = run("全开", mu=0.2, sigma=0.5)
print("done")

fig = plt.figure(figsize=(18, 14))
ax3d = fig.add_subplot(221, projection='3d')
for ax2d, idx in [(fig.add_subplot(222), 0), (fig.add_subplot(223), 1), (fig.add_subplot(224), 2)]:
    pass

# === 3D ===
ax3d.plot(pts0[:, 0], pts0[:, 1], pts0[:, 2], 'gray', ls='--', lw=1, alpha=0.5)
ax3d.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2], 'green', lw=0.6, alpha=0.3)
ax3d.plot(t0[:,0], t0[:,1], t0[:,2], 'blue', lw=1.2, label=f'无扰动 ({np.mean(f0[-500:]):.1f}±{np.std(f0[-500:]):.1f}N)')
ax3d.plot(t1[:,0], t1[:,1], t1[:,2], 'orange', lw=1.0, label=f'只摩擦 ({np.mean(f1[-500:]):.1f}±{np.std(f1[-500:]):.1f}N)')
ax3d.plot(t2[:,0], t2[:,1], t2[:,2], 'green', lw=1.0, label=f'只噪声 ({np.mean(f2[-500:]):.1f}±{np.std(f2[-500:]):.1f}N)')
ax3d.plot(t3[:,0], t3[:,1], t3[:,2], 'red', lw=1.0, label=f'全开 ({np.mean(f3[-500:]):.1f}±{np.std(f3[-500:]):.1f}N)')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title('V5 3D轨迹: 扰动对比'); ax3d.legend(fontsize=7)

# === 力时序 ===
ax_f = fig.add_subplot(222)
for flog, color, label in [(f0,'blue','无扰动'),(f1,'orange','只摩擦'),(f2,'green','只噪声'),(f3,'red','全开')]:
    ax_f.plot(flog, color, lw=0.5, alpha=0.7, label=label)
ax_f.axhline(8.0, color='gray', ls='--', lw=0.5)
ax_f.set_ylabel('|F| (N)'); ax_f.set_xlabel('Step')
ax_f.legend(fontsize=7); ax_f.grid(alpha=0.3); ax_f.set_title('力全程')

# XY投影
ax_xy = fig.add_subplot(223)
ax_xy.plot(t0[:,0], t0[:,1], 'blue', lw=1.0, label='无扰动')
ax_xy.plot(t1[:,0], t1[:,1], 'orange', lw=0.8, label='只摩擦')
ax_xy.plot(t2[:,0], t2[:,1], 'green', lw=0.8, label='只噪声')
ax_xy.plot(t3[:,0], t3[:,1], 'red', lw=0.8, label='全开')
ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y'); ax_xy.set_title('XY投影')
ax_xy.legend(fontsize=7); ax_xy.grid(alpha=0.3)
ax_xy.set_aspect('equal')

# XZ投影
ax_xz = fig.add_subplot(224)
ax_xz.plot(t0[:,0], t0[:,2], 'blue', lw=1.0, label='无扰动')
ax_xz.plot(t1[:,0], t1[:,2], 'orange', lw=0.8, label='只摩擦')
ax_xz.plot(t2[:,0], t2[:,2], 'green', lw=0.8, label='只噪声')
ax_xz.plot(t3[:,0], t3[:,2], 'red', lw=0.8, label='全开')
ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z'); ax_xz.set_title('XZ投影')
ax_xz.legend(fontsize=7); ax_xz.grid(alpha=0.3)
ax_xz.set_aspect('equal')

fig.suptitle('V5 扰动分离 — 3D + 投影 (滤波 a=0.2)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_trajectory_compare.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
