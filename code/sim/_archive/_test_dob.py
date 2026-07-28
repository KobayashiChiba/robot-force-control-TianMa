"""
DOB 对比: 无DOB vs 有DOB（均含摩擦+噪声）
"""
import sys, os, pickle, numpy as np, time
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

DT = 0.005; MU = 0.2; SIGMA = 0.5; N_STEPS = 3000
cy0, cz0 = load_standard_cylinders()
rng = np.random.RandomState(42)


class DOB:
    """扰动观测器: 低通估计力偏差 → 前馈补偿"""
    def __init__(self, alpha=0.1):
        self.a = alpha
        self.dist_n, self.dist_o = 0.0, 0.0
        self.ok = False

    def compensate(self, F_meas, n, o, F_target=-8.0):
        Fn = np.dot(F_meas, n)
        Fo = np.dot(F_meas, o)
        err_n = Fn - F_target
        err_o = Fo
        if not self.ok:
            self.dist_n, self.dist_o = err_n, err_o
            self.ok = True
        else:
            self.dist_n = self.a * err_n + (1 - self.a) * self.dist_n
            self.dist_o = self.a * err_o + (1 - self.a) * self.dist_o
        return F_meas - self.dist_n * n - self.dist_o * o


def run(label, use_dob=False, mu=0.0, sigma=0.0):
    ctrl = ForceController(cy0, cz0)
    dob = DOB(alpha=0.05) if use_dob else None
    pos = ctrl.ball_ref[0]
    traj, flog, flog_dist = [], [], []
    v_prev = np.zeros(3)
    for step in range(N_STEPS):
        s_cur = step / (N_STEPS - 1)
        F_raw, _ = sphere_contact_force(pos, cz0, cy0)

        P_ct = ctrl._nearest_contact(pos)
        basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
        n, o = basis.normal, basis.ortho
        Fn = np.dot(F_raw, n)

        v_norm = np.linalg.norm(v_prev)
        F_fric = mu * abs(Fn) * (-v_prev / v_norm) if v_norm > 1e-6 else np.zeros(3)
        F_noise = rng.randn(3) * sigma
        F_meas = F_raw + F_fric + F_noise

        if dob:
            F_comp = dob.compensate(F_meas, n, o)
            flog_dist.append(np.linalg.norm(F_meas - F_comp))
        else:
            F_comp = F_meas

        v_3d = ctrl.step(F_comp, s_cur, pos, N_STEPS, DT)
        pos = pos + v_3d * DT
        v_prev = v_3d.copy()
        traj.append(pos.copy())
        flog.append(np.linalg.norm(F_meas))  # 显示真实力

    return np.array(traj), np.array(flog), np.array(flog_dist) if flog_dist else None


print("DOB 对比 (μ=0.2, σ=0.5, a=0.1)")
print("=" * 50)
rng = np.random.RandomState(42); t_none, f_none, _ = run("无DOB", mu=0.2, sigma=0.5)
rng = np.random.RandomState(42); t_dob,  f_dob,  d_dob = run("DOB", use_dob=True, mu=0.2, sigma=0.5)

print(f"无DOB: |F|={np.mean(f_none[-500:]):.2f}+/-{np.std(f_none[-500:]):.2f}N")
print(f"有DOB: |F|={np.mean(f_dob[-500:]):.2f}+/-{np.std(f_dob[-500:]):.2f}N")

# 图
geom0 = sample_intersection(cy0, cz0, n_samples=500)
pts0 = geom0.sample_pts
with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    ball_ref = pickle.load(f)['ball_center_500']

fig = plt.figure(figsize=(16, 12))
ax3d = fig.add_subplot(221, projection='3d')

ax3d.plot(pts0[:, 0], pts0[:, 1], pts0[:, 2], 'gray', ls='--', lw=1, alpha=0.5)
ax3d.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2], 'green', lw=0.6, alpha=0.3)
ax3d.plot(t_none[:,0], t_none[:,1], t_none[:,2], 'orange', lw=1.0, label=f'无DOB ({np.mean(f_none[-500:]):.1f}±{np.std(f_none[-500:]):.1f}N)')
ax3d.plot(t_dob[:,0], t_dob[:,1], t_dob[:,2], 'blue', lw=1.2, label=f'DOB ({np.mean(f_dob[-500:]):.1f}±{np.std(f_dob[-500:]):.1f}N)')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.legend(fontsize=8); ax3d.set_title('DOB 3D轨迹对比')

ax_f = fig.add_subplot(222)
ax_f.plot(f_none, 'orange', lw=0.5, alpha=0.7, label=f'无DOB ({np.mean(f_none[-500:]):.1f}±{np.std(f_none[-500:]):.1f})')
ax_f.plot(f_dob, 'blue', lw=0.8, label=f'DOB ({np.mean(f_dob[-500:]):.1f}±{np.std(f_dob[-500:]):.1f})')
ax_f.axhline(8.0, color='gray', ls='--', lw=0.5)
ax_f.set_ylabel('|F| (N)'); ax_f.set_xlabel('Step')
ax_f.legend(fontsize=7); ax_f.grid(alpha=0.3); ax_f.set_title('力全程')

ax_d = fig.add_subplot(223)
ax_d.plot(d_dob, 'purple', lw=0.5)
ax_d.set_ylabel('|F_disturb| (N)'); ax_d.set_xlabel('Step')
ax_d.set_title(f'DOB 扰动估计 ({np.mean(d_dob):.2f}±{np.std(d_dob):.2f}N)')
ax_d.grid(alpha=0.3)

ax_last = fig.add_subplot(224)
for flog, color, label in [(f_none,'orange','无DOB'),(f_dob,'blue','DOB')]:
    ax_last.plot(range(len(flog)-200, len(flog)), flog[-200:], color, lw=0.8, label=label)
ax_last.axhline(8.0, color='gray', ls='--', lw=0.5)
ax_last.set_ylabel('|F| (N)'); ax_last.set_xlabel('Step')
ax_last.legend(fontsize=7); ax_last.grid(alpha=0.3); ax_last.set_title('稳态: 最后200步')

fig.suptitle('V5 DOB对比 (μ=0.2, σ=0.5, DOB α=0.05)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_dob_compare.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
