"""
对比: 无扰动 / 只摩擦 / 只噪声 / 全开
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

DT = 0.005
MU = 0.2
SIGMA = 0.5
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
    flog = []
    v_prev = np.zeros(3)
    for step in range(N_STEPS):
        s_cur = step / (N_STEPS - 1)
        F_raw, _ = sphere_contact_force(pos, cz0, cy0)
        P_ct = ctrl._nearest_contact(pos)
        basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
        Fn = np.dot(F_raw, basis.normal)
        f_dir = F_raw / np.linalg.norm(F_raw)
        v_tang = v_prev - np.dot(v_prev, f_dir) * f_dir
        v_norm = np.linalg.norm(v_tang)
        F_fric = mu * abs(Fn) * (-v_tang / v_norm) if v_norm > 1e-6 else np.zeros(3)
        F_noise = rng.randn(3) * sigma
        F_vec = F_raw + F_fric + F_noise
        v_3d = ctrl.step(F_vec, s_cur, pos, N_STEPS, DT)
        pos = pos + v_3d * DT
        v_prev = v_3d.copy()
        flog.append(np.linalg.norm(F_vec))
    flog = np.array(flog)
    last500 = flog[-500:]
    print(f"  [{label}] |F|={np.mean(last500):.2f}+/-{np.std(last500):.2f}N")
    return flog


print("V5 扰动分离对比 (a=0.2)")
print("=" * 50)
rng = np.random.RandomState(42)
f_clean = run("无扰动")
rng = np.random.RandomState(42)
f_fric  = run("只摩擦 μ=0.2", mu=0.2)
rng = np.random.RandomState(42)
f_noise = run("只噪声 σ=0.5", sigma=0.5)
rng = np.random.RandomState(42)
f_both  = run("全开", mu=0.2, sigma=0.5)

# === 四合一图 ===
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
configs = [
    (f_clean, "无扰动", "blue"),
    (f_fric,  "只摩擦 (μ=0.2)", "orange"),
    (f_noise, "只噪声 (σ=0.5)", "green"),
    (f_both,  "全开", "red"),
]
for ax, (flog, title, color) in zip(axes.flat, configs):
    last500 = flog[-500:]
    ax.plot(flog, color, lw=0.5)
    ax.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax.set_title(f'{title}\n{np.mean(last500):.2f}±{np.std(last500):.2f}N')
    ax.set_ylabel('|F| (N)'); ax.set_xlabel('Step')
    ax.grid(alpha=0.3)

fig.suptitle('V5 扰动分离 (滤波 a=0.2)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_disturbance_compare.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)
