"""V5 随机误差 + 摩擦+噪声 批量测试"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders, translate_cz
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT=0.005; MU=0.2; SIGMA=0.5; N=3000
rng0 = np.random.RandomState(42)
cy0, cz0 = load_standard_cylinders()

contact_std = sample_intersection(cy0, cz0, n_samples=500).sample_pts
with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    ball_ref = pickle.load(f)['ball_center_500']


def run(cy_err, cz_err, rng):
    ctrl = ForceController(cy0, cz0)
    pos = ctrl.ball_ref[0]
    traj, flog = [], []
    v_prev = np.zeros(3)
    for step in range(N):
        sc = step/(N-1)
        F_raw,_ = sphere_contact_force(pos, cz_err, cy_err)
        P_ct = ctrl._nearest_contact(pos)
        b = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
        Fn = np.dot(F_raw, b.normal)
        f_norm = np.linalg.norm(F_raw)
        if f_norm < 1e-6:
            F_fric = np.zeros(3)
        else:
            f_dir = F_raw / f_norm
            v_tang = v_prev - np.dot(v_prev, f_dir) * f_dir
            vn_t = np.linalg.norm(v_tang)
            F_fric = MU*abs(Fn)*(-v_tang/vn_t) if vn_t>1e-6 else np.zeros(3)
        F_meas = F_raw + F_fric + rng.randn(3)*SIGMA
        v_3d = ctrl.step(F_meas, sc, pos, N, DT)
        pos+=v_3d*DT; v_prev=v_3d.copy()
        traj.append(pos.copy()); flog.append(np.linalg.norm(F_meas))
    flog=np.array(flog)
    return np.array(traj), flog, np.mean(flog[-500:]), np.std(flog[-500:])


print(f"V5 随机误差 + 摩擦(μ={MU})+噪声(σ={SIGMA})")
print("="*60)
results = []
for seed in range(10):
    rng = np.random.RandomState(seed*100+42)  # 每组独立噪声
    np.random.seed(seed)
    dx,dy,dz = np.random.uniform(-0.5,0.5,3)
    czr = translate_cz(cz0, dx=dx, dy=dy, dz=dz)
    label = f'±0.5 (#{seed})'
    traj, flog, fm, fs = run(cy0, czr, rng)
    print(f"  {label} ({dx:+.2f},{dy:+.2f},{dz:+.2f}): |F|={fm:.2f}±{fs:.2f}N")
    results.append((label, dx,dy,dz, fm, fs, traj, flog))

# 汇总图
fig, axes = plt.subplots(3,4,figsize=(18,12))
axes = axes.flatten()
for i, (label, dx,dy,dz, fm, fs, traj, flog) in enumerate(results[:10]):
    ax = axes[i]
    ax.plot(flog, lw=0.5)
    ax.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax.set_title(f'{label}\n({dx:+.1f},{dy:+.1f},{dz:+.1f}) {fm:.2f}±{fs:.2f}')
    ax.set_ylim(0,16); ax.grid(alpha=0.2)

axes[10].axis('off'); axes[11].axis('off')
fig.suptitle(f'V5 随机误差 ±0.5mm (Kp=12/4, μ={MU}, σ={SIGMA})', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_rand10_fric_noise.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)

# 统计
fms = [r[4] for r in results]
fss = [r[5] for r in results]
print(f'\n汇总: |F|={np.mean(fms):.2f}±{np.mean(fss):.2f}N  (range {min(fms):.2f}~{max(fms):.2f})')
