"""
plot_inverse_field.py — (Fn,Fo) → (dn,db) 逆推热力图
与力场图对偶：横轴Fn/纵轴Fo，颜色=偏差量(dn或db)
叠加实际采样点验证拟合精度
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.force_field_quadratic import inverse
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts

# ── 逆推网格 ──
N = 41
Fn_grid = np.linspace(-30, 5, N)
Fo_grid = np.linspace(-10, 10, N)
DN = np.zeros((N, N)); DB = np.zeros((N, N))

for i, fn in enumerate(Fn_grid):
    for j, fo in enumerate(Fo_grid):
        dn, db = inverse(fn, fo)
        DN[j, i] = dn
        DB[j, i] = db

# ── 实际采样点（用于散点验证）──
R_SWEEP, N_SWEEP = 2.0, 21
dn_g = np.linspace(-R_SWEEP, R_SWEEP, N_SWEEP)
db_g = np.linspace(-R_SWEEP, R_SWEEP, N_SWEEP)
positions = np.linspace(0, 0.25, 6)

import force_feedback_v3.lib.sphere_contact as sc_mod
sc_mod.K_C = 7.37  # 当前K_C
scatter_fn, scatter_fo, scatter_dn, scatter_db = [], [], [], []
for p in positions:
    i = int(p * (len(ball_ref) - 1))
    P_b = ball_ref[i]
    idx = np.argmin(np.linalg.norm(ct - P_b, axis=1))
    basis = compute_point_basis_ortho(ct[idx], sim.contact_geom)
    n, o = basis.normal, basis.ortho
    for dni in dn_g:
        for dbj in db_g:
            pos = P_b + dni*n + dbj*o
            F, a = sphere_contact_force(pos, cz, cy)
            if a < 0.01: continue
            fn = np.dot(F, n); fo = np.dot(F, o)
            if fn < -30 or fn > 5 or fo < -10 or fo > 10: continue
            scatter_fn.append(fn); scatter_fo.append(fo)
            scatter_dn.append(dni); scatter_db.append(dbj)

# ── 画图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, data, title, scat_data in [
    (axes[0], DN, 'dn (mm)', scatter_dn),
    (axes[1], DB, 'db (mm)', scatter_db),
]:
    vmax = max(abs(data).max(), 1e-6)
    vmin = -vmax
    cs = ax.contourf(Fn_grid, Fo_grid, data, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cs, ax=ax)
    # 采样点散点（颜色=实际偏差量）
    sp = ax.scatter(scatter_fn, scatter_fo, c=scat_data, s=3,
                    cmap='RdBu_r', vmin=vmin, vmax=vmax, edgecolors='none', alpha=0.6)
    # 锚定点
    ax.plot(-8, 0, 'ko', ms=8, mew=2)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
    ax.set_title(title)
    ax.set_aspect('auto')

fig.suptitle('Inverse field: (Fn,Fo) → (dn,db)  —  dots=samples  ●=anchor(-8,0)→(0,0)', fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'inverse_field.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
