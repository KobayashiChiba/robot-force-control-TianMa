"""
plot_inverse_models_compare.py — 三种逆推模型的力场对比

模型:
  1. 线性固定 (force_field_fixed):  dn=(F_target-Fn)/KN, db=-Fo/KO
  2. 物理分块 (force_field_physical): Fn↔dn分段+dn·db耦合, db不对称
  3. 二次锚定 (force_field_quadratic): 6系数锚定二次曲面

每个模型画 (Fn,Fo)→dn 和 (Fn,Fo)→db 两张热力图
3模型×2方向 = 6个面板
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.simulator import Simulator
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.sphere_contact import sphere_contact_force

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 三种模型 ──
# 1. 线性固定
F_TARGET = -8.0; KN = -24.533; KO = 1.4
def inv_fixed(fn, fo):
    if fn >= 0: return 0.5, 0.0
    return (F_TARGET - fn) / KN, -fo / KO

# 2. 物理分块
K_FN = 11.70; C_FN = 6.48; K_RU = 5.65; K_RD = 1.60; FN_TH = 2.0
def inv_physical(fn, fo):
    fa = abs(fn)
    if fa < 0.5: return 0.0, 0.0
    dn = (fa - C_FN) / K_FN
    if dn <= 0: return 0.0, 0.0
    if fa < FN_TH: return dn, 0.0
    if fo > 0:  db = fo / (K_RU * dn)
    elif fo < 0: db = fo / (K_RD * dn)
    else: db = 0.0
    return dn, db

# 3. 二次锚定
_COEF_DN = np.array([-0.496811, -0.055361,  0.081592,  0.000843,  0.003985,  0.003034])
_COEF_DB = np.array([ 0.540874,  0.075355,  0.447720,  0.000968,  0.010277, -0.003157])
def inv_quadratic(fn, fo):
    if fn >= 0: fn_c = 0.0
    else: fn_c = fn
    fn_c = max(-30.0, min(0.0, fn_c))
    fo_c = max(-10.0, min(10.0, fo))
    x = np.array([1.0, fn_c, fo_c, fn_c**2, fn_c*fo_c, fo_c**2])
    return float(x @ _COEF_DN), float(x @ _COEF_DB)

# ── 真实力场采样点 ──
cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts

R_SWEEP, N_SWEEP = 2.0, 21
dn_g = np.linspace(-R_SWEEP, R_SWEEP, N_SWEEP)
db_g = np.linspace(-R_SWEEP, R_SWEEP, N_SWEEP)
positions = np.linspace(0, 0.25, 6)

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

# ── 逆推网格 ──
N = 41
Fn_grid = np.linspace(-20, 5, N)
Fo_grid = np.linspace(-8, 8, N)

models = [
    ('线性固定 dn',  inv_fixed,     'dn'),
    ('线性固定 db',  inv_fixed,     'db'),
    ('物理分块 dn',  inv_physical,  'dn'),
    ('物理分块 db',  inv_physical,  'db'),
    ('二次锚定 dn',  inv_quadratic, 'dn'),
    ('二次锚定 db',  inv_quadratic, 'db'),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 18))

for idx, (title, model_fn, field) in enumerate(models):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    data = np.zeros((N, N))
    for i, fn in enumerate(Fn_grid):
        for j, fo in enumerate(Fo_grid):
            dn_v, db_v = model_fn(fn, fo)
            data[j, i] = dn_v if field == 'dn' else db_v

    vmax = max(abs(data).max(), 1e-6)
    vmin = -vmax
    cs = ax.contourf(Fn_grid, Fo_grid, data, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(cs, ax=ax, shrink=0.85)

    # 采样点散点
    sc_data = scatter_dn if field == 'dn' else scatter_db
    ax.scatter(scatter_fn, scatter_fo, c=sc_data, s=2,
               cmap='RdBu_r', vmin=vmin, vmax=vmax, edgecolors='none', alpha=0.4)

    # 目标点
    ax.plot(-8, 0, 'ko', ms=8, mew=2, zorder=10)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
    ax.set_title(f'{title}')
    ax.set_aspect('auto')

fig.suptitle(
    '三种逆推模型力场对比  (Fn,Fo)→(dn,db)  —  热力图=模型预测  散点=真实力场采样',
    fontsize=14, y=0.995
)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'output', 'inverse_models_compare.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'✓ {out}')
