"""
plot_inverse_residual.py — 逆推残差热力图（6位置，dn/db域）
与力场图同坐标，颜色=逆推残差，对照看出拟合精度
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.force_field_quadratic import inverse
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts

PROGRESSES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
R_OFF = 2.0; N_GRID = 41
dn = np.linspace(-R_OFF, R_OFF, N_GRID)
db = np.linspace(-R_OFF, R_OFF, N_GRID)

fig, axes = plt.subplots(2, len(PROGRESSES), figsize=(5.5*len(PROGRESSES), 10))

for col, p in enumerate(PROGRESSES):
    i = int(p * (len(ball_ref) - 1))
    P_ball = ball_ref[i]
    idx = np.argmin(np.linalg.norm(ct - P_ball, axis=1))
    basis = compute_point_basis_ortho(ct[idx], sim.contact_geom)
    n, o = basis.normal, basis.ortho

    ERR_DN = np.zeros((N_GRID, N_GRID))
    ERR_DB = np.zeros((N_GRID, N_GRID))
    for ii, dni in enumerate(dn):
        for jj, dbj in enumerate(db):
            pos = P_ball + dni*n + dbj*o
            F, _ = sphere_contact_force(pos, cz, cy)
            fn = np.dot(F, n); fo = np.dot(F, o)
            dn_pred, db_pred = inverse(fn, fo)
            ERR_DN[jj, ii] = abs(dn_pred - dni)
            ERR_DB[jj, ii] = abs(db_pred - dbj)

    for row, (data, ttl) in enumerate([(ERR_DN, '|Δdn| (mm)'), (ERR_DB, '|Δdb| (mm)')]):
        ax = axes[row, col]
        cs = ax.contourf(dn, db, data, levels=15, cmap='YlOrRd', vmin=0, vmax=max(data.max(), 0.01))
        plt.colorbar(cs, ax=ax, shrink=0.8)
        ax.plot(0, 0, 'k+', ms=8, mew=2)
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_xlim(-R_OFF, R_OFF); ax.set_ylim(-R_OFF, R_OFF)
        ax.set_aspect('equal')
        if row == 0: ax.set_title(f'p = {p:.2f}')
        if col == 0: ax.set_ylabel(ttl)

fig.suptitle('Inverse residual (|predicted − actual|), same domain as force field', fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'inverse_residual.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
