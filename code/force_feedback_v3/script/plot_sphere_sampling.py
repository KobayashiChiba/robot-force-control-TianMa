"""
plot_sphere_sampling.py — 球刀球面 + Fibonacci 采样点示意图

用法:
    cd code && python force_feedback_v3/script/plot_sphere_sampling.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from force_feedback_v3.lib.sphere_contact import R_BALL, N_SPHERE, _UNIT_SPHERE

# ── 采样点 ──
pts = R_BALL * _UNIT_SPHERE  # (12800, 3)

# ── 球面线框 ──
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 40)
x = R_BALL * np.outer(np.cos(u), np.sin(v))
y = R_BALL * np.outer(np.sin(u), np.sin(v))
z = R_BALL * np.outer(np.ones_like(u), np.cos(v))

# ── 绘图 ──
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(x, y, z, color='gray', linewidth=0.3, alpha=0.4)
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           c='steelblue', s=0.15, alpha=0.5)

# 球心
ax.scatter(0, 0, 0, c='red', s=30, zorder=5)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(f'Ball sphere (R={R_BALL}mm) + {N_SPHERE} sample points')
from matplotlib.lines import Line2D

# 图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
           markersize=6, label=f'{N_SPHERE} Fibonacci points'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='ball center'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# 等比例
lim = R_BALL * 1.3
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_aspect('equal')

out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'sphere_sampling.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'✓ Saved: {out_path}')
print(f'  R={R_BALL}mm, {N_SPHERE} points (Fibonacci)')
