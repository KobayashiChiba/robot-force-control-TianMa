"""
plot_curves_and_basis.py — 绘制接触曲线、球刀中心曲线、及某点的力分解基底

用法:
    cd code && python force_feedback_v3/script/plot_curves_and_basis.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.simulator import Simulator

# ── 加载数据 ──
cy, cz = load_cylinders()
ball_ref, L = load_ball_ref()
sim = Simulator(cy, cz)
contact_pts = sim.contact_pts  # (2000, 3)

# ── 均匀选 20 个展示点 ──
n_basis = 20
indices = np.linspace(0, len(contact_pts) - 1, n_basis, dtype=int)


# ── 3D 绘图 ──
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 接触曲线
ax.plot(contact_pts[:, 0], contact_pts[:, 1], contact_pts[:, 2],
        'b-', linewidth=0.8, alpha=0.6, label='contact curve')

# 球刀中心曲线
ax.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2],
        'r-', linewidth=0.8, alpha=0.6, label='ball center curve')

# 展示点
ax.scatter(contact_pts[indices, 0], contact_pts[indices, 1], contact_pts[indices, 2],
           c='k', s=15, zorder=5)

# 力分解基底（长度 2mm，细线）
scale = 2.0
colors = {'tangent': 'red', 'normal': 'green', 'ortho': 'blue'}
labels_done = set()
for idx in indices:
    P = contact_pts[idx]
    basis = compute_point_basis_ortho(P, sim.contact_geom)
    for vec_name, vec in [('tangent', basis.tangent),
                           ('normal', basis.normal),
                           ('ortho', basis.ortho)]:
        lbl = f'{vec_name[0]}' if vec_name not in labels_done else None
        if lbl:
            labels_done.add(vec_name)
        ax.quiver(*P, *(vec * scale), color=colors[vec_name], linewidth=0.6,
                  arrow_length_ratio=0.2, label=lbl)

# ── 等比例 + 标签 ──
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(f'Contact curve & {n_basis} basis frames')
ax.legend(loc='upper left', fontsize=8)

# 统一坐标范围
all_pts = np.vstack([contact_pts, ball_ref])
mid = all_pts.mean(axis=0)
span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 + 3
for ax_obj, lim in [(ax, 'x'), (ax, 'y'), (ax, 'z')]:
    getattr(ax, f'set_{lim}lim')(mid[0] - span, mid[0] + span)  # rough, fine-tune below

# 精确: 三轴统一 span
rng = np.ptp(all_pts, axis=0).max() / 2 + 5
cx, cy_c, cz_c = all_pts.mean(axis=0)
ax.set_xlim(cx - rng, cx + rng)
ax.set_ylim(cy_c - rng, cy_c + rng)
ax.set_zlim(cz_c - rng, cz_c + rng)
ax.set_aspect('equal')

out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'curves_and_basis.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'✓ Saved: {out_path}')
print(f'  contact curve: {contact_pts.shape[0]} pts')
print(f'  ball center curve: {ball_ref.shape[0]} pts')
print(f'  {n_basis} basis frames drawn')
