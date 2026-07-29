"""
plot_contact_patch.py — 球刀与工件接触面示意图

展示双圆柱工件 + 球刀（含网格线）+ 接触斑 + 接触力箭头。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import R_BALL, _UNIT_SPHERE, sphere_contact_force
from force_feedback_v3.lib.simulator import Simulator
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
contact_curve = sim.contact_pts

# ── 球刀位置：从参考轨迹沿法向压深 0.5mm ──
P_ball0 = ball_ref[0].copy()
# 找最近接触点，获取法向
idx = np.argmin(np.linalg.norm(contact_curve - P_ball0, axis=1))
P_ct = contact_curve[idx]
basis0 = compute_point_basis_ortho(P_ct, sim.contact_geom)
n_dir = basis0.normal  # 指向工件内部
P_ball = P_ball0 + 0.5 * n_dir
print(f'Ball center: ({P_ball[0]:.1f}, {P_ball[1]:.1f}, {P_ball[2]:.1f})')
print(f'  pushed +0.5mm along normal')

# ── 球面采样点 + 接触判断 ──
pts = P_ball + R_BALL * _UNIT_SPHERE
in_z = (np.sqrt((pts[:, 0] - cz.p1[0])**2 + (pts[:, 1] - cz.p1[1])**2) < cz.radius - 1e-6)
in_y = (np.sqrt((pts[:, 0] - cy.p1[0])**2 + (pts[:, 2] - cy.p1[2])**2) < cy.radius - 1e-6)
contact = ~in_z & ~in_y
n_contact = contact.sum()
print(f'Contact: {n_contact}/{len(pts)} ({100*n_contact/len(pts):.1f}%)')

# ── 接触力 ──
F_vec, area = sphere_contact_force(P_ball, cz, cy)
F_mag = np.linalg.norm(F_vec)
print(f'Force: |F|={F_mag:.2f}N, area={area:.2f}mm²')

# ── 球面线框 ──
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 25)
xs = P_ball[0] + R_BALL * np.outer(np.cos(u), np.sin(v))
ys = P_ball[1] + R_BALL * np.outer(np.sin(u), np.sin(v))
zs = P_ball[2] + R_BALL * np.outer(np.ones_like(u), np.cos(v))

# ── 圆柱面 ──
cmin, cmax = contact_curve.min(axis=0) - 3, contact_curve.max(axis=0) + 3

dz2 = P_ball[0:2] - cz.p1[0:2]
theta_z_center = np.arctan2(dz2[1], dz2[0])
theta_z = np.linspace(theta_z_center - np.pi/3, theta_z_center + np.pi/3, 240)
z_vals = np.linspace(cmin[2], cmax[2], 160)
Tz, Thz = np.meshgrid(z_vals, theta_z)
Xz = cz.p1[0] + cz.radius * np.cos(Thz)
Yz = cz.p1[1] + cz.radius * np.sin(Thz)
Zz = Tz

theta = np.linspace(0, 2*np.pi, 320)
y_vals = np.linspace(cmin[1], cmax[1], 160)
Ty, Thy = np.meshgrid(y_vals, theta)
Xy = cy.p1[0] + cy.radius * np.cos(Thy)
Zy = cy.p1[2] + cy.radius * np.sin(Thy)
Yy = Ty

mask_z_in_y = np.sqrt((Xz - cy.p1[0])**2 + (Zz - cy.p1[2])**2) < cy.radius - 0.1
Zz_hollow = Zz.copy(); Zz_hollow[mask_z_in_y] = np.nan
mask_y_in_z = np.sqrt((Xy - cz.p1[0])**2 + (Yy - cz.p1[1])**2) < cz.radius - 0.1
Zy_hollow = Zy.copy(); Zy_hollow[mask_y_in_z] = np.nan

# ── 3D ──
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Xz, Yz, Zz_hollow, color='lightblue', alpha=0.35, linewidth=0)
ax.plot_surface(Xy, Yy, Zy_hollow, color='lightcoral', alpha=0.35, linewidth=0)
ax.plot(contact_curve[:, 0], contact_curve[:, 1], contact_curve[:, 2],
        'k-', linewidth=1.2, alpha=0.8, label='intersection curve')

ax.plot([cz.p1[0], cz.p1[0]], [cz.p1[1], cz.p1[1]],
        [cmin[2], cmax[2]], 'b--', linewidth=0.5, alpha=0.3)
ax.plot([cy.p1[0], cy.p1[0]], [cmin[1], cmax[1]],
        [cy.p1[2], cy.p1[2]], 'r--', linewidth=0.5, alpha=0.3)

# 球面线框
ax.plot_wireframe(xs, ys, zs, color='gray', linewidth=0.25, alpha=0.5)

# 非接触球面点
ax.scatter(pts[~contact, 0], pts[~contact, 1], pts[~contact, 2],
           c='lightgray', s=0.3, alpha=0.25)
# 接触球面点
ax.scatter(pts[contact, 0], pts[contact, 1], pts[contact, 2],
           c='red', s=2.0, alpha=0.85, label=f'contact ({n_contact} pts)')

ax.scatter(*P_ball, c='darkred', s=50, zorder=5, label='ball center')

# 接触力箭头（放大 0.3 倍方便看）
scale_f = 0.3
ax.quiver(*P_ball, *(F_vec * scale_f), color='orange', linewidth=2,
          arrow_length_ratio=0.15, label=f'|F|={F_mag:.1f}N')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f'Contact patch (R={R_BALL}mm, push +0.5mm)')
ax.legend(loc='upper left', fontsize=8)

rng = (np.ptp(contact_curve, axis=0).max() / 2) + 8
cx_c, cy_c, cz_c = contact_curve.mean(axis=0)
ax.set_xlim(cx_c - rng, cx_c + rng)
ax.set_ylim(cy_c - rng, cy_c + rng)
ax.set_zlim(cz_c - rng, cz_c + rng)
ax.set_aspect('equal')

out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'contact_patch.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'✓ Saved: {out_path}')
