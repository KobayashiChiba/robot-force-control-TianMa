"""
plot_sections_full.py — 完整截面，退化位置画直线
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts

PROGS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
N = 2000

def _should_degenerate_z(p):
    return min(abs(p-0), abs(p-0.5), abs(p-1.0)) < 0.05

def _should_degenerate_y(p):
    return min(abs(p-0.25), abs(p-0.75)) < 0.05

def sec_z(phi, cz, t, P0):
    x = cz.p1[0] + cz.radius * np.cos(phi)
    y = cz.p1[1] + cz.radius * np.sin(phi)
    z = (np.dot(t,P0) - t[0]*x - t[1]*y) / t[2]
    return np.array([x, y, z])

def sec_y(th, cy, t, P0):
    x = cy.p1[0] + cy.radius * np.cos(th)
    z = cy.p1[2] + cy.radius * np.sin(th)
    y = (np.dot(t,P0) - t[0]*x - t[2]*z) / t[1]
    return np.array([x, y, z])

def _plot_section_split(ax, u, v, color):
    """u>0 → 实线, u<0 → 虚线，符号变化处插值插入 u=0，确保两端都到原点"""
    # 插值插入 u=0 点
    transitions = np.where(np.diff(u >= 0))[0]
    if len(transitions) > 0:
        u_new, v_new = [], []
        for i in range(len(u)):
            u_new.append(u[i]); v_new.append(v[i])
            if i in transitions:
                t_ratio = -u[i] / (u[i+1] - u[i]) if abs(u[i+1]-u[i]) > 1e-12 else 0
                u_new.append(0.0)
                v_new.append(v[i] + t_ratio * (v[i+1] - v[i]))
        u, v = np.array(u_new), np.array(v_new)

    mask_pos = u >= 0

    for mask, ls in [(u < 0, (0, (4, 3))), (mask_pos, '-')]:
        segments = []
        start = 0
        in_seg = mask[0]
        for i in range(1, len(mask)):
            if mask[i] != in_seg:
                if in_seg:
                    segments.append((start, i))
                start = i
                in_seg = mask[i]
        if in_seg:
            segments.append((start, len(mask)))

        for i0, i1 in segments:
            seg_u = np.array(u[i0:i1])
            seg_v = np.array(v[i0:i1])
            # 虚线段：首尾附加 u=0 插值点确保到达原点
            if ls != '-' and i1 < len(mask) and mask[i1] != mask[i1-1]:
                r0 = -u[i1-1] / (u[i1] - u[i1-1]) if abs(u[i1]-u[i1-1]) > 1e-12 else 0
                seg_u = np.append(seg_u, 0.0)
                seg_v = np.append(seg_v, v[i1-1] + r0*(v[i1]-v[i1-1]))
            if ls != '-' and i0 > 0 and mask[i0-1] != mask[i0]:
                r0 = u[i0-1] / (u[i0-1] - u[i0]) if abs(u[i0-1]-u[i0]) > 1e-12 else 0
                seg_u = np.insert(seg_u, 0, 0.0)
                seg_v = np.insert(seg_v, 0, v[i0] + r0*(v[i0-1]-v[i0]))
            ax.plot(seg_u, seg_v, color=color, lw=1.2, linestyle=ls)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for col, p in enumerate(PROGS):
    i = int(p * (len(ball_ref)-1))
    idx = np.argmin(np.linalg.norm(ct - ball_ref[i], axis=1))
    P_ct = ct[idx]
    basis = compute_point_basis_ortho(P_ct, sim.contact_geom)
    n, o, t = basis.normal, basis.ortho, basis.tangent

    deg_z = _should_degenerate_z(p)
    deg_y = _should_degenerate_y(p)

    ax = axes[col]
    title_parts = [f'p={p:.2f}']
    if deg_z: title_parts.append('Z→直线')
    if deg_y: title_parts.append('Y→直线')

    # Z 圆柱
    if abs(t[2]) > 1e-6:
        if deg_z:
            du, dv = n[2], o[2]; L = np.hypot(du, dv)
            if L > 1e-6:
                du, dv = du/L, dv/L
                ext = 40
                sign = 1 if du > 0 else -1
                ax.plot([0, sign*ext*du], [0, sign*ext*dv], color='darkcyan', lw=1.5)
                ax.plot([0, -sign*ext*du], [0, -sign*ext*dv], color='darkcyan', lw=1.5, linestyle='--')
                ax.plot(0, 0, 'k+', ms=8, mew=1.5)
        else:
            pts = np.array([sec_z(phi, cz, t, P_ct) for phi in np.linspace(0,2*np.pi,N)])
            u = np.array([np.dot(pt-P_ct,n) for pt in pts])
            v = np.array([np.dot(pt-P_ct,o) for pt in pts])
            _plot_section_split(ax, u, v, 'darkcyan')

    # Y 圆柱
    if abs(t[1]) > 1e-6:
        if deg_y:
            du, dv = n[1], o[1]; L = np.hypot(du, dv)
            if L > 1e-6:
                du, dv = du/L, dv/L
                ext = 40
                sign = 1 if du > 0 else -1
                ax.plot([0, sign*ext*du], [0, sign*ext*dv], color='green', lw=1.5)
                ax.plot([0, -sign*ext*du], [0, -sign*ext*dv], color='green', lw=1.5, linestyle='--')
                ax.plot(0, 0, 'k+', ms=8, mew=1.5)
        else:
            pts = np.array([sec_y(th, cy, t, P_ct) for th in np.linspace(0,2*np.pi,N)])
            u = np.array([np.dot(pt-P_ct,n) for pt in pts])
            v = np.array([np.dot(pt-P_ct,o) for pt in pts])
            _plot_section_split(ax, u, v, 'green')

    ax.plot(0, 0, 'k+', ms=12, mew=2)
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)
    ax.set_aspect('equal')
    ax.set_title(', '.join(title_parts))

fig.suptitle('Full sections — Z:cyan Y:green, solid=dash degenerate', fontsize=14)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__),'..','output','sections_full.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
