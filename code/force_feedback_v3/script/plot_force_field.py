"""
plot_force_field.py — 多点力场热力图 + 法平面截面叠加

6 个曲线进度位置，每个画 |F|/Fn/Fo 三面板热力图，叠加圆柱截面 + 球刀圆。
截面虚实按"落入对方圆柱内部 = 虚线"划分（移植自 section_with_ball.py 验证过的逻辑）。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts; N_PTS = len(ball_ref)

PROGRESSES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
R_OFF = 2.0; N_GRID = 41
dn = np.linspace(-R_OFF, R_OFF, N_GRID)
db = np.linspace(-R_OFF, R_OFF, N_GRID)

# ═══════════════════════════════════════════
# 截面采样（移植自 section_with_ball.py）
# ═══════════════════════════════════════════

def _section_z_normal(phi, cyl_z, t, P0):
    X0, Y0 = cyl_z.p1[0], cyl_z.p1[1]; R = cyl_z.radius
    x = X0 + R * np.cos(phi); y = Y0 + R * np.sin(phi)
    h = (np.dot(t, P0) - t[0]*x - t[1]*y) / t[2]
    return np.array([x, y, h])

def _section_y_normal(theta, cyl_y, t, P0):
    X0, Z0 = cyl_y.p1[0], cyl_y.p1[2]; R = cyl_y.radius
    x = X0 + R * np.cos(theta); z = Z0 + R * np.sin(theta)
    h = (np.dot(t, P0) - t[0]*x - t[2]*z) / t[1]
    return np.array([x, h, z])

def _section_z_normal(phi, cyl_z, t, P0):
    X0, Y0 = cyl_z.p1[0], cyl_z.p1[1]; R = cyl_z.radius
    x = X0 + R * np.cos(phi); y = Y0 + R * np.sin(phi)
    h = (np.dot(t, P0) - t[0]*x - t[1]*y) / t[2]
    return np.array([x, y, h])

def _should_degenerate_z(p):
    """Z 圆柱截面拉长退化：p≈0 或 p≈0.5 附近（法平面∥Z轴）"""
    return min(abs(p - 0), abs(p - 0.5), abs(p - 1.0)) < 0.05


def _should_degenerate_y(p):
    """Y 圆柱截面拉长退化：p≈0.25 或 p≈0.75 附近（法平面∥Y轴）"""
    return min(abs(p - 0.25), abs(p - 0.75)) < 0.05


def _plot_split(ax, u, v, color):
    """u>0→实线 u<0→虚线，符号变化处插值插入 u=0，确保两端都到原点"""
    # 插值插入 u=0 点
    tr = np.where(np.diff(u >= 0))[0]
    if len(tr) > 0:
        u2, v2 = [], []
        for i in range(len(u)):
            u2.append(u[i]); v2.append(v[i])
            if i in tr:
                r = -u[i] / (u[i+1] - u[i]) if abs(u[i+1]-u[i]) > 1e-12 else 0
                u2.append(0.0); v2.append(v[i] + r*(v[i+1]-v[i]))
        u, v = np.array(u2), np.array(v2)

    # 分段：虚线段末尾附加 u=0 点
    mask_pos = u >= 0
    for mask, ls in [(u < 0, (0, (4, 3))), (mask_pos, '-')]:
        segs, start, in_seg = [], 0, mask[0]
        for i in range(1, len(mask)):
            if mask[i] != in_seg:
                if in_seg: segs.append((start, i))
                start = i; in_seg = mask[i]
        if in_seg: segs.append((start, len(mask)))
        for i0, i1 in segs:
            seg_u = np.array(u[i0:i1])
            seg_v = np.array(v[i0:i1])
            # 虚线段：末尾附加 u=0 插值点确保到达原点
            if ls != '-' and i1 < len(mask) and mask[i1] != mask[i1-1]:
                r0 = -u[i1-1] / (u[i1] - u[i1-1]) if abs(u[i1]-u[i1-1]) > 1e-12 else 0
                seg_u = np.append(seg_u, 0.0)
                seg_v = np.append(seg_v, v[i1-1] + r0*(v[i1]-v[i1-1]))
            if ls != '-' and i0 > 0 and mask[i0-1] != mask[i0]:
                r0 = u[i0-1] / (u[i0-1] - u[i0]) if abs(u[i0-1]-u[i0]) > 1e-12 else 0
                seg_u = np.insert(seg_u, 0, 0.0)
                seg_v = np.insert(seg_v, 0, v[i0] + r0*(v[i0-1]-v[i0]))
            ax.plot(seg_u, seg_v, color=color, lw=1.2, linestyle=ls)


def _plot_sections(ax, P_ct, t, n, o, cyl_z, cyl_y, degenerate_z, degenerate_y,
                   uv_clip=5.0):
    """截面叠加：退化→原点出发的两段直线，正常→u>0实线u<0虚线"""
    N = 2000
    def to_uv(P): return np.dot(P-P_ct, n), np.dot(P-P_ct, o)
    clip = lambda u, v: (np.abs(u) < uv_clip) & (np.abs(v) < uv_clip)

    # Z 圆柱
    if abs(t[2]) > 1e-6:
        if degenerate_z:
            du, dv = n[2], o[2]; L = np.hypot(du, dv)
            if L > 1e-6:
                du, dv = du/L, dv/L; sign = 1 if du>0 else -1; ext = uv_clip*1.5
                ax.plot([0, sign*ext*du], [0, sign*ext*dv], color='darkcyan', lw=1.5)
                ax.plot([0, -sign*ext*du], [0, -sign*ext*dv], color='darkcyan', lw=1.5, linestyle=(0,(4,3)))
        else:
            uv_pts = []
            for phi in np.linspace(0, 2*np.pi, N):
                P = _section_z_normal(phi, cyl_z, t, P_ct)
                uu, vv = to_uv(P)
                if clip(uu, vv):
                    uv_pts.append([uu, vv])
            if uv_pts:
                a = np.array(uv_pts)
                _plot_split(ax, a[:,0], a[:,1], 'darkcyan')

    # Y 圆柱
    if abs(t[1]) > 1e-6:
        if degenerate_y:
            du, dv = n[1], o[1]; L = np.hypot(du, dv)
            if L > 1e-6:
                du, dv = du/L, dv/L; sign = 1 if du>0 else -1; ext = uv_clip*1.5
                ax.plot([0, sign*ext*du], [0, sign*ext*dv], color='green', lw=1.5)
                ax.plot([0, -sign*ext*du], [0, -sign*ext*dv], color='green', lw=1.5, linestyle=(0,(4,3)))
        else:
            uv_pts = []
            for phi in np.linspace(0, 2*np.pi, N):
                P = _section_y_normal(phi, cyl_y, t, P_ct)
                uu, vv = to_uv(P)
                if clip(uu, vv):
                    uv_pts.append([uu, vv])
            if uv_pts:
                a = np.array(uv_pts)
                _plot_split(ax, a[:,0], a[:,1], 'green')

# ═══════════════════════════════════════════
# 画图
# ═══════════════════════════════════════════
fig, all_axes = plt.subplots(3, len(PROGRESSES), figsize=(5.5*len(PROGRESSES), 16))

for col, p in enumerate(PROGRESSES):
    i = int(p * (N_PTS - 1))
    P_ball = ball_ref[i]
    idx = np.argmin(np.linalg.norm(ct - P_ball, axis=1))
    P_ct = ct[idx]
    basis = compute_point_basis_ortho(P_ct, sim.contact_geom)
    n, o, t = basis.normal, basis.ortho, basis.tangent

    # 力场热力图
    Fmag = np.zeros((N_GRID, N_GRID))
    Fn_arr = np.zeros_like(Fmag); Fo_arr = np.zeros_like(Fmag)
    for ii, dni in enumerate(dn):
        for jj, dbj in enumerate(db):
            pos = P_ball + dni * n + dbj * o
            F, _ = sphere_contact_force(pos, cz, cy)
            Fmag[jj, ii] = np.linalg.norm(F)
            Fn_arr[jj, ii] = np.dot(F, n)
            Fo_arr[jj, ii] = np.dot(F, o)

    for row, (data, ttl, cmap) in enumerate([
        (Fmag, '|F| (N)', 'Reds'), (Fn_arr, 'Fn (N)', 'RdBu_r'), (Fo_arr, 'Fo (N)', 'RdBu_r'),
    ]):
        ax = all_axes[row, col]
        vmax = max(abs(data).max(), 1e-6)
        vmin = 0 if cmap == 'Reds' else -vmax
        cs = ax.contourf(dn, db, data, levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(cs, ax=ax, shrink=0.8)
        ax.plot(0, 0, 'r+', ms=8, mew=2)
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_xlim(-R_OFF, R_OFF); ax.set_ylim(-R_OFF, R_OFF)
        ax.set_aspect('equal')
        if row == 0: ax.set_title(f'p = {p:.2f}')
        if col == 0: ax.set_ylabel(ttl)
        dz = _should_degenerate_z(p)
        dy = _should_degenerate_y(p)
        _plot_sections(ax, P_ct, t, n, o, cz, cy, dz, dy)
        ax.add_patch(Circle((0, 0), R_BALL, fill=False, color='white', lw=1))
        ax.plot(0, 0, 'ko', ms=4)

fig.suptitle('Multi-position force field (ball center = origin, section at contact point)', fontsize=14)
fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'force_field.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f'✓ Saved: {out_path}')
