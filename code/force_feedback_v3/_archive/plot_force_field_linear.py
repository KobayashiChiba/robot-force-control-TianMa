"""
plot_force_field_linear.py — 线性逆推力场 vs 真实力场对比

线性模型: Fn = F_TARGET - KN*dn, Fo = -KO*db
真实力场: sphere_contact_force 采样

每个位置叠两层: 真实力场(热力图) + 线性模型(等值线)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 线性逆推参数 ──
F_TARGET = -8.0
KN = -24.533
KO = 1.4

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts; N_PTS = len(ball_ref)

PROGRESSES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
R_OFF = 2.0; N_GRID = 51
dn = np.linspace(-R_OFF, R_OFF, N_GRID)
db = np.linspace(-R_OFF, R_OFF, N_GRID)

# ═══════════════════════════════════════════
# 截面叠加（复用自 plot_force_field.py）
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

def _should_degenerate_z(p):
    return min(abs(p - 0), abs(p - 0.5), abs(p - 1.0)) < 0.05

def _should_degenerate_y(p):
    return min(abs(p - 0.25), abs(p - 0.75)) < 0.05

def _plot_split(ax, u, v, color, lw=1.2):
    tr = np.where(np.diff(u >= 0))[0]
    if len(tr) > 0:
        u2, v2 = [], []
        for i in range(len(u)):
            u2.append(u[i]); v2.append(v[i])
            if i in tr:
                r = -u[i] / (u[i+1] - u[i]) if abs(u[i+1]-u[i]) > 1e-12 else 0
                u2.append(0.0); v2.append(v[i] + r*(v[i+1]-v[i]))
        u, v = np.array(u2), np.array(v2)
    mask_pos = u >= 0
    for mask, ls in [(u < 0, (0, (4, 3))), (mask_pos, '-')]:
        segs, start, in_seg = [], 0, mask[0]
        for i in range(1, len(mask)):
            if mask[i] != in_seg:
                if in_seg: segs.append((start, i))
                start = i; in_seg = mask[i]
        if in_seg: segs.append((start, len(mask)))
        for i0, i1 in segs:
            seg_u, seg_v = np.array(u[i0:i1]), np.array(v[i0:i1])
            ax.plot(seg_u, seg_v, color=color, lw=lw, linestyle=ls)

def _plot_sections(ax, P_ct, t, n, o, cz, cy, dz, dy, uv_clip=5.0):
    N = 2000
    def to_uv(P): return np.dot(P-P_ct, n), np.dot(P-P_ct, o)
    clip = lambda u, v: (np.abs(u) < uv_clip) & (np.abs(v) < uv_clip)
    if abs(t[2]) > 1e-6:
        if dz:
            du, dv = n[2], o[2]; L = np.hypot(du, dv)
            if L > 1e-6:
                du, dv = du/L, dv/L; sign = 1 if du>0 else -1; ext = uv_clip*1.5
                ax.plot([0, sign*ext*du], [0, sign*ext*dv], color='darkcyan', lw=1.5)
                ax.plot([0, -sign*ext*du], [0, -sign*ext*dv], color='darkcyan', lw=1.5, linestyle=(0,(4,3)))
        else:
            uv_pts = []
            for phi in np.linspace(0, 2*np.pi, N):
                P = _section_z_normal(phi, cz, t, P_ct)
                uu, vv = to_uv(P)
                if clip(uu, vv): uv_pts.append([uu, vv])
            if uv_pts:
                a = np.array(uv_pts); _plot_split(ax, a[:,0], a[:,1], 'darkcyan')
    if abs(t[1]) > 1e-6:
        if dy:
            du, dv = n[1], o[1]; L = np.hypot(du, dv)
            if L > 1e-6:
                du, dv = du/L, dv/L; sign = 1 if du>0 else -1; ext = uv_clip*1.5
                ax.plot([0, sign*ext*du], [0, sign*ext*dv], color='green', lw=1.5)
                ax.plot([0, -sign*ext*du], [0, -sign*ext*dv], color='green', lw=1.5, linestyle=(0,(4,3)))
        else:
            uv_pts = []
            for phi in np.linspace(0, 2*np.pi, N):
                P = _section_y_normal(phi, cy, t, P_ct)
                uu, vv = to_uv(P)
                if clip(uu, vv): uv_pts.append([uu, vv])
            if uv_pts:
                a = np.array(uv_pts); _plot_split(ax, a[:,0], a[:,1], 'green')

# ═══════════════════════════════════════════
# 画图: 3行×6列 = |F|/Fn/Fo × 6个位置
# 每格: 真实力场热力图 + 线性模型等值线(白色虚线)
# ═══════════════════════════════════════════
fig, all_axes = plt.subplots(3, len(PROGRESSES), figsize=(5.5*len(PROGRESSES), 16))

for col, p in enumerate(PROGRESSES):
    i = int(p * (N_PTS - 1))
    P_ball = ball_ref[i]
    idx = np.argmin(np.linalg.norm(ct - P_ball, axis=1))
    P_ct = ct[idx]
    basis = compute_point_basis_ortho(P_ct, sim.contact_geom)
    n, o, t = basis.normal, basis.ortho, basis.tangent

    # 真实力场采样
    Fmag = np.zeros((N_GRID, N_GRID))
    Fn_arr = np.zeros_like(Fmag); Fo_arr = np.zeros_like(Fmag)
    for ii, dni in enumerate(dn):
        for jj, dbj in enumerate(db):
            pos = P_ball + dni * n + dbj * o
            F, _ = sphere_contact_force(pos, cz, cy)
            Fmag[jj, ii] = np.linalg.norm(F)
            Fn_arr[jj, ii] = np.dot(F, n)
            Fo_arr[jj, ii] = np.dot(F, o)

    # 线性模型力场 (解析): Fn只依赖dn, Fo只依赖db
    Fn_lin_2d = np.tile(F_TARGET - KN * dn, (N_GRID, 1))       # 每行相同
    Fo_lin_2d = np.tile(-KO * db.reshape(-1, 1), (1, N_GRID))  # 每列相同

    for row, (data, lin_data, ttl, cmap) in enumerate([
        (Fmag, None, '|F| (N)', 'Reds'),
        (Fn_arr, Fn_lin_2d, 'Fn (N)', 'RdBu_r'),
        (Fo_arr, Fo_lin_2d, 'Fo (N)', 'RdBu_r'),
    ]):
        ax = all_axes[row, col]
        vmax = max(abs(data).max(), 1e-6)
        vmin = 0 if cmap == 'Reds' else -vmax
        cs = ax.contourf(dn, db, data, levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(cs, ax=ax, shrink=0.8)

        # 线性模型等值线 (白色虚线)
        if lin_data is not None:
            # 只画有限几根等值线
            lv_min, lv_max = lin_data.min(), lin_data.max()
            levels = np.linspace(lv_min, lv_max, 7)
            ax.contour(dn, db, lin_data, levels=levels, colors='white',
                       linewidths=1.0, linestyles='--', alpha=0.7)

        # 线性模型零线 (dn/db=0 时的力)
        if cmap != 'Reds':
            ax.axhline(0, color='white', lw=0.8, ls=':', alpha=0.5)
            ax.axvline(0, color='white', lw=0.8, ls=':', alpha=0.5)
            # 标记线性模型的 Fn=F_TARGET 线 (dn=0)
            # 和 Fo=0 线 (db=0)

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

fig.suptitle(
    '力场对比 — 热力图=真实力场  |  白虚线=线性逆推模型  (dn∥法向 db∥o方向)',
    fontsize=14, y=0.995
)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'force_field_linear_compare.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'✓ {out}')
