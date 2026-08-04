"""
plot_error_force_field.py — 误差工件局部力场 (seed5, p=0.25)

(dn,db)∈[-2,2]mm → sphere_contact_force on error cyls → |F|/Fn/Fo + 截面叠加
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from force_feedback_v3.lib import load_cylinders, load_ball_ref, generate_error_cylinders
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.cylinder_geometry import sample_intersection

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(5)
cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
rng = np.random.RandomState(5)
cz_err, cy_err = generate_error_cylinders(cy, cz, rng)

# 误差接触曲线
contact_err_geom = sample_intersection(cy_err, cz_err, n_samples=2000)
ct = contact_err_geom.sample_pts

# p=0.78 (步500对应角度9°→ball_ref[389])
p = 0.78; i = 389
bc0 = ball_ref[i]
idx = np.argmin(np.linalg.norm(ct - bc0, axis=1))
P_ct = ct[idx]
basis = compute_point_basis_ortho(P_ct, contact_err_geom)
n, o, t = basis.normal, basis.ortho, basis.tangent

print(f'p=0.25, P_ct={np.round(P_ct,3)}')
print(f'n={np.round(n,3)} o={np.round(o,3)}')

# ── 截面叠加 ──
def _sec_z(phi, cyl, t_vec, P0):
    X0,Y0=cyl.p1[0],cyl.p1[1]; R=cyl.radius
    x=X0+R*np.cos(phi); y=Y0+R*np.sin(phi)
    return np.array([x,y,(np.dot(t_vec,P0)-t_vec[0]*x-t_vec[1]*y)/t_vec[2]])
def _sec_y(th, cyl, t_vec, P0):
    X0,Z0=cyl.p1[0],cyl.p1[2]; R=cyl.radius
    x=X0+R*np.cos(th); z=Z0+R*np.sin(th)
    return np.array([x,(np.dot(t_vec,P0)-t_vec[0]*x-t_vec[2]*z)/t_vec[1],z])

def _plot_split(ax, u, v, color):
    """u>0→实线 u<0→虚线"""
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
            seg_u = np.array(u[i0:i1])
            seg_v = np.array(v[i0:i1])
            if ls != '-' and i1 < len(mask) and mask[i1] != mask[i1-1]:
                r0 = -u[i1-1] / (u[i1] - u[i1-1]) if abs(u[i1]-u[i1-1]) > 1e-12 else 0
                seg_u = np.append(seg_u, 0.0)
                seg_v = np.append(seg_v, v[i1-1] + r0*(v[i1]-v[i1-1]))
            if ls != '-' and i0 > 0 and mask[i0-1] != mask[i0]:
                r0 = u[i0-1] / (u[i0-1] - u[i0]) if abs(u[i0-1]-u[i0]) > 1e-12 else 0
                seg_u = np.insert(seg_u, 0, 0.0)
                seg_v = np.insert(seg_v, 0, v[i0] + r0*(v[i0-1]-v[i0]))
            ax.plot(seg_u, seg_v, color=color, lw=1.2, linestyle=ls)

def draw_sections(ax, cy_cyl, cz_cyl, P_ct, t, n, o):
    N2 = 2000; uv_clip = 5.0
    to_uv = lambda P: (np.dot(P-P_ct,n), np.dot(P-P_ct,o))
    clip = lambda u,v: (abs(u)<uv_clip) & (abs(v)<uv_clip)

    # Z圆柱截面
    if abs(t[2]) > 1e-6:
        uv_pts = []
        for phi in np.linspace(0, 2*np.pi, N2):
            P = _sec_z(phi, cz_cyl, t, P_ct)
            uu, vv = to_uv(P)
            if clip(uu, vv): uv_pts.append([uu, vv])
        if uv_pts:
            a = np.array(uv_pts)
            _plot_split(ax, a[:,0], a[:,1], 'darkcyan')

    # Y圆柱截面
    if abs(t[1]) > 1e-6:
        uv_pts = []
        for phi in np.linspace(0, 2*np.pi, N2):
            P = _sec_y(phi, cy_cyl, t, P_ct)
            uu, vv = to_uv(P)
            if clip(uu, vv): uv_pts.append([uu, vv])
        if uv_pts:
            a = np.array(uv_pts)
            _plot_split(ax, a[:,0], a[:,1], 'green')

# ── 力场采样 ──
R_OFF = 2.0; NG = 51
dv = np.linspace(-R_OFF, R_OFF, NG)
Fmag = np.zeros((NG, NG)); Fn_arr = np.zeros_like(Fmag); Fo_arr = np.zeros_like(Fmag)
for ii, dni in enumerate(dv):
    for jj, dbj in enumerate(dv):
        pos = bc0 + dni * n + dbj * o
        F, area = sphere_contact_force(pos, cz_err, cy_err)
        Fmag[jj, ii] = np.linalg.norm(F)
        Fn_arr[jj, ii] = np.dot(F, n)
        Fo_arr[jj, ii] = np.dot(F, o)

# ── 标记真实球心点 ──
ball_dn = 1.46  # mm (步500球刀 vs bc0 p=0.78)
ball_db = 0.52  # mm
ball_label = f'sim pos\n({ball_dn:.2f},{ball_db:.2f})'

# ── 画图 1×3 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, data, title, cmap in [
    (axes[0], Fmag, '|F| (N)', 'Reds'),
    (axes[1], Fn_arr, 'Fn (N)', 'RdBu_r'),
    (axes[2], Fo_arr, 'Fo (N)', 'RdBu_r'),
]:
    vmax = max(abs(data).max(), 1e-6)
    vmin = 0 if cmap == 'Reds' else -vmax
    cs = ax.contourf(dv, dv, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cs, ax=ax, shrink=0.85)
    draw_sections(ax, cy, cz, P_ct, t, n, o)
    ax.add_patch(Circle((0, 0), R_BALL, fill=False, color='white', lw=1))
    ax.plot(0, 0, 'k+', ms=10, mew=2, zorder=10)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlim(-R_OFF, R_OFF); ax.set_ylim(-R_OFF, R_OFF)
    ax.set_aspect('equal')
    ax.set_xlabel('dn (mm)'); ax.set_ylabel('db (mm)')
    ax.set_title(title)
    ax.plot(ball_dn, ball_db, 'rx', ms=12, mew=3, zorder=10, label=ball_label)
    ax.legend(fontsize=8, loc='lower right')
    draw_sections(ax, cy, cz, P_ct, t, n, o)

fig.suptitle('Error workpiece force field — seed=5, p=0.25 (error cyls + sections)', fontsize=14)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'error_force_field_p025.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'✓ {out}')
