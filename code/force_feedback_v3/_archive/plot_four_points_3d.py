"""
plot_four_points_3d.py — p=0.75四个关键点3D标注
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref, generate_error_cylinders
from force_feedback_v3.lib.cylinder_geometry import sample_intersection
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(5)
cy, cz = load_cylinders(); ball_ref, _ = load_ball_ref()
rng = np.random.RandomState(5)
cz_err, cy_err = generate_error_cylinders(cy, cz, rng)

bc0 = ball_ref[375]  # p=0.75

# 标准接触曲线
ct_std = sample_intersection(cy, cz, n_samples=2000).sample_pts
is_std = np.argmin(np.linalg.norm(ct_std-bc0, axis=1))
Pc_std = ct_std[is_std]
basis = compute_point_basis_ortho(Pc_std, sample_intersection(cy,cz,n_samples=2000))
n,o,t = basis.normal,basis.ortho,basis.tangent

# 误差接触曲线
ct_err = sample_intersection(cy_err, cz_err, n_samples=2000).sample_pts
ie_near = np.argmin(np.linalg.norm(ct_err-bc0, axis=1))
Pc_near = ct_err[ie_near]

# 误差截面交点（正确：双扫phi+theta）
best_3d=1e9; bp=None; bt=None
for phi in np.linspace(0,2*np.pi,2000):
    xz=cz_err.p1[0]+cz_err.radius*np.cos(phi); yz=cz_err.p1[1]+cz_err.radius*np.sin(phi)
    zz=(np.dot(t,Pc_std)-t[0]*xz-t[1]*yz)/t[2]
    for theta in np.linspace(0,2*np.pi,2000):
        xc=cy_err.p1[0]+cy_err.radius*np.cos(theta); zc=cy_err.p1[2]+cy_err.radius*np.sin(theta)
        if abs(t[1])<1e-6: continue
        yc=(np.dot(t,Pc_std)-t[0]*xc-t[2]*zc)/t[1]
        d3=np.sqrt((xz-xc)**2+(yz-yc)**2+(zz-zc)**2)
        if d3<best_3d: best_3d=d3; bp=phi; bt=theta
xz=cz_err.p1[0]+cz_err.radius*np.cos(bp); yz=cz_err.p1[1]+cz_err.radius*np.sin(bp)
zz=(np.dot(t,Pc_std)-t[0]*xz-t[1]*yz)/t[2]
xc=cy_err.p1[0]+cy_err.radius*np.cos(bt); zc=cy_err.p1[2]+cy_err.radius*np.sin(bt)
yc=(np.dot(t,Pc_std)-t[0]*xc-t[2]*zc)/t[1]
Pc_section=np.array([(xz+xc)/2,(yz+yc)/2,(zz+zc)/2])

# 参考轨迹
ref = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_trajectory.npz'))
ref_ball = ref['ball_actual']

# ── 3D图 ──
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 曲线
ax.plot(ct_std[:,0], ct_std[:,1], ct_std[:,2], 'gray', ls='--', lw=0.8, label='contact std', zorder=1)
ax.plot(ct_err[:,0], ct_err[:,1], ct_err[:,2], 'green', lw=0.8, alpha=0.4, label='contact err', zorder=1)
ax.plot(ref_ball[:,0], ref_ball[:,1], ref_ball[:,2], 'steelblue', ls=':', lw=1, alpha=0.3, label='ball ref', zorder=1)

# 四个点
ax.scatter(*bc0, c='black', s=120, marker='o', zorder=10, label='bc0 (p=0.75)')
ax.scatter(*Pc_std, c='blue', s=120, marker='s', zorder=10, label='Pc std')
ax.scatter(*Pc_section, c='red', s=150, marker='D', zorder=10, label='Pc err (section)')
ax.scatter(*Pc_near, c='magenta', s=120, marker='^', zorder=10, label='Pc near bc0')

# bc0到四个点的连线
for pt, color, style in [(Pc_std, 'blue', '--'), (Pc_section, 'red', '-'), (Pc_near, 'magenta', ':')]:
    ax.plot([bc0[0], pt[0]], [bc0[1], pt[1]], [bc0[2], pt[2]], color=color, lw=0.8, ls=style, alpha=0.5)

c = np.mean(ct_std, axis=0)
ax.set_xlim(c[0]-15, c[0]+15); ax.set_ylim(c[1]-15, c[1]+15); ax.set_zlim(c[2]-15, c[2]+15)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('p=0.75: 4 key points on 3D curves')
ax.legend(fontsize=7, loc='upper left')
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'four_points_3d.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'✓ {out}')
print(f'\nbc0:            {np.round(bc0,2)}')
print(f'Pc_std(蓝):     {np.round(Pc_std,2)}  dist={np.linalg.norm(Pc_std-bc0):.2f}mm')
print(f'Pc_section(红): {np.round(Pc_section,2)}  dist={np.linalg.norm(Pc_section-bc0):.2f}mm')
print(f'Pc_near(紫):    {np.round(Pc_near,2)}  dist={np.linalg.norm(Pc_near-bc0):.2f}mm')
