"""
p0_force_field.py — p=0 标准球心力场热力图

原点 (0,0) = 标准球刀中心，±2mm 偏移
三个面板：|F|（Reds）、Fn（RdBu_r）、Fo（RdBu_r）
红+ = 标准球心位置
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from force_profile import sphere_contact_force
from contact_frame_v2 import compute_frame

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

with open('../data/force_model.pkl', 'rb') as f:
    d = pickle.load(f)

bc0 = d['ball_center_500'][0]
cy, cz = d['cyl_contact_y'], d['cyl_contact_z']
cg = d['contact_geom'].sample_pts

idx = np.argmin(np.linalg.norm(cg - bc0, axis=1))
Pc = cg[idx]

frame = compute_frame(Pc, cy, cz)
n_vec = frame.normal
b_vec = np.cross(frame.tangent, n_vec)
b_vec /= np.linalg.norm(b_vec)

# 网格：以标准球心为原点，±2mm
R = 2
N = 41
dn = np.linspace(-R, R, N)
dv = np.linspace(-R, R, N)
Fmag = np.zeros((N, N))
Fn = np.zeros_like(Fmag)
Fo = np.zeros_like(Fmag)

for i, dni in enumerate(dn):
    for j, dbj in enumerate(dv):
        f, _ = sphere_contact_force(bc0 + dni*n_vec + dbj*b_vec,
                                    np.array([0, -1, 0]), cz, cy)
        Fmag[j, i] = np.linalg.norm(f)
        Fn[j, i] = np.dot(f, n_vec)
        Fo[j, i] = np.dot(f, b_vec)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, data, title, cmap in zip(
    axes,
    [Fmag, Fn, Fo],
    ['|F|', 'Fn', 'Fo'],
    ['Reds', 'RdBu_r', 'RdBu_r'],
):
    vmax = abs(data).max()
    vmin = 0 if cmap == 'Reds' else -vmax
    cs = ax.contourf(dn, dv, data, levels=15, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cs, ax=ax, label='N', shrink=0.8)
    ax.plot(0, 0, 'r+', ms=10, mew=2)
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('dn (mm)')
    ax.set_ylabel('db (mm)')
    ax.set_aspect('equal')
    ax.set_title(title)

fig.suptitle('p=0 力场 ±2mm (原点=标准球心)', fontsize=14)
fig.tight_layout()
fig.savefig('output/p0_force_field.png', dpi=150)
print('ok')
