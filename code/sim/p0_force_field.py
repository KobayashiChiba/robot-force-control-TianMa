"""
p0_force_field.py — p=0 接触点力场图

力场：球心 = 接触点 + dn*n + db*b
原点 (0,0) = 接触点
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

# 网格
dn = np.linspace(-6, 1, 71)
db = np.linspace(-3, 3, 61)
Fmag = np.zeros((len(db), len(dn)))
Fn   = np.zeros_like(Fmag)
Fo   = np.zeros_like(Fmag)

for i, dni in enumerate(dn):
    for j, dbj in enumerate(db):
        bc = Pc + dni * n_vec + dbj * b_vec
        f, _ = sphere_contact_force(bc, np.array([0, -1, 0]), cz, cy)
        Fmag[j, i] = np.linalg.norm(f)
        Fn[j, i]   = np.dot(f, n_vec)
        Fo[j, i]   = np.dot(f, b_vec)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, data, title in zip(axes, [Fmag, Fn, Fo], ['|F|', 'Fn', 'Fo']):
    cs = ax.contourf(dn, db, data, levels=15, cmap='RdBu_r')
    plt.colorbar(cs, ax=ax, label='N', shrink=0.8)
    ax.plot(0, 0, 'ko', ms=6)
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('n (mm)')
    ax.set_ylabel('b (mm)')
    ax.set_aspect('equal')
    ax.set_title(title)

fig.suptitle('p=0 力场 (原点 = 接触点)', fontsize=14)
fig.tight_layout()
fig.savefig('output/p0_force_field.png', dpi=150)
print('ok')
