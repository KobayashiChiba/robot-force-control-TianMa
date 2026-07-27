"""
error_fingerprints.py — 不同误差方向的正交分解指纹

多方向偏移对比：标准 / Z柱X偏移 / Z柱Y偏移 / Y柱Z偏移
输出：output/error_fingerprints.png（3面板：Ft/Fn/Fo）
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from cylinder_def import CylinderDef
from force_profile import sphere_contact_force
from contact_frame_v2 import compute_frame

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

with open('../data/force_model.pkl', 'rb') as f:
    d = pickle.load(f)

ball = d['ball_center_500']
cy = d['cyl_contact_y']
cz = d['cyl_contact_z']
cg = d['contact_geom']
N = len(ball)
prog = np.linspace(0, 1, N)

experiments = {
    '标准': (cz, cy),
    'Z柱 X+1mm': (CylinderDef(p1=cz.p1+[1,0,0], p2=cz.p2+[1,0,0], radius=cz.radius), cy),
    'Z柱 Y+1mm': (CylinderDef(p1=cz.p1+[0,1,0], p2=cz.p2+[0,1,0], radius=cz.radius), cy),
    'Y柱 Z+1mm': (cz, CylinderDef(p1=cy.p1+[0,0,1], p2=cy.p2+[0,0,1], radius=cy.radius)),
}

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

for label, (ccz, ccy) in experiments.items():
    Ft, Fn, Fo = [], [], []
    for i, bc in enumerate(ball):
        v = ball[1] - ball[0] if i > 0 else ball[1] - ball[0]
        f, _ = sphere_contact_force(bc, v/np.linalg.norm(v), ccz, ccy)
        di = np.linalg.norm(cg.sample_pts - bc, axis=1)
        Pc = cg.sample_pts[np.argmin(di)]
        frame = compute_frame(Pc, ccy, ccz)
        t = frame.tangent
        nb = frame.normal
        ob = np.cross(t, nb)
        ob /= np.linalg.norm(ob)
        Ft.append(np.dot(f, t))
        Fn.append(np.dot(f, nb))
        Fo.append(np.dot(f, ob))
    Ft = np.array(Ft)
    Fn = np.array(Fn)
    Fo = np.array(Fo)
    ls, kw = ('--', {'lw': 2, 'alpha': 0.7}) if label == '标准' else ('-', {'lw': 1})
    axes[0].plot(prog, Ft, ls, label=label, **kw)
    axes[1].plot(prog, Fn, ls, label=label, **kw)
    axes[2].plot(prog, Fo, ls, label=label, **kw)

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

axes[0].set_ylabel('Ft (N)')
axes[1].set_ylabel('Fn (N)')
axes[2].set_xlabel('曲线进度')
axes[2].set_ylabel('Fo (N)')
fig.suptitle('不同误差方向的正交分解指纹', fontsize=12)
fig.tight_layout()
fig.savefig('output/error_fingerprints.png', dpi=150)
print('saved')
