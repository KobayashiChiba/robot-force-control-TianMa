"""
error_experiment.py — 轴线误差对接触力的影响

单次偏移实验：标准 vs Z柱X偏移 vs Z柱Y偏移
输出：output/error_experiment.png
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from cylinder_def import CylinderDef
from force_profile import sphere_contact_force

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

with open('../data/force_model.pkl', 'rb') as f:
    d = pickle.load(f)

ball = d['ball_center_500']
cy = d['cyl_contact_y']
cz = d['cyl_contact_z']

# 两柱面各自偏移
experiments = {
    '标准': (cz, cy),
    'Z柱 X+1mm': (CylinderDef(p1=cz.p1+[1,0,0], p2=cz.p2+[1,0,0], radius=cz.radius), cy),
    'Z柱 Y+1mm': (CylinderDef(p1=cz.p1+[0,1,0], p2=cz.p2+[0,1,0], radius=cz.radius), cy),
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
progress = np.linspace(0, 1, len(ball))

from contact_frame_v2 import compute_frame

for label, (ccz, ccy) in experiments.items():
    fs = []
    for bc in ball:
        v = ball[1] - ball[0]
        f, _ = sphere_contact_force(bc, v/np.linalg.norm(v), ccz, ccy)
        fs.append(np.linalg.norm(f))
    fs = np.array(fs)
    ax1.plot(progress, fs, lw=1, label=f'{label} (m={fs.mean():.0f}N)')

    # 正交分解
    Fn_vals = []
    for i, bc in enumerate(ball):
        v = ball[1] - ball[0]
        f, _ = sphere_contact_force(bc, v/np.linalg.norm(v), ccz, ccy)
        dists = np.linalg.norm(d['contact_geom'].sample_pts - bc, axis=1)
        Pc = d['contact_geom'].sample_pts[np.argmin(dists)]
        frame = compute_frame(Pc, ccy, ccz)
        Fn_vals.append(np.dot(f, frame.normal))
    Fn_vals = np.array(Fn_vals)
    ax2.plot(progress, Fn_vals, lw=1, label=f'{label} (m={Fn_vals.mean():.0f}N)')

ax1.set_ylabel('|F| (N)')
ax1.axhline(8, color='gray', ls='--')
ax1.legend()
ax1.grid(alpha=0.3)
ax2.set_ylabel('Fn 法向 (N)')
ax2.axhline(-8, color='gray', ls='--')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlabel('曲线进度')
fig.suptitle('轴线误差对接触力的影响', fontsize=12)
fig.tight_layout()
fig.savefig('output/error_experiment.png', dpi=150)
print('saved output/error_experiment.png')
