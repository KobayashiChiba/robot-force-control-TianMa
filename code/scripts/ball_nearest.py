"""
ball_nearest.py — 球刀中心 → 接触曲线最近点连线（X 正方向视角）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# 1. 数据
df = pd.read_excel(os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx'))
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])

# 2. 接触曲线（密集采样）
_, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')
curve = resample_curve(geom_c, n_samples=10000)

# 3. 球刀中心每5个取1个
step = 5
ball_sub = pts_ball[::step]
print(f'球刀中心: {len(pts_ball)} → 每{step}取1 → {len(ball_sub)}个')

# 4. 找最近点
tree = cKDTree(curve.sample_pts)
dists, idxs = tree.query(ball_sub, k=1)
nearest = curve.sample_pts[idxs]

print(f'\n各点距离:')
for i in range(len(ball_sub)):
    print(f'  点{i*step}: dist={dists[i]:.3f}mm | 球心=({ball_sub[i,0]:.1f},{ball_sub[i,1]:.1f},{ball_sub[i,2]:.1f}) → 最近=({nearest[i,0]:.1f},{nearest[i,1]:.1f},{nearest[i,2]:.1f})')

# 5. YZ 投影图（X 正方向视角）
fig, ax = plt.subplots(figsize=(10, 9))

# 接触曲线
ax.plot(curve.sample_pts[:, 1], curve.sample_pts[:, 2],
        'b-', linewidth=1.5, alpha=0.5, label='Contact curve')

# 球刀中心
ax.scatter(ball_sub[:, 1], ball_sub[:, 2],
           c='red', s=30, zorder=5, label=f'Ball center (every {step})')

# 最近点
ax.scatter(nearest[:, 1], nearest[:, 2],
           c='blue', s=20, zorder=5, marker='x', label='Nearest on curve')

# 连线
for i in range(len(ball_sub)):
    ax.plot([ball_sub[i, 1], nearest[i, 1]],
            [ball_sub[i, 2], nearest[i, 2]],
            'gray', linewidth=0.7, alpha=0.6)

ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')
ax.set_title(f'Ball Center → Nearest Point on Contact Curve (X+ view)\n'
             f'Distance: mean={dists.mean():.2f} ± {dists.std():.2f} mm',
             fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axis('equal')

fig.tight_layout()
out_path = os.path.join(ROOT, 'output', 'fig_ball_nearest_yz.png')
fig.savefig(out_path, dpi=150)
print(f'\n图已保存: {out_path}')
