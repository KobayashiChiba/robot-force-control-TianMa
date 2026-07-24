"""
ball_shifted.py — 平移球刀中心 Z 轴对齐接触曲线，重新计算距离
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# 1. 数据
df = pd.read_excel(os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx'))
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])

# 2. 接触曲线
_, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')
curve = resample_curve(geom_c, n_samples=10000)

# 3. 平移球刀中心：使 Z 均值对齐
ball_z_mean = pts_ball[:, 2].mean()
curve_z_mean = curve.sample_pts[:, 2].mean()
shift_z = curve_z_mean - ball_z_mean

pts_ball_shifted = pts_ball.copy()
pts_ball_shifted[:, 2] += shift_z

print(f'球刀中心 Z 均值: {ball_z_mean:.3f}')
print(f'接触曲线 Z 均值: {curve_z_mean:.3f}')
print(f'平移量: {shift_z:+.3f} mm')
print()

# 4. 重新计算距离
tree = cKDTree(curve.sample_pts)
dists_old, _ = tree.query(pts_ball, k=1)
dists_new, idxs = tree.query(pts_ball_shifted, k=1)

print(f'原始: mean={dists_old.mean():.3f} ± {dists_old.std():.3f} mm  [{dists_old.min():.2f}~{dists_old.max():.2f}]')
print(f'平移后: mean={dists_new.mean():.3f} ± {dists_new.std():.3f} mm  [{dists_new.min():.2f}~{dists_new.max():.2f}]')
print()

# 5. 每5个取1，画连线图
step = 5
ball_sub = pts_ball_shifted[::step]
dists_sub, idxs_sub = tree.query(ball_sub, k=1)
nearest = curve.sample_pts[idxs_sub]

fig, ax = plt.subplots(figsize=(10, 9))
ax.plot(curve.sample_pts[:, 1], curve.sample_pts[:, 2],
        'b-', linewidth=1.5, alpha=0.5, label='Contact curve')
ax.scatter(ball_sub[:, 1], ball_sub[:, 2],
           c='red', s=30, zorder=5, label=f'Ball center (shifted, every {step})')
ax.scatter(nearest[:, 1], nearest[:, 2],
           c='blue', s=20, zorder=5, marker='x', label='Nearest on curve')

for i in range(len(ball_sub)):
    ax.plot([ball_sub[i, 1], nearest[i, 1]],
            [ball_sub[i, 2], nearest[i, 2]],
            'gray', linewidth=0.7, alpha=0.6)

ax.set_xlabel('Y (mm)')
ax.set_ylabel('Z (mm)')
ax.set_title(f'Ball Center (Z-shifted {shift_z:+.1f}mm) → Nearest on Contact Curve\n'
             f'Distance: mean={dists_new.mean():.2f} ± {dists_new.std():.2f} mm',
             fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axis('equal')

fig.tight_layout()
out_path = os.path.join(ROOT, 'output', 'fig_ball_shifted_yz.png')
fig.savefig(out_path, dpi=150)
print(f'图已保存: {out_path}')

# 前10个点距离
print('\n平移后各点距离（每5个）:')
for i, j in enumerate(range(0, len(pts_ball), step)):
    print(f'  点{j}: {dists_sub[i]:.3f}mm')
