"""
test_fitting_ball.py — 用球刀中心坐标（大写 X/Y/Z）验证 cylinder_fitting
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# ============================================================
# 1. 读取 — 球刀中心（大写 X/Y/Z）
# ============================================================
path = os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx')
df = pd.read_excel(path)
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])
print(f'实测点数: {len(pts_ball)}')

# ============================================================
# 2. 拟合两套
# ============================================================
print('\n=== 球刀中心坐标 ===')
pb, geom_b = fit_cylinders_from_points(pts_ball, 'Y', 'Z')

print('\n=== 接触点坐标 ===')
pc, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')

b1, b2 = pb
c1, c2 = pc
print(f'\nY圆柱 半径: 球刀中心={b1.radius:.2f}mm  接触点={c1.radius:.2f}mm  (球刀偏置={b1.radius - c1.radius:.2f}mm)')
print(f'Z圆柱 半径: 球刀中心={b2.radius:.2f}mm  接触点={c2.radius:.2f}mm  (球刀偏置={b2.radius - c2.radius:.2f}mm)')
print(f'\nY圆柱 RMS:  球刀中心={b1.rms:.4f}mm  接触点={c1.rms:.4f}mm')
print(f'Z圆柱 RMS:  球刀中心={b2.rms:.4f}mm  接触点={c2.rms:.4f}mm')

# ============================================================
# 3. 重采样
# ============================================================
curve_b = resample_curve(geom_b, n_samples=500)
curve_c = resample_curve(geom_c, n_samples=500)

# ============================================================
# 4. 图
# ============================================================
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts_ball[:, 0], pts_ball[:, 1], pts_ball[:, 2],
           c='dodgerblue', s=20, alpha=0.7, label='Ball center (81 pts)')
ax.scatter(pts_contact[:, 0], pts_contact[:, 1], pts_contact[:, 2],
           c='red', s=20, alpha=0.7, label='Contact (81 pts)')

ax.plot(curve_b.sample_pts[:, 0], curve_b.sample_pts[:, 1], curve_b.sample_pts[:, 2],
        'b-', linewidth=2.0, label='Fitted curve (ball)')
ax.plot(curve_c.sample_pts[:, 0], curve_c.sample_pts[:, 1], curve_c.sample_pts[:, 2],
        'r--', linewidth=2.0, label='Fitted curve (contact)')

ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
ax.set_title(f'Ball Center vs Contact — Fitting Comparison\n'
             f'Y: r_ball={b1.radius:.2f} r_ct={c1.radius:.2f} | '
             f'Z: r_ball={b2.radius:.2f} r_ct={c2.radius:.2f}',
             fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 等比例
all_pts = np.vstack([pts_ball, pts_contact, curve_b.sample_pts, curve_c.sample_pts])
mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
ax.set_xlim(mid[0] - half, mid[0] + half)
ax.set_ylim(mid[1] - half, mid[1] + half)
ax.set_zlim(mid[2] - half, mid[2] + half)

fig.tight_layout()
out_path = os.path.join(ROOT, 'output', 'fig_ball_vs_contact.png')
fig.savefig(out_path, dpi=150)
print(f'\n图已保存: {out_path}')
