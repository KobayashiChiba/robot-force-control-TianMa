"""
test_fitting.py — 用实测数据验证 cylinder_fitting 模块
读 81 个实测点 → 拟合两圆柱 → 重采样交线 → 3D 对比图
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
# 1. 读取数据
# ============================================================
path = os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx')
df = pd.read_excel(path)
pts = np.column_stack([df['x'].values, df['y'].values, df['z'].values])
print(f'实测点数: {len(pts)}')

# ============================================================
# 2. 拟合
# ============================================================
params_list, geom = fit_cylinders_from_points(pts, 'Y', 'Z')

p1, p2 = params_list
print(f'\nY 圆柱 (轴∥Y):')
print(f'  轴心: ({p1.axis_point[0]:.3f}, {p1.axis_point[1]:.3f}, {p1.axis_point[2]:.3f})')
print(f'  半径: {p1.radius:.4f} mm')
print(f'  RMS : {p1.rms:.4f} mm,  max_err: {p1.max_err:.4f} mm')

print(f'\nZ 圆柱 (轴∥Z):')
print(f'  轴心: ({p2.axis_point[0]:.3f}, {p2.axis_point[1]:.3f}, {p2.axis_point[2]:.3f})')
print(f'  半径: {p2.radius:.4f} mm')
print(f'  RMS : {p2.rms:.4f} mm,  max_err: {p2.max_err:.4f} mm')

print(f'\nGeom (1位小数): c1={geom.c1}, r1={geom.r1}')
print(f'                c2={geom.c2}, r2={geom.r2}')

# ============================================================
# 3. 重采样交线曲线
# ============================================================
geom_curve = resample_curve(geom, n_samples=500)
print(f'\n重采样曲线: {geom_curve.n_samples} 点')

# ============================================================
# 4. 对比图
# ============================================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           c='red', s=25, alpha=0.85, label=f'Measured ({len(pts)} pts)')

ax.plot(geom_curve.sample_pts[:, 0],
        geom_curve.sample_pts[:, 1],
        geom_curve.sample_pts[:, 2],
        'b-', linewidth=2.0, label='Fitted curve (500 pts)')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(f'Measured Points vs Fitted Intersection Curve\n'
             f'Y-cyl: r={p1.radius:.2f}mm RMS={p1.rms:.3f}mm | '
             f'Z-cyl: r={p2.radius:.2f}mm RMS={p2.rms:.3f}mm',
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 等比例
all_pts = np.vstack([pts, geom_curve.sample_pts])
mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
ax.set_xlim(mid[0] - half, mid[0] + half)
ax.set_ylim(mid[1] - half, mid[1] + half)
ax.set_zlim(mid[2] - half, mid[2] + half)

fig.tight_layout()
out_path = os.path.join(ROOT, 'output', 'fig_fitting_test.png')
fig.savefig(out_path, dpi=150)
print(f'\n图已保存: {out_path}')
