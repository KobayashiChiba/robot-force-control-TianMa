"""
ball_radius.py — 计算球刀中心到拟合接触曲线的最短距离
验证是否为恒定值（球刀半径）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# 1. 读取数据
df = pd.read_excel(os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx'))
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])

# 2. 接触点拟合 → 密集采样接触曲线
_, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')
curve = resample_curve(geom_c, n_samples=10000)  # 密集采样
print(f'接触曲线: {curve.n_samples} 点')

# 3. 对每个球刀中心点，找曲线上最近点
tree = cKDTree(curve.sample_pts)
dists, idxs = tree.query(pts_ball, k=1)
nearest_pts = curve.sample_pts[idxs]

# 4. 统计
print(f'\n球刀中心到接触曲线距离:')
print(f'  均值:   {dists.mean():.4f} mm')
print(f'  标准差: {dists.std():.4f} mm')
print(f'  最小:   {dists.min():.4f} mm')
print(f'  最大:   {dists.max():.4f} mm')
print(f'  中位数: {np.median(dists):.4f} mm')

# 5. 对比：球刀中心到实测接触点的距离
dist_direct = np.linalg.norm(pts_ball - pts_contact, axis=1)
print(f'\n球刀中心到实测接触点距离（按采样序号对应）:')
print(f'  均值:   {dist_direct.mean():.4f} mm')
print(f'  标准差: {dist_direct.std():.4f} mm')

# 6. 输出各点距离
print(f'\n前10个点到拟合曲线的距离:')
for i in range(10):
    print(f'  点{i}: {dists[i]:.4f} mm (最近曲线点索引={idxs[i]})')
