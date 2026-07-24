"""
export_shifted.py — 导出 Z 轴修正后的球刀中心数据
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

df = pd.read_excel(os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx'))

# 计算 Z 平移量
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])

# 用接触曲线 Z 均值
from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve
_, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')
curve = resample_curve(geom_c, n_samples=10000)
shift_z = curve.sample_pts[:, 2].mean() - pts_ball[:, 2].mean()

# 加修正列
df['X_shifted'] = df['X']
df['Y_shifted'] = df['Y']
df['Z_shifted'] = df['Z'] + shift_z
df['shift_Z'] = shift_z  # 记录平移量

out = os.path.join(ROOT, 'data', '球刀中心点_修正后.xlsx')
df.to_excel(out, index=False)

# 打印摘要
print(f'Z 平移量: {shift_z:+.4f} mm')
print(f'球刀中心 Z 均值: {pts_ball[:,2].mean():.3f} → {df["Z_shifted"].mean():.3f}')
print(f'已导出: {out}')
print(f'\n前5行:')
print(df[['X','Y','Z','X_shifted','Y_shifted','Z_shifted']].head())
