"""
ball_radius_full.py — 每个球刀中心点 → 拟合接触曲线最近点 → 球刀半径
输出所有81点并保存到文件
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# 1. 数据
df = pd.read_excel(os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx'))
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])

# 2. 接触点拟合 → 密集采样接触曲线（作为真实曲线）
_, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')
curve = resample_curve(geom_c, n_samples=10000)
print('接触曲线采样: %d 点' % curve.n_samples)

# 3. 每个球刀中心点 → 找曲线上最近点
tree = cKDTree(curve.sample_pts)
dists, idxs = tree.query(pts_ball, k=1)
nearest_pts = curve.sample_pts[idxs]

# 4. 原始结果统计
print()
print('=' * 60)
print('【原始数据】球刀中心 → 拟合接触曲线最近距离')
print('=' * 60)
print('  均值:   %.4f mm' % dists.mean())
print('  标准差: %.4f mm' % dists.std())
print('  最小:   %.4f mm' % dists.min())
print('  最大:   %.4f mm' % dists.max())
print('  中位数: %.4f mm' % float(np.median(dists)))

# 5. Z轴修正：平移对齐
ball_z_mean = pts_ball[:, 2].mean()
curve_z_mean = curve.sample_pts[:, 2].mean()
shift_z = curve_z_mean - ball_z_mean

pts_ball_shifted = pts_ball.copy()
pts_ball_shifted[:, 2] += shift_z

dists_s, idxs_s = tree.query(pts_ball_shifted, k=1)
nearest_pts_s = curve.sample_pts[idxs_s]

print()
print('=' * 60)
print('【Z轴修正后】球刀中心 → 拟合接触曲线最近距离')
print('  Z偏移 = %+.3f mm' % shift_z)
print('=' * 60)
print('  均值:   %.4f mm' % dists_s.mean())
print('  标准差: %.4f mm' % dists_s.std())
print('  最小:   %.4f mm' % dists_s.min())
print('  最大:   %.4f mm' % dists_s.max())
print('  中位数: %.4f mm' % float(np.median(dists_s)))
print()
print('→ 球刀半径 ≈ %.2f ± %.2f mm' % (dists_s.mean(), dists_s.std()))

# 6. 打印全部81个点（修正后）
print()
print('=' * 60)
print('全部81点距离（Z修正后）:')
print('%-6s %-10s %s' % ('点号', '距离mm', '最近曲线点索引'))
print('-' * 40)
for i in range(len(pts_ball)):
    print('%-6d %-10.4f %d' % (i, dists_s[i], idxs_s[i]))

# 7. 保存到文件
out_txt = os.path.join(ROOT, 'output', 'ball_radius_results.txt')
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('球刀半径计算结果\n')
    f.write('=' * 60 + '\n')
    f.write('方法: 球刀中心 → 拟合接触曲线(10000点) 最近距离\n')
    f.write('\n原始数据统计:\n')
    f.write('  均值=%.4f 标准差=%.4f 范围=[%.4f, %.4f]\n' % (
        dists.mean(), dists.std(), dists.min(), dists.max()))
    f.write('\nZ轴偏移=%+.3fmm 修正后统计:\n' % shift_z)
    f.write('  均值=%.4f 标准差=%.4f 范围=[%.4f, %.4f]\n' % (
        dists_s.mean(), dists_s.std(), dists_s.min(), dists_s.max()))
    f.write('  球刀半径 ≈ %.2f ± %.2f mm\n' % (dists_s.mean(), dists_s.std()))
    f.write('\n%-6s %-10s %-40s %s\n' % ('点号', '距离mm', '球刀中心(shifted XYZ)', '最近曲线点(XYZ)'))
    f.write('-' * 100 + '\n')
    for i in range(len(pts_ball)):
        b = pts_ball_shifted[i]
        n = nearest_pts_s[i]
        f.write('%-6d %-10.4f (%.2f,%.2f,%.2f)%s(%.2f,%.2f,%.2f)\n' % (
            i, dists_s[i], b[0], b[1], b[2],
            ' ' * (42 - len('(%.2f,%.2f,%.2f)' % (b[0], b[1], b[2]))),
            n[0], n[1], n[2]))

print()
print('结果已保存: %s' % out_txt)
