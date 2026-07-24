"""
explore_normal.py — 球刀中心→接触曲线两个交点→中点→法向量探索
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
from scipy.spatial import cKDTree

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# 加载标准曲线
with open(os.path.join(ROOT, 'data', 'standard_curves.pkl'), 'rb') as f:
    data = pickle.load(f)

ball_curve = data['ball_center'].sample_pts   # (500, 3)
contact_curve = data['contact'].sample_pts    # (500, 3)
R = data['ball_radius']                       # 4.0

print('球刀中心曲线: %d 点' % len(ball_curve))
print('接触曲线:     %d 点' % len(contact_curve))
print('球刀半径 R = %.1f mm' % R)

# KDTree 用于快速找最近点
tree = cKDTree(contact_curve)

# 取一个样本点（比如第100个）研究
idx = 100
P = ball_curve[idx]
print('\n样本点 %d: P_ball = (%.2f, %.2f, %.2f)' % (idx, P[0], P[1], P[2]))

# 找接触曲线上所有距离 ≈ R 的点（容差 ±0.1mm）
dists_all = np.linalg.norm(contact_curve - P, axis=1)
near_mask = np.abs(dists_all - R) < 0.5  # 先宽松一点
near_idxs = np.where(near_mask)[0]
near_dists = dists_all[near_idxs]
near_pts = contact_curve[near_idxs]

print('距离≈R (±0.5mm) 的点: %d 个' % len(near_idxs))
if len(near_idxs) > 0:
    print('  距离范围: [%.3f, %.3f]' % (near_dists.min(), near_dists.max()))

# 找最近的 2 个点
k = min(5, len(contact_curve))
dists_k, idxs_k = tree.query(P, k=k)
print('\n最近 %d 个接触点:' % k)
for i in range(k):
    pt = contact_curve[idxs_k[i]]
    print('  #%d idx=%d dist=%.4fmm  (%.2f, %.2f, %.2f)' % (
        i, idxs_k[i], dists_k[i], pt[0], pt[1], pt[2]))

# --- 对所有 500 个点批量算 ---
print('\n' + '=' * 60)
print('批量：对每个球刀中心点，找接触曲线上最近的2个点')

# 对每个球刀中心点找 top-2 最近
dists_top2, idxs_top2 = tree.query(ball_curve, k=2)

# 最近点距离统计
d1 = dists_top2[:, 0]
d2 = dists_top2[:, 1]
print('\n最近点距离:  mean=%.4f ± %.4f  [%.4f, %.4f]' % (
    d1.mean(), d1.std(), d1.min(), d1.max()))
print('次近点距离: mean=%.4f ± %.4f  [%.4f, %.4f]' % (
    d2.mean(), d2.std(), d2.min(), d2.max()))

# 两个交点求中点
nearest_pt = contact_curve[idxs_top2[:, 0]]
second_pt = contact_curve[idxs_top2[:, 1]]
mid_pts = (nearest_pt + second_pt) / 2

# 法向量：球刀中心 → 中点
normals = mid_pts - ball_curve
normal_len = np.linalg.norm(normals, axis=1)
unit_normals = normals / normal_len[:, None]

print('\n中点距离球心: mean=%.4f ± %.4f  [%.4f, %.4f]' % (
    normal_len.mean(), normal_len.std(), normal_len.min(), normal_len.max()))

# 打印几个样本的法向量
print('\n样本法向量（球刀中心 → 中点）:')
for i in range(0, 500, 100):
    print('  点%3d: n=(%+.4f, %+.4f, %+.4f)  |n|=%.4f' % (
        i, unit_normals[i, 0], unit_normals[i, 1], unit_normals[i, 2], normal_len[i]))

# 也直接用最近点方向作为法向量对比
nearest_normals = nearest_pt - ball_curve
nearest_normal_len = np.linalg.norm(nearest_normals, axis=1)
print('\n对比：直接用最近点方向（球刀中心→最近接触点）:')
print('  距离: mean=%.4f ± %.4f  [%.4f, %.4f]' % (
    nearest_normal_len.mean(), nearest_normal_len.std(),
    nearest_normal_len.min(), nearest_normal_len.max()))
for i in range(0, 500, 100):
    u = nearest_normals[i] / nearest_normal_len[i]
    print('  点%3d: n=(%+.4f, %+.4f, %+.4f)' % (i, u[0], u[1], u[2]))
