"""
compute_normals.py — 批量计算法向量并保存
方法：球刀中心点 → 接触曲线上距离≈R的点分两簇 → 中点方向
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
from scipy.spatial import cKDTree

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

with open(os.path.join(ROOT, 'data', 'standard_curves.pkl'), 'rb') as f:
    data = pickle.load(f)
ball_curve = data['ball_center'].sample_pts
contact_curve = data['contact'].sample_pts
R = data['ball_radius']

N = len(ball_curve)
normals = np.zeros((N, 3))
midpoints = np.zeros((N, 3))
centroids0 = np.zeros((N, 3))
centroids1 = np.zeros((N, 3))
stats = []

for i in range(N):
    P = ball_curve[i]
    dists = np.linalg.norm(contact_curve - P, axis=1)
    
    # 找距离 ≈ R ± 0.15mm 的点
    mask = np.abs(dists - R) < 0.15
    near_idx = np.where(mask)[0]
    
    if len(near_idx) < 4:
        # 容差不够，放宽
        mask = np.abs(dists - R) < 0.25
        near_idx = np.where(mask)[0]
    
    if len(near_idx) < 4:
        # 还是不够，用top-2最近（退化情况）
        dists_k, idxs_k = cKDTree(contact_curve).query(P, k=2)
        c0 = contact_curve[idxs_k[0]]
        c1 = contact_curve[idxs_k[1]]
    else:
        # 按索引连续性分两簇（接触曲线是闭合的，首尾相连）
        gaps = np.diff(near_idx)
        split = np.argmax(gaps) + 1  # 最大间隙处分界
        
        cluster0_idx = near_idx[:split]
        cluster1_idx = near_idx[split:]
        
        # 如果分界在结尾（闭合环），重新检查
        if len(cluster1_idx) == 0 and len(cluster0_idx) > 4:
            # 尝试在中间找第二大间隙
            if len(gaps) >= 2:
                sorted_gaps = np.argsort(gaps)[::-1]
                split = sorted_gaps[1] + 1
                cluster0_idx = near_idx[:split]
                cluster1_idx = near_idx[split:]
        
        if len(cluster1_idx) == 0:
            # 退化：所有点连在一起，用top-2
            dists_k, idxs_k = cKDTree(contact_curve).query(P, k=2)
            c0 = contact_curve[idxs_k[0]]
            c1 = contact_curve[idxs_k[1]]
        else:
            c0 = contact_curve[cluster0_idx].mean(axis=0)
            c1 = contact_curve[cluster1_idx].mean(axis=0)
    
    centroids0[i] = c0
    centroids1[i] = c1
    mid = (c0 + c1) / 2
    midpoints[i] = mid
    normal = mid - P
    n_len = np.linalg.norm(normal)
    normals[i] = normal / n_len
    
    stats.append({
        'idx': i,
        'ball_pt': P,
        'c0': c0, 'c1': c1,
        'mid': mid,
        'normal': normal / n_len,
        'mid_dist': n_len,
        'angle_between': np.degrees(np.arccos(
            np.dot(c0-P, c1-P) / (np.linalg.norm(c0-P) * np.linalg.norm(c1-P))
        )),
        'n_near': len(near_idx),
    })

# 保存
output = {
    'ball_curve': ball_curve,
    'contact_curve': contact_curve,
    'normals': normals,           # 单位法向量 (N,3)
    'midpoints': midpoints,       # 中点坐标 (N,3)
    'centroids0': centroids0,     # 簇1中心 (N,3)
    'centroids1': centroids1,     # 簇2中心 (N,3)
    'ball_radius': R,
    'stats': stats,
}

npz_path = os.path.join(ROOT, 'data', 'normals.npz')
np.savez(npz_path,
         ball_curve=ball_curve,
         contact_curve=contact_curve,
         normals=normals,
         midpoints=midpoints,
         ball_radius=R)

pkl_path = os.path.join(ROOT, 'data', 'normals.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(output, f)

# 统计
mid_dists = [s['mid_dist'] for s in stats]
angles = [s['angle_between'] for s in stats]
print('%d 点全部计算完毕' % N)
print('中点距离球心: mean=%.4f ± %.4f  [%.4f, %.4f]' % (
    np.mean(mid_dists), np.std(mid_dists), min(mid_dists), max(mid_dists)))
print('两簇夹角:     mean=%.1f° ± %.1f°  [%.1f°, %.1f°]' % (
    np.mean(angles), np.std(angles), min(angles), max(angles)))

# 采样输出
print('\n采样法向量:')
for i in range(0, N, 100):
    s = stats[i]
    print('  点%3d: n=(%+.4f, %+.4f, %+.4f)  |mid|=%.3f  ang=%.1f°  near=%d' % (
        i, normals[i,0], normals[i,1], normals[i,2],
        s['mid_dist'], s['angle_between'], s['n_near']))

print('\n已保存: normals.npz / normals.pkl')
