"""
explore_normal_v2.py — 对球刀中心点，在接触曲线上找距离≈R的点
看它们是否分成两个空间簇（两个真实交点）
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
from scipy.spatial import cKDTree, distance

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

with open(os.path.join(ROOT, 'data', 'standard_curves.pkl'), 'rb') as f:
    data = pickle.load(f)
ball_curve = data['ball_center'].sample_pts
contact_curve = data['contact'].sample_pts
R = data['ball_radius']

# 取几个样本点研究
for sample_name, idx in [('起点', 0), ('1/4处', 125), ('中间', 250), ('3/4处', 375)]:
    P = ball_curve[idx]
    dists = np.linalg.norm(contact_curve - P, axis=1)
    
    # 找距离 ≈ R ± 0.2mm 的点
    mask = np.abs(dists - R) < 0.2
    near_idx = np.where(mask)[0]
    near_pts = contact_curve[near_idx]
    
    print('=' * 60)
    print('样本点 %s (idx=%d) P=(%.1f, %.1f, %.1f)' % (sample_name, idx, P[0], P[1], P[2]))
    print('  距离≈R(±0.2): %d 个点' % len(near_idx))
    
    if len(near_idx) >= 2:
        # 尝试用k-means分成2簇
        from sklearn.cluster import KMeans
        if len(near_idx) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(near_pts)
            
            c0 = near_pts[labels == 0]
            c1 = near_pts[labels == 1]
            centroid0 = c0.mean(axis=0)
            centroid1 = c1.mean(axis=0)
            
            print('  簇1: %d 点 中心=(%.2f, %.2f, %.2f)  dist_to_P=%.3f' % (
                len(c0), centroid0[0], centroid0[1], centroid0[2],
                np.linalg.norm(centroid0 - P)))
            print('  簇2: %d 点 中心=(%.2f, %.2f, %.2f)  dist_to_P=%.3f' % (
                len(c1), centroid1[0], centroid1[1], centroid1[2],
                np.linalg.norm(centroid1 - P)))
            
            mid = (centroid0 + centroid1) / 2
            normal = mid - P
            normal_u = normal / np.linalg.norm(normal)
            print('  中点: (%.2f, %.2f, %.2f)  |mid-P|=%.3f' % (mid[0], mid[1], mid[2], np.linalg.norm(normal)))
            print('  法向量(单位): (%.4f, %.4f, %.4f)' % (normal_u[0], normal_u[1], normal_u[2]))
            
            # 也输出两个簇中心的方向
            dir0 = centroid0 - P
            dir1 = centroid1 - P
            cos_angle = np.dot(dir0, dir1) / (np.linalg.norm(dir0) * np.linalg.norm(dir1))
            print('  两簇中心夹角: %.1f°' % np.degrees(np.arccos(cos_angle)))
    print()

# 最快：取几个点，打印 near_idx 的索引分布，看是否连续
P = ball_curve[0]
dists = np.linalg.norm(contact_curve - P, axis=1)
mask = np.abs(dists - R) < 0.1
near_idx = np.where(mask)[0]
print('点0 附近点索引（距离≈R±0.1）:')
print('  共%d个: %s' % (len(near_idx), near_idx[:30].tolist()))
# 找间隙——索引跳变处就是两个簇的分界
gaps = np.diff(near_idx)
big_gaps = np.where(gaps > 10)[0]
print('  索引跳变>10 的位置: %s' % ([near_idx[i] for i in big_gaps] if len(big_gaps) > 0 else '无'))
print('  跳变值: %s' % gaps[big_gaps].tolist() if len(big_gaps) > 0 else '无')
