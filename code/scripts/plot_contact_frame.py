"""
plot_contact_frame.py — 在标准接触曲线上采样20个点，画出 t/n/rz 三个向量
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contact_frame import compute_frame

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# 加载标准接触曲线
with open(os.path.join(ROOT, 'data', 'standard_curves.pkl'), 'rb') as f:
    data = pickle.load(f)
contact = data['contact'].sample_pts  # (500, 3)

# 均匀采样 20 个点
N_sample = 20
indices = np.linspace(0, len(contact)-1, N_sample, dtype=int)
pts = contact[indices]

# 圆柱参数
cy = np.array([51.497, 65.151, -39.700])
cz = np.array([72.503, 65.000, -39.763])
ry_r = 9.0
rz_r = 18.0

# 计算标架
frames = [compute_frame(p, cy, cz, ry_r, rz_r) for p in pts]

# ============================================================
# 画图
# ============================================================
fig = plt.figure(figsize=(16, 13))
ax = fig.add_subplot(111, projection='3d')

# 接触曲线
ax.plot(contact[:, 0], contact[:, 1], contact[:, 2],
        'gray', linewidth=0.8, alpha=0.4, label='Contact curve')

# 采样点
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           c='black', s=40, zorder=5, label='Sample points (%d)' % N_sample)

# 箭头缩放
scale = 1.0
colors = {'t': 'red', 'n': 'blue', 'rz': 'green'}

for i, (p, f) in enumerate(zip(pts, frames)):
    for vec_name, vec, color in [
        ('t', f.tangent, 'red'),
        ('n', f.normal, 'blue'),
        ('rz', f.radial_z, 'green'),
    ]:
        label = vec_name if i == 0 else None  # 只第一个点加图例
        ax.quiver(p[0], p[1], p[2],
                  vec[0]*scale, vec[1]*scale, vec[2]*scale,
                  color=color, linewidth=1.5, alpha=0.85, label=label,
                  arrow_length_ratio=0.15)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Contact Curve — Local Frame (t/n/rz) × 20 points\n'
             'red=tangent  blue=normal  green=radial_z  scale=%.0f×' % scale,
             fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)

# 等比例
all_data = np.vstack([contact, pts])
mid = (all_data.max(axis=0) + all_data.min(axis=0)) / 2
half = (all_data.max(axis=0) - all_data.min(axis=0)).max() / 2 * 1.1
ax.set_xlim(mid[0] - half, mid[0] + half)
ax.set_ylim(mid[1] - half, mid[1] + half)
ax.set_zlim(mid[2] - half, mid[2] + half)

fig.tight_layout()
out_path = os.path.join(ROOT, 'output', 'contact_frame_20pts.png')
fig.savefig(out_path, dpi=150)
print('已保存: %s' % out_path)
print('采样点数: %d' % N_sample)
