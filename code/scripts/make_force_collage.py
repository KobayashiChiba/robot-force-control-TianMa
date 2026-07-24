"""
make_force_collage.py — 拼合力分解对比图
前4张 → 2×2（两方案对比）
后2张 → 1×2（力分解 + 运动趋势）
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'
CACHE = r'C:\Users\KCserver\AppData\Local\hermes\image_cache'

# 前4张：两种方案对比
files_4 = [
    'img_4e0797b28e27.jpg',   # 方案1 视角1
    'img_c3d81dae6342.jpg',   # 方案1 视角2
    'img_c20ce9b84491.jpg',   # 方案2 视角1
    'img_13a0047aa58d.jpg',   # 方案2 视角2
]

# 后2张：力分解 + 运动趋势
files_2 = [
    'img_95e1851b7852.jpg',   # 力分解
    'img_6ff557d6991f.jpg',   # 运动趋势
]

# --- 2×2 拼合 ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
titles_4 = ['Scheme 1: Orthogonal (Frenet) — View 1',
            'Scheme 1: Orthogonal (Frenet) — View 2',
            'Scheme 2: Normal+Tangent+Vertical — View 1',
            'Scheme 2: Normal+Tangent+Vertical — View 2']
for ax, fname, title in zip(axes.flat, files_4, titles_4):
    img = mpimg.imread(os.path.join(CACHE, fname))
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
fig.suptitle('Force Decomposition Scheme Comparison', fontsize=14, y=0.98)
fig.tight_layout()
out1 = os.path.join(ROOT, 'output', 'fig_schemes_compare.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f'图1已保存: {out1}')

# --- 1×2 拼合 ---
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
titles_2 = ['Force Decomposition (Scheme 2 basis)',
            'Decomposed Forces + Motion Trends + Expected Force']
for ax, fname, title in zip(axes2, files_2, titles_2):
    img = mpimg.imread(os.path.join(CACHE, fname))
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
fig2.suptitle('Force Analysis — Decomposition & Motion Trends', fontsize=14, y=0.98)
fig2.tight_layout()
out2 = os.path.join(ROOT, 'output', 'fig_force_analysis.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f'图2已保存: {out2}')
