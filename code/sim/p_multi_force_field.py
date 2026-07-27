"""
p_multi_force_field.py — 多点力场热力图（含截面叠加）

沿曲线取多个进度位置（默认 p=0.05, 0.1, 0.15, 0.2, 0.25），
每个位置画三面板热力图（|F| / Fn / Fo），并叠加截面线 + 球刀圆。

===== 设计要义 =====

[1] 热力图坐标系
  - 原点(0,0) = 该位置的标准球刀中心
  - 横轴 dn = 法向力方向偏移，纵轴 db = 复法向力方向偏移
  - 范围 ±2mm，0=白色，红正蓝负
  - |F| 用 Reds（非负），Fn/Fo 用 RdBu_r（有正负）

[2] 截面图叠加
  - 截面图是一张独立坐标系：中心(0,0) = 该位置的接触点
  - 截面线（Z柱青色 + Y柱绿色）+ 球刀圆（白色，R=4.2mm）+ 接触点（黑点）
  - 直接叠加在同一 axes 上，不做坐标平移

[3] 为什么两个坐标系可以直接叠加
  - 球刀中心偏移和接触点偏移是同步的：球刀在法平面内偏多少，接触点也偏多少
  - 热力图原点（标准球心）和截面图原点（对应标准接触点）在偏移坐标系里自然对齐
  - 因此两个独立坐标系叠加后，截面区段和热力等高线有直观的几何对应关系

[4] 用途
  - 观察不同进度位置，球刀中心沿 n-b 方向偏移 ±2mm 时的力变化
  - 结合截面几何，理解哪些偏移方向会增大/减小接触力
  - 为阻抗力控算法提供力场特征：不同位置的力场形状不同（力曲线指纹）

用法: python p_multi_force_field.py
输出: output/p_multi_force_field.png
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from force_profile import sphere_contact_force
from contact_frame_v2 import compute_frame
from section_with_ball import sample_sections

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 参数 ──────────────────────────────────────────
PROGRESSES = [0.05, 0.1, 0.15, 0.2, 0.25]  # 曲线进度位置
R = 2       # 偏移范围 (mm)
N_GRID = 41  # 网格分辨率
BALL_R = 4.2  # 球刀半径 (mm)
SECTION_RANGE = 25  # 截面采样范围 (mm)

# ── 加载数据 ──────────────────────────────────────
with open('../data/force_model.pkl', 'rb') as f:
    data = pickle.load(f)

ball = data['ball_center_500']
cy = data['cyl_contact_y']
cz = data['cyl_contact_z']
cg = data['contact_geom'].sample_pts
N_PTS = len(ball)

# 截面采样范围（以接触点为中心）
cmin = cg.min(0)
cmax = cg.max(0)
ctr = (cmin + cmax) / 2
z_range = np.array([ctr[2] - SECTION_RANGE, ctr[2] + SECTION_RANGE])
y_range = np.array([ctr[1] - SECTION_RANGE, ctr[1] + SECTION_RANGE])

# ── 网格 ──────────────────────────────────────────
dn = np.linspace(-R, R, N_GRID)
db = np.linspace(-R, R, N_GRID)

# ── 画图 ──────────────────────────────────────────
fig, all_axes = plt.subplots(3, len(PROGRESSES), figsize=(5.5*len(PROGRESSES), 16))

for col, p in enumerate(PROGRESSES):
    axes_col = all_axes[:, col]

    # 取该进度位置的球心 + 接触点
    i = int(p * N_PTS)
    bc0 = ball[i]
    idx = np.argmin(np.linalg.norm(cg - bc0, axis=1))
    Pc = cg[idx]

    # 计算该位置的局部标架
    frame = compute_frame(Pc, cy, cz)
    n_vec = frame.normal
    b_vec = np.cross(frame.tangent, n_vec)
    b_vec /= np.linalg.norm(b_vec)

    # ── 热力图：球面采样算力 ──
    Fmag = np.zeros((N_GRID, N_GRID))
    Fn = np.zeros_like(Fmag)
    Fo = np.zeros_like(Fmag)
    for ii, dni in enumerate(dn):
        for jj, dbj in enumerate(db):
            f, _ = sphere_contact_force(
                bc0 + dni*n_vec + dbj*b_vec,
                np.array([0, -1, 0]), cz, cy)
            Fmag[jj, ii] = np.linalg.norm(f)
            Fn[jj, ii] = np.dot(f, n_vec)
            Fo[jj, ii] = np.dot(f, b_vec)

    # ── 截面数据 ──
    sec = sample_sections(Pc, frame.tangent, n_vec, b_vec, cz, cy,
                          z_range, y_range)

    # ── 三面板 ──
    for row, ax, data_3d, ttl, cmap in zip(
        [0, 1, 2], axes_col,
        [Fmag, Fn, Fo],
        ['|F|', 'Fn', 'Fo'],
        ['Reds', 'RdBu_r', 'RdBu_r'],
    ):
        # 等高线
        vmax = abs(data_3d).max()
        vmin = 0 if cmap == 'Reds' else -vmax
        cs = ax.contourf(dn, db, data_3d, levels=15, cmap=cmap,
                         vmin=vmin, vmax=vmax)
        plt.colorbar(cs, ax=ax, shrink=0.8)

        # 原点标记（标准球心）= 红+
        ax.plot(0, 0, 'r+', ms=8, mew=2)
        ax.axhline(0, color='gray', lw=0.3)
        ax.axvline(0, color='gray', lw=0.3)
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect('equal')

        if row == 0:
            ax.set_title(f'p = {p:.2f}')
        if col == 0:
            ax.set_ylabel(ttl)

        # ── 截面叠加（独立坐标系，接触点=原点）──
        def plot_uv(uvdata, color):
            if uvdata is None:
                return
            items = uvdata if isinstance(uvdata, list) else [uvdata]
            for L in items:
                ax.plot(L[:, 0], L[:, 1], color=color, lw=0.8)
        plot_uv(sec['z_uv'], 'cyan')
        plot_uv(sec['y_uv'], 'lime')
        ax.add_patch(Circle((0, 0), BALL_R, fill=False, color='white', lw=1))
        ax.plot(0, 0, 'ko', ms=4)

fig.suptitle('多点力场热力图 (原点=球心, 截面中心=接触点)', fontsize=16)
fig.tight_layout()
fig.savefig('output/p_multi_force_field.png', dpi=150)
print(f'已保存 output/p_multi_force_field.png')
