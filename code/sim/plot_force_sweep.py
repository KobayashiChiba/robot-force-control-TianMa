"""
plot_force_sweep.py — force_sweep_data.npz 散点图可视化
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
_sdir = os.path.dirname(os.path.abspath(__file__))

d = np.load(os.path.join(_sdir, 'output', 'force_sweep_data.npz'))
Fn, Fo = d['Fn'], d['Fo']
dn, db = d['dn'], d['db']
pos = d['pos_idx']

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ── 左上: Fn vs dn，颜色=db ──
ax = axes[0, 0]
sc = ax.scatter(dn, Fn, c=db, s=2, alpha=0.5, cmap='coolwarm', vmin=-2, vmax=2)
plt.colorbar(sc, ax=ax, label='db (mm)')
ax.axhline(-8.0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('dn (mm)'); ax.set_ylabel('Fn (N)')
ax.set_title('Fn vs dn')
ax.grid(alpha=0.3)

# ── 右上: Fo vs db，颜色=dn ──
ax = axes[0, 1]
sc = ax.scatter(db, Fo, c=dn, s=2, alpha=0.5, cmap='viridis')
plt.colorbar(sc, ax=ax, label='dn (mm)')
ax.set_xlabel('db (mm)'); ax.set_ylabel('Fo (N)')
ax.set_title('Fo vs db')
ax.grid(alpha=0.3)

# ── 左下: Fn vs dn，颜色=位置 ──
ax = axes[1, 0]
sc = ax.scatter(dn, Fn, c=pos, s=2, alpha=0.5, cmap='tab20')
plt.colorbar(sc, ax=ax, label='位置序号')
ax.axhline(-8.0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('dn (mm)'); ax.set_ylabel('Fn (N)')
ax.set_title('Fn vs dn（按位置着色）')
ax.grid(alpha=0.3)

# ── 右下: Fo vs db，颜色=位置 ──
ax = axes[1, 1]
sc = ax.scatter(db, Fo, c=pos, s=2, alpha=0.5, cmap='tab20')
plt.colorbar(sc, ax=ax, label='位置序号')
ax.set_xlabel('db (mm)'); ax.set_ylabel('Fo (N)')
ax.set_title('Fo vs db（按位置着色）')
ax.grid(alpha=0.3)

fig.suptitle('力场采样数据 — 20个位置，4779点', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'force_sweep_scatter.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
