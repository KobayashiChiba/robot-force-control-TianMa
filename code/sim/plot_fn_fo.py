"""
plot_fn_fo.py — 横轴Fn 纵轴Fo，dn/db各一张热力图
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

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# ── 左: Fn-Fo 空间，颜色=dn ──
ax = axes[0]
sc = ax.scatter(Fn, Fo, c=dn, s=4, alpha=0.6, cmap='RdYlGn', vmin=-2, vmax=2)
plt.colorbar(sc, ax=ax, label='dn (mm)', shrink=0.85)
ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title('Fn-Fo 空间 × dn 着色')
ax.grid(alpha=0.2)

# ── 右: Fn-Fo 空间，颜色=db ──
ax = axes[1]
sc = ax.scatter(Fn, Fo, c=db, s=4, alpha=0.6, cmap='coolwarm', vmin=-2, vmax=2)
plt.colorbar(sc, ax=ax, label='db (mm)', shrink=0.85)
ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title('Fn-Fo 空间 × db 着色')
ax.grid(alpha=0.2)

fig.suptitle('力场采样 — Fn(Fo) 空间 (20位置, 4779点)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'force_sweep_fn_fo.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
