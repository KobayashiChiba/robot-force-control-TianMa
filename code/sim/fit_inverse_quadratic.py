"""
fit_inverse_quadratic.py — 全局二次拟合: (Fn, Fo) → (dn, db)

模型:
  dn = a0 + a1*Fn + a2*Fo + a3*Fn² + a4*Fn*Fo + a5*Fo²
  db = b0 + b1*Fn + b2*Fo + b3*Fn² + b4*Fn*Fo + b5*Fo²

输出: 拟合系数 + 残差统计 + 预测vs实际散点图
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import os
_sdir = os.path.dirname(os.path.abspath(__file__))

# ── 加载数据 ──
d = np.load(os.path.join(_sdir, 'output', 'force_sweep_data.npz'))
Fn, Fo = d['Fn'], d['Fo']
dn_true, db_true = d['dn'], d['db']

N = len(Fn)

# ── 构造特征矩阵 X = [1, Fn, Fo, Fn², Fn*Fo, Fo²] ──
X = np.column_stack([
    np.ones(N),
    Fn, Fo,
    Fn**2, Fn*Fo, Fo**2
])

# ── 最小二乘拟合 ──
coef_dn, _, _, _ = np.linalg.lstsq(X, dn_true, rcond=None)
coef_db, _, _, _ = np.linalg.lstsq(X, db_true, rcond=None)

dn_pred = X @ coef_dn
db_pred = X @ coef_db

err_dn = dn_pred - dn_true
err_db = db_pred - db_true

# ── 系数报告 ──
labels = ['1', 'Fn', 'Fo', 'Fn²', 'Fn*Fo', 'Fo²']
print("二次拟合系数：")
print(f"  {'':>8}  {'dn':>10}  {'db':>10}")
print(f"  {'':>8}  {'──':>10}  {'──':>10}")
for lab, cd, cb in zip(labels, coef_dn, coef_db):
    print(f"  {lab:>8}  {cd:10.6f}  {cb:10.6f}")

print(f"\n残差统计:")
print(f"  dn: MAE={np.mean(np.abs(err_dn)):.4f}mm  "
      f"RMSE={np.sqrt(np.mean(err_dn**2)):.4f}mm  "
      f"max_err={np.max(np.abs(err_dn)):.4f}mm")
print(f"  db: MAE={np.mean(np.abs(err_db)):.4f}mm  "
      f"RMSE={np.sqrt(np.mean(err_db**2)):.4f}mm  "
      f"max_err={np.max(np.abs(err_db)):.4f}mm")

# ── PC95 ──
err_dn_sorted = np.sort(np.abs(err_dn))
err_db_sorted = np.sort(np.abs(err_db))
pc95 = int(N * 0.95)
print(f"  dn PC95: {err_dn_sorted[pc95]:.4f}mm")
print(f"  db PC95: {err_db_sorted[pc95]:.4f}mm")

# ── 按位置残差 ──
pos_idx = d['pos_idx']
for pi in range(20):
    mask = pos_idx == pi
    if mask.sum() > 0:
        e_dn = np.mean(np.abs(err_dn[mask]))
        e_db = np.mean(np.abs(err_db[mask]))
        fn0 = np.mean(Fn[mask])
        print(f"  位置{pi:2d} ({mask.sum():3d}点)  Fn0={fn0:+.1f}N  dn_MAE={e_dn:.4f}  db_MAE={e_db:.4f}")

# ── 绘图 ──
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# dn: 预测 vs 实际
ax = axes[0, 0]
ax.scatter(dn_true, dn_pred, c=pos_idx, s=3, alpha=0.4, cmap='tab20')
mn, mx = dn_true.min(), dn_true.max()
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
ax.set_xlabel('dn 真实 (mm)'); ax.set_ylabel('dn 预测 (mm)')
ax.set_title(f'dn: 全局二次拟合  RMSE={np.sqrt(np.mean(err_dn**2)):.3f}mm')
ax.grid(alpha=0.3)

# db: 预测 vs 实际
ax = axes[0, 1]
ax.scatter(db_true, db_pred, c=pos_idx, s=3, alpha=0.4, cmap='tab20')
mn, mx = db_true.min(), db_true.max()
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
ax.set_xlabel('db 真实 (mm)'); ax.set_ylabel('db 预测 (mm)')
ax.set_title(f'db: 全局二次拟合  RMSE={np.sqrt(np.mean(err_db**2)):.3f}mm')
ax.grid(alpha=0.3)

# dn 残差分布
ax = axes[1, 0]
ax.hist(err_dn, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('dn 残差 (mm)'); ax.set_ylabel('频次')
ax.set_title(f'dn 残差分布  MAE={np.mean(np.abs(err_dn)):.3f}mm')
ax.grid(alpha=0.3, axis='y')

# db 残差分布
ax = axes[1, 1]
ax.hist(err_db, bins=80, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('db 残差 (mm)'); ax.set_ylabel('频次')
ax.set_title(f'db 残差分布  MAE={np.mean(np.abs(err_db)):.3f}mm')
ax.grid(alpha=0.3, axis='y')

fig.suptitle('全局二次拟合: (Fn,Fo) → (dn,db)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'inverse_quadratic_fit.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)

# ── 保存系数 ──
np.savez(os.path.join(_sdir, 'output', 'inverse_quadratic.npz'),
          coef_dn=coef_dn, coef_db=coef_db)
print("已保存 inverse_quadratic.npz")
