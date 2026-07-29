"""
analyze_db_outliers.py — 分析 db 逆推的离群值分布
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
dn_true, db_true = d['dn'], d['db']
pos_idx = d['pos_idx'].astype(int)

# 用之前拟合的系数
coef = np.load(os.path.join(_sdir, 'output', 'inverse_quadratic.npz'))
coef_db = coef['coef_db']

# 预测
X = np.column_stack([np.ones(len(Fn)), Fn, Fo, Fn**2, Fn*Fo, Fo**2])
db_pred = X @ coef_db
err_db = db_pred - db_true
abs_err = np.abs(err_db)

# 取 top-20 离群值和阈值标记
N = len(db_true)
cut_95 = np.sort(abs_err)[int(N*0.95)]     # ~0.68mm
cut_99 = np.sort(abs_err)[int(N*0.99)]
outlier_mask = abs_err > cut_95
extreme_mask = abs_err > cut_99

print(f"db 残差分位数:")
print(f"  50%: {np.median(abs_err):.4f}mm")
print(f"  80%: {np.sort(abs_err)[int(N*0.80)]:.4f}mm")
print(f"  90%: {np.sort(abs_err)[int(N*0.90)]:.4f}mm")
print(f"  95%: {cut_95:.4f}mm")
print(f"  99%: {cut_99:.4f}mm")
print(f"  100%: {abs_err.max():.4f}mm")
print(f"\n离群值 (>P95, {cut_95:.2f}mm): {outlier_mask.sum()} 个 ({outlier_mask.sum()/N*100:.1f}%)")
print(f"极端值 (>P99, {cut_99:.2f}mm): {extreme_mask.sum()} 个")

# 离群值的特征
print(f"\n离群值特征分布:")
o_Fn = Fn[outlier_mask]
o_Fo = Fo[outlier_mask]
o_dn = dn_true[outlier_mask]
o_db = db_true[outlier_mask]
o_err = err_db[outlier_mask]
o_pos = pos_idx[outlier_mask]

print(f"  Fn: [{o_Fn.min():.1f}, {o_Fn.max():.1f}] 均值={o_Fn.mean():.1f}N")
print(f"  Fo: [{o_Fo.min():.1f}, {o_Fo.max():.1f}] 均值={o_Fo.mean():.1f}N")
print(f"  dn: [{o_dn.min():.2f}, {o_dn.max():.2f}] 均值={o_dn.mean():.2f}mm")
print(f"  db: [{o_db.min():.2f}, {o_db.max():.2f}] 均值={o_db.mean():.2f}mm")

# 按位置统计离群值比例
print(f"\n各位置离群值比例:")
for pi in range(20):
    mask_p = pos_idx == pi
    n_ol = (abs_err > cut_95)[mask_p].sum()
    n_p = mask_p.sum()
    print(f"  位置{pi:2d}: {n_ol:3d}/{n_p:3d} = {n_ol/n_p*100:4.1f}%")

# ── 可视化 ──
fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# 1. Fn-Fo 空间，颜色=残差绝对值，高亮离群值
ax = axes[0, 0]
sc = ax.scatter(Fn, Fo, c=np.clip(abs_err, 0, 1.0), s=3, alpha=0.5,
                cmap='YlOrRd', vmin=0, vmax=1.0)
ax.scatter(Fn[outlier_mask], Fo[outlier_mask], s=15, edgecolors='red',
           facecolors='none', linewidths=0.8, alpha=0.8, label=f'离群值>{cut_95:.2f}mm')
plt.colorbar(sc, ax=ax, label='|err_db| (mm)', shrink=0.85)
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title(f'Fn-Fo 空间 × db残差 (离群>{cut_95:.2f}mm, {outlier_mask.sum()}点)')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# 2. 残差 vs db_true（看是否db极端值时残差大）
ax = axes[0, 1]
ax.scatter(db_true[~outlier_mask], abs_err[~outlier_mask], s=3, alpha=0.3, c='steelblue', label='正常')
ax.scatter(db_true[outlier_mask], abs_err[outlier_mask], s=10, alpha=0.8, c='red', label=f'离群值')
ax.axhline(cut_95, color='red', ls='--', lw=0.8)
ax.set_xlabel('真实 db (mm)'); ax.set_ylabel('|err_db| (mm)')
ax.set_title('残差 vs 真实 db')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# 3. 残差 vs dn_true
ax = axes[0, 2]
ax.scatter(dn_true[~outlier_mask], abs_err[~outlier_mask], s=3, alpha=0.3, c='steelblue', label='正常')
ax.scatter(dn_true[outlier_mask], abs_err[outlier_mask], s=10, alpha=0.8, c='red', label=f'离群值')
ax.axhline(cut_95, color='red', ls='--', lw=0.8)
ax.set_xlabel('真实 dn (mm)'); ax.set_ylabel('|err_db| (mm)')
ax.set_title('残差 vs 真实 dn')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# 4. Fn-Fo 空间，颜色=dn
ax = axes[1, 0]
sc = ax.scatter(Fn, Fo, c=dn_true, s=3, alpha=0.5, cmap='Greens', vmin=0, vmax=2)
ax.scatter(Fn[outlier_mask], Fo[outlier_mask], s=15, edgecolors='red',
           facecolors='none', linewidths=0.8, alpha=0.9)
plt.colorbar(sc, ax=ax, label='dn (mm)', shrink=0.85)
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title('Fn-Fo 空间 × dn（离群值圈出）')
ax.grid(alpha=0.2)

# 5. Fn-Fo 空间，颜色=db
ax = axes[1, 1]
sc = ax.scatter(Fn, Fo, c=db_true, s=3, alpha=0.5, cmap='coolwarm', vmin=-2, vmax=2)
ax.scatter(Fn[outlier_mask], Fo[outlier_mask], s=15, edgecolors='red',
           facecolors='none', linewidths=0.8, alpha=0.9)
plt.colorbar(sc, ax=ax, label='db (mm)', shrink=0.85)
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title('Fn-Fo 空间 × db（离群值圈出）')
ax.grid(alpha=0.2)

# 6. 残差 vs Fo
ax = axes[1, 2]
ax.scatter(Fo[~outlier_mask], abs_err[~outlier_mask], s=3, alpha=0.3, c='steelblue', label='正常')
ax.scatter(Fo[outlier_mask], abs_err[outlier_mask], s=10, alpha=0.8, c='red', label=f'离群值')
ax.axhline(cut_95, color='red', ls='--', lw=0.8)
ax.set_xlabel('Fo (N)'); ax.set_ylabel('|err_db| (mm)')
ax.set_title('残差 vs Fo')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle('db 逆推离群值分析', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'db_outliers_analysis.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)
