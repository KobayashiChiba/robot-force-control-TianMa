"""
fit_inverse_anchored.py — 带参考点锚定的二次拟合

约束: (Fn=-8, Fo=0) → (dn=0, db=0)
消去常数项: a0 = 8*a1 - 64*a3, b0 = 8*b1 - 64*b3
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
dn_t, db_t = d['dn'], d['db']
pos = d['pos_idx']

# ── 构造特征矩阵（5参数: 消去常数项） ──
X5 = np.column_stack([
    Fn + 8,          # → a1
    Fo,              # → a2
    Fn**2 - 64,      # → a3
    Fn * Fo,         # → a4
    Fo**2,           # → a5
])

c5_dn, _, _, _ = np.linalg.lstsq(X5, dn_t, rcond=None)
c5_db, _, _, _ = np.linalg.lstsq(X5, db_t, rcond=None)

# 还原完整系数（含 a0, b0）
a1, a2, a3, a4, a5 = c5_dn
a0 = 8*a1 - 64*a3
coef_dn = np.array([a0, a1, a2, a3, a4, a5])

b1, b2, b3, b4, b5 = c5_db
b0 = 8*b1 - 64*b3
coef_db = np.array([b0, b1, b2, b3, b4, b5])

# 预测
N = len(Fn)
X6 = np.column_stack([np.ones(N), Fn, Fo, Fn**2, Fn*Fo, Fo**2])
dn_p = X6 @ coef_dn
db_p = X6 @ coef_db
e_dn, e_db = dn_p - dn_t, db_p - db_t

# ── 系数 ──
labels = ['1', 'Fn', 'Fo', 'Fn²', 'Fn*Fo', 'Fo²']
print("锚定二次拟合系数  | 约束: (-8,0)→(0,0)")
print(f"  {'':>8}  {'dn':>10}  {'db':>10}")
for lab, cd, cb in zip(labels, coef_dn, coef_db):
    print(f"  {lab:>8}  {cd:10.6f}  {cb:10.6f}")

print(f"\n残差统计:")
print(f"  dn: MAE={np.mean(np.abs(e_dn)):.4f}mm  RMSE={np.sqrt(np.mean(e_dn**2)):.4f}mm  PC95={np.sort(np.abs(e_dn))[int(N*0.95)]:.4f}")
print(f"  db: MAE={np.mean(np.abs(e_db)):.4f}mm  RMSE={np.sqrt(np.mean(e_db**2)):.4f}mm  PC95={np.sort(np.abs(e_db))[int(N*0.95)]:.4f}")

# ── 关键点验证 ──
def predict(fn, fo):
    x = np.array([1, fn, fo, fn**2, fn*fo, fo**2])
    return float(x @ coef_dn), float(x @ coef_db)

for label, fn, fo in [('(-8,0) 目标', -8, 0), ('(0,0) 无力', 0, 0),
                        ('(-5,0)', -5, 0), ('(-10,0)', -10, 0)]:
    pdn, pdb = predict(fn, fo)
    print(f"  {label}: ({fn},{fo}) → dn={pdn:.3f}mm, db={pdb:.3f}mm")

# ── 与无约束版对比 ──
coef_old = np.load(os.path.join(_sdir, 'output', 'inverse_quadratic.npz'))
dn_old = X6 @ coef_old['coef_dn']
db_old = X6 @ coef_old['coef_db']

print(f"\n── 与无约束版对比 ──")
print(f"                     dn_MAE  dn_PC95  db_MAE  db_PC95")
print(f"  无约束(全量):      {np.mean(np.abs(dn_old - dn_t)):.4f}    {np.sort(np.abs(dn_old - dn_t))[int(N*0.95)]:.4f}    {np.mean(np.abs(db_old - db_t)):.4f}    {np.sort(np.abs(db_old - db_t))[int(N*0.95)]:.4f}")
print(f"  锚定(-8,0)→(0,0):  {np.mean(np.abs(e_dn)):.4f}    {np.sort(np.abs(e_dn))[int(N*0.95)]:.4f}    {np.mean(np.abs(e_db)):.4f}    {np.sort(np.abs(e_db))[int(N*0.95)]:.4f}")

# ── 绘图 ──
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

# Fn-Fo空间 × dn残差
ax = axes[0, 0]
sc = ax.scatter(Fn, Fo, c=np.clip(np.abs(e_dn), 0, 0.5), s=3, alpha=0.5, cmap='YlOrRd')
plt.colorbar(sc, ax=ax, label='|err_dn| (mm)')
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title('锚定版: dn残差 在Fn-Fo空间')
ax.grid(alpha=0.2)

# Fn-Fo空间 × db残差
ax = axes[0, 1]
sc = ax.scatter(Fn, Fo, c=np.clip(np.abs(e_db), 0, 1.0), s=3, alpha=0.5, cmap='YlOrRd')
plt.colorbar(sc, ax=ax, label='|err_db| (mm)')
ax.set_xlabel('Fn (N)'); ax.set_ylabel('Fo (N)')
ax.set_title(f'锚定版: db残差 在Fn-Fo空间  MAE={np.mean(np.abs(e_db)):.3f}mm')
ax.grid(alpha=0.2)

# dn预测vs实际
ax = axes[0, 2]
ax.scatter(dn_t, dn_p, c=pos, s=3, alpha=0.4, cmap='tab20')
mn, mx = dn_t.min(), dn_t.max()
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
ax.set_xlabel('dn 真实 (mm)'); ax.set_ylabel('dn 预测 (mm)')
ax.set_title(f'锚定版 dn  RMSE={np.sqrt(np.mean(e_dn**2)):.3f}mm')
ax.grid(alpha=0.3)

# db预测vs实际
ax = axes[1, 0]
ax.scatter(db_t, db_p, c=pos, s=3, alpha=0.4, cmap='tab20')
mn, mx = db_t.min(), db_t.max()
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
ax.set_xlabel('db 真实 (mm)'); ax.set_ylabel('db 预测 (mm)')
ax.set_title(f'锚定版 db  RMSE={np.sqrt(np.mean(e_db**2)):.3f}mm')
ax.grid(alpha=0.3)

# dn残差分布 对比
ax = axes[1, 1]
ax.hist(np.abs(dn_old - dn_t), bins=80, alpha=0.5, color='steelblue', label='无约束')
ax.hist(np.abs(e_dn), bins=80, alpha=0.5, color='coral', label='锚定(-8,0)→(0,0)')
ax.set_xlabel('|err_dn| (mm)'); ax.set_ylabel('频次')
ax.set_title('dn 残差分布对比')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# db残差分布 对比
ax = axes[1, 2]
ax.hist(np.abs(db_old - db_t), bins=80, alpha=0.5, color='steelblue', label='无约束')
ax.hist(np.abs(e_db), bins=80, alpha=0.5, color='coral', label='锚定(-8,0)→(0,0)')
ax.set_xlabel('|err_db| (mm)'); ax.set_ylabel('频次')
ax.set_title('db 残差分布对比')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

fig.suptitle('锚定二次拟合: (-8,0) → (0,0) 硬约束', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'inverse_anchored_fit.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)

# 保存系数
np.savez(os.path.join(_sdir, 'output', 'inverse_quadratic_anchored.npz'),
          coef_dn=coef_dn, coef_db=coef_db)
print("已保存 inverse_quadratic_anchored.npz")
