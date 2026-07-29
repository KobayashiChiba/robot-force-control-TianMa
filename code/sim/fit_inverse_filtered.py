"""
fit_inverse_filtered.py — 按 |Fn| 阈值过滤后二次拟合
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
Fn_all, Fo_all = d['Fn'], d['Fo']
dn_all, db_all = d['dn'], d['db']

# 试几个阈值
thresholds = [0, 1, 2, 3, 4, 6, 8]
results = []

for th in thresholds:
    mask = np.abs(Fn_all) >= th
    N_kept = mask.sum()
    if N_kept < 100:
        continue

    Fn, Fo = Fn_all[mask], Fo_all[mask]
    dn_t, db_t = dn_all[mask], db_all[mask]
    N = len(Fn)

    X = np.column_stack([np.ones(N), Fn, Fo, Fn**2, Fn*Fo, Fo**2])
    c_dn, _, _, _ = np.linalg.lstsq(X, dn_t, rcond=None)
    c_db, _, _, _ = np.linalg.lstsq(X, db_t, rcond=None)

    e_dn = np.abs(X @ c_dn - dn_t)
    e_db = np.abs(X @ c_db - db_t)
    pc95_dn = np.sort(e_dn)[int(N*0.95)]
    pc95_db = np.sort(e_db)[int(N*0.95)]

    results.append((th, N_kept, e_dn.mean(), np.sqrt(np.mean(e_dn**2)),
                    pc95_dn, e_db.mean(), np.sqrt(np.mean(e_db**2)), pc95_db))

# 打印对比
print(f"{'阈值|Fn|>=':>10} {'保留':>6} {'dn_MAE':>8} {'dn_RMSE':>8} {'dn_PC95':>8} {'db_MAE':>8} {'db_RMSE':>8} {'db_PC95':>8}")
print(f"{'':>10} {'':>6} {'mm':>8} {'mm':>8} {'mm':>8} {'mm':>8} {'mm':>8} {'mm':>8}")
print(f"{'':>10} {'':>6} {'──':>8} {'──':>8} {'──':>8} {'──':>8} {'──':>8} {'──':>8}")
for th, n, m_dn, r_dn, p_dn, m_db, r_db, p_db in results:
    marker = " ← 全量" if th == 0 else ""
    print(f"{th:>5}N      {n:>6}  {m_dn:8.4f}  {r_dn:8.4f}  {p_dn:8.4f}  {m_db:8.4f}  {r_db:8.4f}  {p_db:8.4f}{marker}")

# ── 最佳阈值详细拟合 (|Fn|≥3N) ──
best_th = 3
mask = np.abs(Fn_all) >= best_th
Fn, Fo = Fn_all[mask], Fo_all[mask]
dn_t, db_t = dn_all[mask], db_all[mask]
pos = d['pos_idx'][mask]
N = len(Fn)

X = np.column_stack([np.ones(N), Fn, Fo, Fn**2, Fn*Fo, Fo**2])
c_dn, _, _, _ = np.linalg.lstsq(X, dn_t, rcond=None)
c_db, _, _, _ = np.linalg.lstsq(X, db_t, rcond=None)

dn_p = X @ c_dn
db_p = X @ c_db
e_dn, e_db = dn_p - dn_t, db_p - db_t

print(f"\n── 最佳阈值 |Fn|≥{best_th}N ({N}点) 系数 ──")
labels = ['1', 'Fn', 'Fo', 'Fn²', 'Fn*Fo', 'Fo²']
for lab, cd, cb in zip(labels, c_dn, c_db):
    print(f"  {lab:>8}  {cd:10.6f}  {cb:10.6f}")

# 全量验证（在被过滤掉的点上也评估）
mask_full = np.abs(Fn_all) >= best_th
mask_weak = ~mask_full
if mask_weak.sum() > 0:
    X_weak = np.column_stack([np.ones(mask_weak.sum()), Fn_all[mask_weak], Fo_all[mask_weak],
                               Fn_all[mask_weak]**2, Fn_all[mask_weak]*Fo_all[mask_weak], Fo_all[mask_weak]**2])
    e_dn_w = np.abs(X_weak @ c_dn - dn_all[mask_weak])
    e_db_w = np.abs(X_weak @ c_db - db_all[mask_weak])
    print(f"\n  在被过滤的 {mask_weak.sum()} 个浅接触点上验证:")
    print(f"    dn MAE={e_dn_w.mean():.4f}  RMSE={np.sqrt(np.mean(e_dn_w**2)):.4f}")
    print(f"    db MAE={e_db_w.mean():.4f}  RMSE={np.sqrt(np.mean(e_db_w**2)):.4f}")

# 各位置残差
print(f"\n各位置残差 (|Fn|≥{best_th}N):")
for pi in range(20):
    mp = (pos == pi)
    if mp.sum() > 0:
        print(f"  位置{pi:2d} ({mp.sum():3d}点)  dn_MAE={np.abs(e_dn[mp]).mean():.4f}  db_MAE={np.abs(e_db[mp]).mean():.4f}")

# ── 绘图：过滤前 vs 过滤后 ──
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

# 用全量系数（之前的结果）对比
coef_old = np.load(os.path.join(_sdir, 'output', 'inverse_quadratic.npz'))
X_all = np.column_stack([np.ones(len(Fn_all)), Fn_all, Fo_all, Fn_all**2, Fn_all*Fo_all, Fo_all**2])
db_old = X_all @ coef_old['coef_db']
e_old = np.abs(db_old - db_all)

# 1. db残差 vs |Fn|（过滤前）
ax = axes[0, 0]
ax.scatter(np.abs(Fn_all), np.abs(db_old - db_all), s=2, alpha=0.3, c='steelblue')
ax.axvline(best_th, color='red', ls='--', lw=1.2, label=f'阈值|Fn|={best_th}N')
ax.set_xlabel('|Fn| (N)'); ax.set_ylabel('|err_db| (mm)')
ax.set_title('过滤前: db残差 vs |Fn|')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# 2. db残差 vs |Fn|（过滤后拟合）
ax = axes[0, 1]
e_new_all = np.abs(X_all @ c_db - db_all)
ax.scatter(np.abs(Fn_all), e_new_all, s=2, alpha=0.3, c='coral')
ax.axvline(best_th, color='red', ls='--', lw=1.2)
ax.set_xlabel('|Fn| (N)'); ax.set_ylabel('|err_db| (mm)')
ax.set_title(f'过滤后 (|Fn|≥{best_th}N): db残差 vs |Fn|')
ax.grid(alpha=0.2)

# 3. dn预测vs实际
ax = axes[0, 2]
ax.scatter(dn_t, dn_p, c=pos, s=3, alpha=0.4, cmap='tab20')
mn, mx = dn_t.min(), dn_t.max()
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
ax.set_xlabel('dn 真实 (mm)'); ax.set_ylabel('dn 预测 (mm)')
ax.set_title(f'dn 拟合 (|Fn|≥{best_th}N)  RMSE={np.sqrt(np.mean(e_dn**2)):.3f}mm')
ax.grid(alpha=0.3)

# 4. db预测vs实际
ax = axes[1, 0]
ax.scatter(db_t, db_p, c=pos, s=3, alpha=0.4, cmap='tab20')
mn, mx = db_t.min(), db_t.max()
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
ax.set_xlabel('db 真实 (mm)'); ax.set_ylabel('db 预测 (mm)')
ax.set_title(f'db 拟合 (|Fn|≥{best_th}N)  RMSE={np.sqrt(np.mean(e_db**2)):.3f}mm')
ax.grid(alpha=0.3)

# 5. db残差分布 过滤前vs后
ax = axes[1, 1]
ax.hist(np.abs(db_old - db_all), bins=80, alpha=0.5, color='steelblue', label='过滤前拟合')
ax.hist(e_new_all, bins=80, alpha=0.5, color='coral', label=f'|Fn|≥{best_th}N拟合')
ax.axvline(np.median(np.abs(db_old - db_all)), color='steelblue', ls='-', lw=1.2)
ax.axvline(np.median(e_new_all), color='coral', ls='-', lw=1.2)
ax.set_xlabel('|err_db| (mm)'); ax.set_ylabel('频次')
ax.set_title('db 残差分布对比')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# 6. dn残差分布 过滤前vs后
ax = axes[1, 2]
dn_old = X_all @ coef_old['coef_dn']
ax.hist(np.abs(dn_old - dn_all), bins=80, alpha=0.5, color='steelblue', label='过滤前拟合')
ax.hist(np.abs(X_all @ c_dn - dn_all), bins=80, alpha=0.5, color='coral', label=f'|Fn|≥{best_th}N拟合')
ax.set_xlabel('|err_dn| (mm)'); ax.set_ylabel('频次')
ax.set_title('dn 残差分布对比')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

fig.suptitle(f'二次拟合: 过滤 vs 全量 (|Fn|≥{best_th}N, 保留{N}/{len(Fn_all)}点)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'inverse_filtered_fit.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)

# 保存过滤版系数
np.savez(os.path.join(_sdir, 'output', 'inverse_quadratic_filtered.npz'),
          coef_dn=c_dn, coef_db=c_db, threshold=best_th)
print("已保存 inverse_quadratic_filtered.npz")
