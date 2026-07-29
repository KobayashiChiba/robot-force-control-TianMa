"""
calibrate_force.py — 力模型完整标定（Fibonacci 采样）

三步：
  1. K_C 标定：在 ball_ref 上取 20 点，按目标力 8N 反推 K_C
  2. 网格力场采集：6 位置 × 21×21 网格 → (Fn, Fo, dn, db) 数据集
  3. 锚定二次拟合 → 输出系数 + 精度报告

输出图：K_C 拟合线 + 逆推残差分布

用法: cd code && python force_feedback_v3/script/calibrate_force.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 加载 ──
cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts
N_BALL = len(ball_ref)

# ═══════════════════════════════════════════
# Step 1: K_C 标定
# ═══════════════════════════════════════════
TARGET_F = 8.0
n_kc = 20
p_kc = np.linspace(0, 1, n_kc + 1)[:-1]
f_kc = []
for p in p_kc:
    i = int(p * (N_BALL - 1))
    F, _ = sphere_contact_force(ball_ref[i], cz, cy)
    f_kc.append(np.linalg.norm(F))
f_kc = np.array(f_kc)
K_C_OLD = np.array([7.37])
K_C_NEW = K_C_OLD[0] * TARGET_F / f_kc.mean()

# 用新 K_C 模拟一次验证
import force_feedback_v3.lib.sphere_contact as sc_mod
sc_mod.K_C = K_C_NEW
f_new = []
areas_new = []
for p in p_kc:
    i = int(p * (N_BALL - 1))
    F, a = sphere_contact_force(ball_ref[i], cz, cy)
    f_new.append(np.linalg.norm(F))
    areas_new.append(a)
f_new = np.array(f_new)
areas_new = np.array(areas_new)
sc_mod.K_C = K_C_OLD[0]

print(f"K_C: {K_C_OLD[0]:.2f} → {K_C_NEW:.4f}  (×{K_C_NEW/K_C_OLD[0]:.4f})")
print(f"  ball_ref |F|: mean={f_new.mean():.2f} std={f_new.std():.2f} N")

# ═══════════════════════════════════════════
# Step 2: 网格力场采集
# ═══════════════════════════════════════════
R_SWEEP, N_SWEEP = 2.0, 21
dn_g = np.linspace(-R_SWEEP, R_SWEEP, N_SWEEP)
db_g = np.linspace(-R_SWEEP, R_SWEEP, N_SWEEP)
positions = np.linspace(0, 0.25, 6)

sc_mod.K_C = K_C_NEW
collect = []
for p in positions:
    i = int(p * (N_BALL - 1))
    P_b = ball_ref[i]
    idx = np.argmin(np.linalg.norm(ct - P_b, axis=1))
    basis = compute_point_basis_ortho(ct[idx], sim.contact_geom)
    n, o = basis.normal, basis.ortho
    for dni in dn_g:
        for dbj in db_g:
            pos = P_b + dni * n + dbj * o
            F, a = sphere_contact_force(pos, cz, cy)
            if a < 0.01:
                continue
            collect.append([np.dot(F, n), np.dot(F, o), dni, dbj])
sc_mod.K_C = K_C_OLD[0]

D = np.array(collect)
Fn, Fo, dn_t, db_t = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
print(f"  网格采样: {len(D)} 有效点 (6位置×21×21)")

# ═══════════════════════════════════════════
# Step 3: 锚定二次拟合
# ═══════════════════════════════════════════
Fn0, Fo0 = -TARGET_F, 0.0
Fn_s = Fn - Fn0
Fo_s = Fo  # Fo0=0

X = np.column_stack([Fn_s, Fo_s, Fn_s**2, Fn_s * Fo_s, Fo_s**2])
coef_dn, *_ = np.linalg.lstsq(X, dn_t, rcond=None)
coef_db, *_ = np.linalg.lstsq(X, db_t, rcond=None)

c0_dn = -(coef_dn[0]*Fn0 + coef_dn[2]*Fn0**2)
c0_db = -(coef_db[0]*Fn0 + coef_db[2]*Fn0**2)

dn_pred = c0_dn + X @ coef_dn
db_pred = c0_db + X @ coef_db
res_dn = dn_t - dn_pred
res_db = db_t - db_pred

print(f"\n  逆推精度:")
print(f"    dn: MAE={np.mean(np.abs(res_dn)):.3f}  RMSE={np.std(res_dn):.3f}  P95={np.percentile(np.abs(res_dn),95):.3f} mm")
print(f"    db: MAE={np.mean(np.abs(res_db)):.3f}  RMSE={np.std(res_db):.3f}  P95={np.percentile(np.abs(res_db),95):.3f} mm")

# 验证锚定点
d0, db0 = c0_dn + coef_dn[0]*Fn0 + coef_dn[2]*Fn0**2, c0_db + coef_db[0]*Fn0 + coef_db[2]*Fn0**2
print(f"    锚定点: (-{TARGET_F:.0f},0) → ({d0:.2e}, {db0:.2e})")

# ═══════════════════════════════════════════
# 输出系数
# ═══════════════════════════════════════════
print(f"\n  # sphere_contact.py")
print(f"  K_C = {K_C_NEW:.4f}")
print(f"\n  # force_field_quadratic.py")
print(f"  _COEF_DN = np.array([{c0_dn:.6f}, {coef_dn[0]:.6f}, {coef_dn[1]:.6f}, {coef_dn[2]:.6f}, {coef_dn[3]:.6f}, {coef_dn[4]:.6f}])")
print(f"  _COEF_DB = np.array([{c0_db:.6f}, {coef_db[0]:.6f}, {coef_db[1]:.6f}, {coef_db[2]:.6f}, {coef_db[3]:.6f}, {coef_db[4]:.6f}])")

# ═══════════════════════════════════════════
# 图 1: K_C 拟合线
# ═══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
sqrt_a = np.sqrt(areas_new)
ax.plot(sqrt_a, f_new, 'o', ms=4, alpha=0.7, label=f'{n_kc} ball_ref points')
s_line = np.linspace(0, sqrt_a.max(), 50)
ax.plot(s_line, K_C_NEW * s_line, 'r-', lw=2, label=f'|F|={K_C_NEW:.4f}·√S')
ax.axhline(TARGET_F, color='gray', ls='--', lw=0.8)
ax.set_xlabel('√S (mm)'); ax.set_ylabel('|F| (N)')
ax.set_title(f'K_C = {K_C_NEW:.4f}  (目标 {TARGET_F}N)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 图 2: dn 残差
ax = axes[1]
ax.hist(res_dn, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='gray', ls='--')
ax.axvline(-np.std(res_dn), color='red', ls=':', lw=0.8)
ax.axvline(+np.std(res_dn), color='red', ls=':', lw=0.8)
ax.set_xlabel('dn residual (mm)'); ax.set_ylabel('count')
ax.set_title(f'dn inverse: MAE={np.mean(np.abs(res_dn)):.3f} ±{np.std(res_dn):.3f}mm')

# 图 3: db 残差
ax = axes[2]
ax.hist(res_db, bins=40, color='coral', edgecolor='white', alpha=0.8)
ax.axvline(0, color='gray', ls='--')
ax.axvline(-np.std(res_db), color='red', ls=':', lw=0.8)
ax.axvline(+np.std(res_db), color='red', ls=':', lw=0.8)
ax.set_xlabel('db residual (mm)'); ax.set_ylabel('count')
ax.set_title(f'db inverse: MAE={np.mean(np.abs(res_db)):.3f} ±{np.std(res_db):.3f}mm')

fig.suptitle(f'Force calibration (Fibonacci, K_C={K_C_NEW:.4f})', fontsize=13)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'calibration.png')
fig.savefig(out_path, dpi=150)
print(f"\n✓ 图已保存: {out_path}")
