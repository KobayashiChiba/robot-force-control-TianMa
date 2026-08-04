"""run_sim_physical_v2_variants.py — physical_v2 逆推三个修复方向对比

修复方向:
  fix1: 零接触去截断（|Fn|<0.5N 时 dn 连续，不再 return (0,0)）
  fix2: fix1 + db smoothstep 平滑（|Fn|=8N 边界 db 从 0 渐变到反推值）
  fix3: fix2 + Fo=0 两段系数连续过渡（右上/右下系数平滑混合）

跑有误差 seed42，每个版本独立一圈，输出 Fn 统计 + 对比图。
用法: python script/run_sim_physical_v2_variants.py [seed]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from force_feedback_v3.lib import (load_cylinders, load_ball_ref,
                                   generate_error_cylinders)
from force_feedback_v3.lib.simulator import Simulator
from force_feedback_v3.lib.controller import ForceController
import force_feedback_v3.lib.controller as CTRL

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── physical_v2 系数 ──
K_FN, C_FN = 9.917, 8.000
K1_RU, K2_SH, K3_RU = 1.779, -3.260, 2.934
K1_RD, K3_RD = 0.637, 1.714
FN_WEAK, FN_ZERO = 2.0, 0.5
DN_SAT, FO_SAT = 0.3, 0.3   # 平滑过渡半宽

def _db_full(Fo, dn, k1, k3):
    denom = k1 + k3 * dn
    if abs(denom) < 1e-6:
        return 0.0
    return (Fo - K2_SH * dn) / denom

def _pick_k(Fo):
    return (K1_RU, K3_RU) if Fo > 0 else (K1_RD, K3_RD)

def inv_base(Fn_meas, Fo_meas):
    """基线 = 当前 lib 版（dn<=0 保留负值，零接触仍截 0）"""
    fa = abs(Fn_meas)
    if fa < FN_ZERO:
        return 0.0, 0.0
    dn = (fa - C_FN) / K_FN
    if dn <= 0 or fa < FN_WEAK:
        return dn, 0.0
    k1, k3 = _pick_k(Fo_meas)
    return dn, _db_full(Fo_meas, dn, k1, k3)

def inv_fix1(Fn_meas, Fo_meas):
    """零接触去截断（全范围 dn 连续）"""
    fa = abs(Fn_meas)
    dn = (fa - C_FN) / K_FN          # 全范围连续，不再截断
    if dn <= 0 or fa < FN_WEAK:
        return dn, 0.0
    k1, k3 = _pick_k(Fo_meas)
    return dn, _db_full(Fo_meas, dn, k1, k3)

def inv_fix2(Fn_meas, Fo_meas):
    """fix1 + db smoothstep（dn∈[0,DN_SAT] 从 0 渐变）"""
    fa = abs(Fn_meas)
    dn = (fa - C_FN) / K_FN
    if dn <= 0 or fa < FN_WEAK:
        return dn, 0.0
    k1, k3 = _pick_k(Fo_meas)
    db = _db_full(Fo_meas, dn, k1, k3)
    if dn < DN_SAT:                  # 浅深接触过渡：db 渐变
        t = dn / DN_SAT
        w = t * t * (3 - 2 * t)
        db = w * db
    return dn, db

def inv_fix3(Fn_meas, Fo_meas):
    """fix2 + Fo=0 两段系数连续过渡"""
    fa = abs(Fn_meas)
    dn = (fa - C_FN) / K_FN
    if dn <= 0 or fa < FN_WEAK:
        return dn, 0.0
    # Fo 在 [-FO_SAT, +FO_SAT] 内混合右上/右下系数
    if Fo_meas > FO_SAT:
        k1, k3 = K1_RU, K3_RU
    elif Fo_meas < -FO_SAT:
        k1, k3 = K1_RD, K3_RD
    else:
        t = (Fo_meas + FO_SAT) / (2 * FO_SAT)
        w = t * t * (3 - 2 * t)
        db_ru = _db_full(Fo_meas, dn, K1_RU, K3_RU)
        db_rd = _db_full(Fo_meas, dn, K1_RD, K3_RD)
        db = (1 - w) * db_rd + w * db_ru
        return dn, db
    db = _db_full(Fo_meas, dn, k1, k3)
    if dn < DN_SAT:
        t = dn / DN_SAT
        w = t * t * (3 - 2 * t)
        db = w * db
    return dn, db


def run_one(inv_func, seed):
    np.random.seed(seed)
    cy, cz = load_cylinders()
    ball_ref, L = load_ball_ref()
    rng = np.random.RandomState(seed)
    cz_err, cy_err = generate_error_cylinders(cy, cz, rng)

    sim = Simulator(cy, cz, mu=0.2, sigma=0.5)
    CTRL.inverse = inv_func              # monkeypatch 替换控制器逆推
    ctrl = ForceController(ball_ref, L, sim.contact_geom)

    DT = 0.005; N_STEPS = 2000; MAX_STEPS = 2 * N_STEPS
    cy0 = np.mean(sim.contact_pts[:, 1]); cz0 = np.mean(sim.contact_pts[:, 2])
    pos = ball_ref[0].copy(); v_prev = np.zeros(3)
    log_Fn = []; accum = 0.0
    theta_prev = np.arctan2(pos[2]-cz0, pos[1]-cy0)

    for k in range(MAX_STEPS):
        F_meas, _, _, _, basis = sim.step(pos, v_prev, cy_err=cy_err, cz_err=cz_err)
        v_3d = ctrl.step(F_meas, pos, N_STEPS, DT)
        pos += v_3d * DT; v_prev = v_3d
        log_Fn.append(np.dot(F_meas, basis.normal))
        theta = np.arctan2(pos[2]-cz0, pos[1]-cy0)
        dtheta = theta - theta_prev
        if dtheta > np.pi: dtheta -= 2*np.pi
        elif dtheta < -np.pi: dtheta += 2*np.pi
        accum += dtheta; theta_prev = theta
        if abs(accum) >= 2*np.pi: break

    log_Fn = np.array(log_Fn)
    n1 = 0  # limit>1mm 统计在下面从轨迹重算，这里简化
    return log_Fn, n1


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    variants = {
        '基线(当前lib)': inv_base,
        'fix1 零接触去截断': inv_fix1,
        'fix2 +db平滑': inv_fix2,
        'fix3 +Fo连续': inv_fix3,
    }
    results = {}
    for name, fn in variants.items():
        log_Fn, _ = run_one(fn, seed)
        results[name] = log_Fn
        mean, std = log_Fn[50:].mean(), log_Fn[50:].std()
        print(f'{name:>16}: {len(log_Fn)} steps  Fn={mean:.2f}±{std:.2f}N')

    # 对比图
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {'基线(当前lib)': 'gray', 'fix1 零接触去截断': 'steelblue',
              'fix2 +db平滑': 'darkorange', 'fix3 +Fo连续': 'darkgreen'}
    for name, log_Fn in results.items():
        c = colors.get(name, 'gray')
        ax.plot(log_Fn, lw=0.9, color=c, label=f'{name} (mean={log_Fn[50:].mean():.2f})')
    ax.axhline(-8, color='gray', ls='--', lw=1, label='target -8N')
    ax.set_xlabel('step'); ax.set_ylabel('Fn (N)'); ax.set_ylim(-20, 0)
    ax.set_title(f'physical_v2 修复方向对比 (seed={seed})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), '..', 'output', f'physical_v2_variants_seed{seed}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f'\n✓ {out}')


if __name__ == '__main__':
    main()
