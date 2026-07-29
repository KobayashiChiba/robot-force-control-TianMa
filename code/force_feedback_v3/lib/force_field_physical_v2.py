"""
force_field_physical_v2.py — 物理分块力场模型 V2

改进：
  Fn = min(0, -(k_fn*dn + c_fn))                    纯线性，逆推一步代数解
  Fo = 0          (dn<0)                             退刀无力
  Fo = k1*db + k2*dn + k3*dn*db  (dn>=0)             分db>0/db<0两组系数

逆推：
  dn = (|Fn| - c_fn) / k_fn                          只要|Fn|>0.5N
  db = (Fo - k2*dn) / (k1 + k3*dn)                   dn已知，一步
  |Fn|<2N时db=0（浅接触不反推）

特点：3+6=9个参数，纯代数解，无需迭代
"""
import numpy as np

# ── 标定系数（2026-07-29 重新拟合，锚定(-8,0)→(0,0)，K_C=6.5192）──
K_FN = 9.917    # Fn 斜率 (N/mm)，锚定后
C_FN = 8.000    # Fn 截距 (N)，锚定: dn=0 → |Fn|=8.0N ✓

# Fo 系数 (db>0, 右上象限) — 连续约束 k2 共享
K1_RU = 1.779   # db 线性系数
K2_SHARED = -3.260  # dn 线性系数（两段共享，保证 db=0 连续）
K3_RU = 2.934   # dn*db 乘积系数

# Fo 系数 (db<0, 右下象限)
K1_RD = 0.637   # db 线性系数
# K2_SHARED 共享
K3_RD = 1.714   # dn*db 乘积系数

FN_WEAK = 2.0   # 浅接触阈值 (N)，低于此值不反推 db
FN_ZERO = 0.5   # 零接触阈值 (N)


def predict(dn, db):
    """正向预测: (dn,db) → (Fn,Fo)"""
    Fn = min(0.0, -K_FN * dn - C_FN)
    if dn < 0:
        Fo = 0.0
    elif db > 0:
        Fo = K1_RU * db + K2_SHARED * dn + K3_RU * dn * db
    elif db < 0:
        Fo = K1_RD * db + K2_SHARED * dn + K3_RD * dn * db
    else:
        Fo = 0.0
    return Fn, Fo


def inverse(Fn_meas, Fo_meas):
    """逆推: (Fn,Fo) → (dn,db) — 纯代数解

    Args:
        Fn_meas: 有符号法向力 (N)，负值=压入
        Fo_meas: 有符号复法向力 (N)

    Returns:
        (dn, db) 偏移量 (mm)
    """
    fa = abs(Fn_meas)

    # 零接触：无意义反推
    if fa < FN_ZERO:
        return 0.0, 0.0

    # dn = (|Fn| - c) / k  (纯线性反推)
    dn = (fa - C_FN) / K_FN
    if dn <= 0:
        return 0.0, 0.0

    # 浅接触：只反推 dn，不反推 db
    if fa < FN_WEAK:
        return dn, 0.0

    # 深接触：Fo = k1*db + k2*dn + k3*dn*db = db*(k1 + k3*dn) + k2*dn
    # → db = (Fo - k2*dn) / (k1 + k3*dn)
    if Fo_meas > 0:
        k1, k3 = K1_RU, K3_RU
    elif Fo_meas < 0:
        k1, k3 = K1_RD, K3_RD
    else:
        return dn, 0.0

    denom = k1 + k3 * dn
    if abs(denom) < 1e-6:
        return dn, 0.0
    db = (Fo_meas - K2_SHARED * dn) / denom
    return dn, db


# ── 自测 ──
if __name__ == '__main__':
    print(f"物理分块 V2 自测")
    print(f"  Fn = min(0, -{K_FN:.3f}·dn - {C_FN:.3f})")
    print(f"  Fo(dn≥0,db>0) = {K1_RU:.3f}·db + ({K2_SHARED:.3f})·dn + {K3_RU:.3f}·dn·db")
    print(f"  Fo(dn≥0,db<0) = {K1_RD:.3f}·db + ({K2_SHARED:.3f})·dn + {K3_RD:.3f}·dn·db")

    tests = [
        (-8.0, 0.0, "目标点"),
        (0.0, 0.0, "零接触"),
        (-5.0, 0.0, "浅接触"),
        (-10.0, 2.0, "深接触右上"),
        (-10.0, -2.0, "深接触右下"),
    ]
    print(f"\n{'工况':>12}  →  {'dn':>8} {'db':>8}")
    for fn, fo, label in tests:
        dn, db = inverse(fn, fo)
        print(f"{label:>12}  →  {dn:8.3f} {db:8.3f}")

    # 验证往返
    print(f"\n往返验证 (随机100点):")
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from force_feedback_v3.lib import load_cylinders, load_ball_ref
    from force_feedback_v3.lib.sphere_contact import sphere_contact_force
    from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
    from force_feedback_v3.lib.simulator import Simulator

    cy, cz = load_cylinders()
    ball_ref, _ = load_ball_ref()
    sim = Simulator(cy, cz)
    ct = sim.contact_pts

    np.random.seed(42)
    errs_dn, errs_db = [], []
    for _ in range(100):
        p = np.random.uniform(0, 1)
        i = min(int(p * len(ball_ref)), len(ball_ref) - 1)
        bc0 = ball_ref[i]
        idx = np.argmin(np.linalg.norm(ct - bc0, axis=1))
        basis = compute_point_basis_ortho(ct[idx], sim.contact_geom)
        n, o = basis.normal, basis.ortho

        dn_t = np.random.uniform(-1.5, 1.5)
        db_t = np.random.uniform(-1.5, 1.5)
        pos = bc0 + dn_t*n + db_t*o
        F, area = sphere_contact_force(pos, cz, cy)
        if area < 0.01: continue
        Fn_m = np.dot(F, n); Fo_m = np.dot(F, o)
        dn_e, db_e = inverse(Fn_m, Fo_m)
        errs_dn.append(abs(dn_e - dn_t))
        errs_db.append(abs(db_e - db_t))

    errs_dn = np.array(errs_dn); errs_db = np.array(errs_db)
    print(f"  dn误差: median={np.median(errs_dn):.3f}mm  mean={errs_dn.mean():.3f}mm  max={errs_dn.max():.3f}mm")
    print(f"  db误差: median={np.median(errs_db):.3f}mm  mean={errs_db.mean():.3f}mm  max={errs_db.max():.3f}mm")
