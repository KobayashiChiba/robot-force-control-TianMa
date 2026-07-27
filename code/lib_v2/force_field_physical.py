"""
force_field_physical.py — 力场模型：物理直觉模型

模型：
  Fn = min(0, -k·dn - c)          ← 法向力只随切深 dn 变化，正值（拉力）截零
  Fo | dn<0         = 0           ← 退刀无复法向力
  Fo | dn>=0, db>0  = k_ru·dn·db  ← 右上：切歪系数 5.65
  Fo | dn>=0, db<0  = k_rd·dn·db  ← 右下：切歪系数 1.60（截面不对称）

接口：
  calibrate()        → 拟合系数，保存到 data/force_field_physical.npz
  predict(dn, db)    → 预测 (Fn, Fo)
  inverse(Fn, Fo)    → 从测量力反推偏移量 (dn, db)，代数解，不需要迭代

特点：3个参数，物理可解释，逆推简单（代数解）。但浅接触区精度低于二次模型。
"""

import sys, os
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, _sdir)
sys.path.insert(0, os.path.join(_sdir, '..', 'sim'))

import pickle
import numpy as np
from force_profile import sphere_contact_force
from contact_frame_v2 import compute_frame


# ── 全局参数（calibrate() 后填充） ──
_K_FN = 11.70   # 法向力斜率 (N/mm)
_C_FN = 6.48    # 法向力截距 (N) — dn=0 时 |Fn|=6.48N
_K_RU = 5.65    # 右上乘积系数
_K_RD = 1.60    # 右下乘积系数
_FN_THRESHOLD = 2.0  # 最小法向力阈值 (N)，低于此值不逆推 db


def calibrate(save_path='../data/force_field_physical.npz'):
    """拟合物理模型参数，返回系数，保存到 npz"""
    global _K_FN, _C_FN, _K_RU, _K_RD

    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)

    ball = d['ball_center_500']
    cy = d['cyl_contact_y']
    cz = d['cyl_contact_z']
    cg = d['contact_geom'].sample_pts
    Npts = len(ball)

    R = 2
    Ng = 41
    dn = np.linspace(-R, R, Ng)
    db = np.linspace(-R, R, Ng)

    all_Fn_pos = []
    all_dn_f = []
    all_Fo_ru = []
    all_dndb_ru = []
    all_Fo_rd = []
    all_dndb_rd = []

    for p in np.linspace(0, 0.99, 20):
        i = min(int(p * Npts), Npts - 1)
        bc0 = ball[i]
        idx = np.argmin(np.linalg.norm(cg - bc0, axis=1))
        Pc = cg[idx]
        frame = compute_frame(Pc, cy, cz)
        n = frame.normal
        b = np.cross(frame.tangent, n)
        b /= np.linalg.norm(b)

        for dni in dn:
            for dbj in db:
                f, _ = sphere_contact_force(
                    bc0 + dni*n + dbj*b, np.array([0, -1, 0]), cz, cy)
                Fm = np.linalg.norm(f)
                if Fm < 0.5:
                    continue

                Fn_v = np.dot(f, n)   # 有符号法向力（负值=压入）
                Fo_v = np.dot(f, b)

                all_Fn_pos.append(-Fn_v)   # 转正
                all_dn_f.append(dni)

                if dni >= 0 and dbj > 0:
                    all_Fo_ru.append(Fo_v)
                    all_dndb_ru.append(dni * dbj)
                elif dni >= 0 and dbj < 0:
                    all_Fo_rd.append(Fo_v)
                    all_dndb_rd.append(dni * dbj)

    all_Fn_pos = np.array(all_Fn_pos)
    all_dn_f = np.array(all_dn_f)

    # Fn = max(0, k*dn + c) → 拟合 |Fn| = k*dn + c
    A = np.column_stack([all_dn_f, np.ones_like(all_dn_f)])
    kc, _, _, _ = np.linalg.lstsq(A, all_Fn_pos, rcond=None)
    _K_FN = float(kc[0])
    _C_FN = float(kc[1])

    # Fo(右上)
    A_ru = np.array(all_dndb_ru).reshape(-1, 1)
    _K_RU = float(np.linalg.lstsq(A_ru, np.array(all_Fo_ru), rcond=None)[0][0])

    # Fo(右下)
    A_rd = np.array(all_dndb_rd).reshape(-1, 1)
    _K_RD = float(np.linalg.lstsq(A_rd, np.array(all_Fo_rd), rcond=None)[0][0])

    np.savez(os.path.join(_sdir, '..', 'data', 'force_field_physical.npz'), k_fn=_K_FN, c_fn=_C_FN, k_ru=_K_RU, k_rd=_K_RD)

    Fn_pred = A @ kc
    rms_fn = np.sqrt(np.mean((all_Fn_pos - Fn_pred) ** 2))
    rms_ru = np.sqrt(np.mean((np.array(all_Fo_ru) - np.array(all_dndb_ru) * _K_RU) ** 2))
    rms_rd = np.sqrt(np.mean((np.array(all_Fo_rd) - np.array(all_dndb_rd) * _K_RD) ** 2))

    print(f'物理模型拟合完成: {len(all_Fn_pos)} 采样点')
    print(f'  Fn = min(0, -{_K_FN:.2f}·dn - {_C_FN:.2f})   RMS={rms_fn:.2f}N')
    print(f'  Fo(dn>=0,db>0) = {_K_RU:.3f}·dn·db          RMS={rms_ru:.2f}N')
    print(f'  Fo(dn>=0,db<0) = {_K_RD:.3f}·dn·db          RMS={rms_rd:.2f}N')

    return {'k_fn': _K_FN, 'c_fn': _C_FN, 'k_ru': _K_RU, 'k_rd': _K_RD}


def _load_if_needed():
    """惰性加载标定数据"""
    global _K_FN, _C_FN, _K_RU, _K_RD
    if _K_FN is not None:
        return
    try:
        npz = np.load(os.path.join(_sdir, '..', 'data', 'force_field_physical.npz'))
        _K_FN = float(npz['k_fn'])
        _C_FN = float(npz['c_fn'])
        _K_RU = float(npz['k_ru'])
        _K_RD = float(npz['k_rd'])
    except FileNotFoundError:
        pass  # 使用默认值


def predict(dn, db):
    """预测偏移量 (dn, db) 下的力 (Fn, Fo)"""
    _load_if_needed()
    # Fn = min(0, -k·dn - c)
    Fn = min(0.0, -_K_FN * dn - _C_FN)

    # Fo
    if dn < 0:
        Fo = 0.0
    elif db > 0:
        Fo = _K_RU * dn * db
    elif db < 0:
        Fo = _K_RD * dn * db
    else:
        Fo = 0.0

    return Fn, Fo


def inverse(Fn_meas, Fo_meas):
    """从测量力反推偏移量（代数解）

    Args:
        Fn_meas: 有符号法向力 (N)，负值=压入
        Fo_meas: 有符号复法向力 (N)
    Returns:
        (dn, db) 偏移量 (mm)
        如果 |Fn| < 阈值（信噪比不足），返回 (dn, 0) — 仅沿法向反推
    """
    _load_if_needed()

    fn_abs = abs(Fn_meas)

    if fn_abs < 0.5:
        return 0.0, 0.0  # 完全脱离接触

    # 反推 dn: |Fn| = k·dn + c → dn = (|Fn| - c) / k
    dn_est = (fn_abs - _C_FN) / _K_FN

    if dn_est <= 0:
        return 0.0, 0.0  # 不在接触区

    # 浅接触保护：|Fn| 太小时 db 反推不可靠
    if fn_abs < _FN_THRESHOLD:
        return dn_est, 0.0

    # 反推 db: Fo = k * dn * db → db = Fo / (k * dn)
    if Fo_meas > 0:
        db_est = Fo_meas / (_K_RU * dn_est)
    elif Fo_meas < 0:
        db_est = Fo_meas / (_K_RD * dn_est)
    else:
        db_est = 0.0

    return dn_est, db_est


# ── 自测 ──
if __name__ == '__main__':
    calibrate()

    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    ball = d['ball_center_500']
    cy = d['cyl_contact_y']
    cz = d['cyl_contact_z']
    cg = d['contact_geom'].sample_pts

    np.random.seed(42)
    errs = []
    n_ok = 0
    n_shallow = 0
    for _ in range(100):
        p = np.random.uniform(0, 1)
        i = min(int(p * len(ball)), len(ball) - 1)
        bc0 = ball[i]
        idx = np.argmin(np.linalg.norm(cg - bc0, axis=1))
        Pc = cg[idx]
        frame = compute_frame(Pc, cy, cz)
        n = frame.normal
        b = np.cross(frame.tangent, n)
        b /= np.linalg.norm(b)

        dn_t = np.random.uniform(-1.5, 1.5)
        db_t = np.random.uniform(-1.5, 1.5)
        f, _ = sphere_contact_force(
            bc0 + dn_t*n + db_t*b, np.array([0, -1, 0]), cz, cy)
        if np.linalg.norm(f) < 0.5:
            continue
        Fn_m = np.dot(f, n)
        Fo_m = np.dot(f, b)
        dn_e, db_e = inverse(Fn_m, Fo_m)

        if abs(Fn_m) < _FN_THRESHOLD:
            n_shallow += 1
            # 浅接触: 只验证 dn 精度
            err = abs(dn_e - dn_t)
        else:
            n_ok += 1
            err = np.linalg.norm([dn_e - dn_t, db_e - db_t])
        errs.append(err)

    errs = np.array(errs)
    print(f'\n自测 {len(errs)} 点 ({n_ok} 深接触, {n_shallow} 浅接触)')
    print(f'  中位误差 {np.median(errs):.3f}mm  均值 {errs.mean():.3f}mm  max {errs.max():.3f}mm')
    print(f'  深接触中位 {np.median(errs[-n_ok:]) if n_ok else 0:.3f}mm')
