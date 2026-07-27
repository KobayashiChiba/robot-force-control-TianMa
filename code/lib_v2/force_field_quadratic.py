"""
force_field_quadratic.py — 力场模型：统一二次多项式

模型：
  ΔFn = Fn - Fn0(p) = c1*dn + c2*db + c3*dn² + c4*dn·db + c5*db²
  ΔFo = Fo - Fo0(p) = c1*dn + c2*db + c3*dn² + c4*dn·db + c5*db²

接口：
  calibrate()        → 拟合系数，保存到 data/force_field_quadratic.npz
  predict(dn, db)    → 预测 (ΔFn, ΔFo)
  inverse(Fn, Fo, p) → 从测量力反推偏移量 (dn, db)，需传入曲线进度 p 用于查基准力

特点：精度高，逆推稳定，需要 Newton 迭代（2×2 方程组，通常3-5步收敛）
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
_COEF = None       # dict: {'dn': [c1..c5], 'dfo': [c1..c5]}
_BASE_TABLE = None  # list of (p_idx, Fn0, Fo0)


def calibrate(save_path='../data/force_field_quadratic.npz'):
    """统一二次拟合，返回系数，保存到 npz"""
    global _COEF, _BASE_TABLE

    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)

    ball = d['ball_center_500']
    cy = d['cyl_contact_y']
    cz = d['cyl_contact_z']
    cg = d['contact_geom'].sample_pts
    Npts = len(ball)

    R = 2
    Ng = 21
    dn = np.linspace(-R, R, Ng)
    db = np.linspace(-R, R, Ng)

    base_list = []
    all_dFn = []
    all_dFo = []
    AA = []

    for p in np.linspace(0, 0.99, 20):
        i = min(int(p * Npts), Npts - 1)
        bc0 = ball[i]
        idx = np.argmin(np.linalg.norm(cg - bc0, axis=1))
        Pc = cg[idx]
        frame = compute_frame(Pc, cy, cz)
        n = frame.normal
        b = np.cross(frame.tangent, n)
        b /= np.linalg.norm(b)

        f0, _ = sphere_contact_force(bc0, np.array([0, -1, 0]), cz, cy)
        Fn0 = np.dot(f0, n)
        Fo0 = np.dot(f0, b)
        base_list.append((i, Fn0, Fo0))

        for dni in dn:
            for dbj in db:
                f, _ = sphere_contact_force(
                    bc0 + dni*n + dbj*b, np.array([0, -1, 0]), cz, cy)
                if np.linalg.norm(f) > 0.5:
                    all_dFn.append(np.dot(f, n) - Fn0)
                    all_dFo.append(np.dot(f, b) - Fo0)
                    AA.append([dni, dbj, dni**2, dni*dbj, dbj**2])

    all_dFn = np.array(all_dFn)
    all_dFo = np.array(all_dFo)
    AA = np.array(AA)

    c_dfn, _, _, _ = np.linalg.lstsq(AA, all_dFn, rcond=None)
    c_dfo, _, _, _ = np.linalg.lstsq(AA, all_dFo, rcond=None)

    _COEF = {'dn': list(c_dfn), 'dfo': list(c_dfo)}
    _BASE_TABLE = base_list

    np.savez(os.path.join(_sdir, '..', 'data', 'force_field_quadratic.npz'),
             c_dfn=c_dfn, c_dfo=c_dfo,
             base_indices=[b[0] for b in base_list],
             base_Fn0=[b[1] for b in base_list],
             base_Fo0=[b[2] for b in base_list])

    print(f'二次拟合完成: {len(all_dFn)} 采样点')
    print(f'  ΔFn = {c_dfn[0]:+.3f}*dn {c_dfn[1]:+.3f}*db {c_dfn[2]:+.3f}*dn² {c_dfn[3]:+.3f}*dn·db {c_dfn[4]:+.3f}*db²')
    print(f'  ΔFo = {c_dfo[0]:+.3f}*dn {c_dfo[1]:+.3f}*db {c_dfo[2]:+.3f}*dn² {c_dfo[3]:+.3f}*dn·db {c_dfo[4]:+.3f}*db²')

    return _COEF, _BASE_TABLE


def _load_if_needed():
    """惰性加载已保存的标定数据"""
    global _COEF, _BASE_TABLE
    if _COEF is not None:
        return
    try:
        npz = np.load(os.path.join(_sdir, '..', 'data', 'force_field_quadratic.npz'))
        _COEF = {'dn': list(npz['c_dfn']), 'dfo': list(npz['c_dfo'])}
        idxs = npz['base_indices']
        fns = npz['base_Fn0']
        fos = npz['base_Fo0']
        _BASE_TABLE = [(int(idxs[i]), float(fns[i]), float(fos[i])) for i in range(len(idxs))]
    except FileNotFoundError:
        raise RuntimeError('未找到标定文件，请先运行 calibrate()')


def predict(dn, db):
    """预测偏移量 (dn, db) 下的力变化 (ΔFn, ΔFo)"""
    _load_if_needed()
    c_fn = _COEF['dn']
    c_fo = _COEF['dfo']
    dFn = c_fn[0]*dn + c_fn[1]*db + c_fn[2]*dn**2 + c_fn[3]*dn*db + c_fn[4]*db**2
    dFo = c_fo[0]*dn + c_fo[1]*db + c_fo[2]*dn**2 + c_fo[3]*dn*db + c_fo[4]*db**2
    return dFn, dFo


def get_base(p_idx):
    """查基准力值 Fn0(p_idx), Fo0(p_idx)
    p_idx: 0~499 的采样点索引
    """
    _load_if_needed()
    # 找最近的基准点
    best = min(_BASE_TABLE, key=lambda x: abs(x[0] - p_idx))
    return best[1], best[2]


def inverse(Fn_meas, Fo_meas, p_idx, max_iter=10, tol=1e-6):
    """从测量力反推偏移量

    Args:
        Fn_meas, Fo_meas: 传感器测到的有符号法向力、复法向力 (N)
        p_idx: 曲线进度索引 (0~499)，用于查基准力
    Returns:
        (dn, db) 偏移量 (mm)，或 (0, 0) 如果脱离接触
    """
    _load_if_needed()
    if abs(Fn_meas) < 0.5:  # 脱离接触，无法判断
        return 0.0, 0.0

    Fn0, Fo0 = get_base(p_idx)
    dFn = Fn_meas - Fn0
    dFo = Fo_meas - Fo0

    c_fn = _COEF['dn']
    c_fo = _COEF['dfo']

    x = np.zeros(2)
    for _ in range(max_iter):
        Fnp = c_fn[0]*x[0] + c_fn[1]*x[1] + c_fn[2]*x[0]**2 + c_fn[3]*x[0]*x[1] + c_fn[4]*x[1]**2
        Fop = c_fo[0]*x[0] + c_fo[1]*x[1] + c_fo[2]*x[0]**2 + c_fo[3]*x[0]*x[1] + c_fo[4]*x[1]**2
        J = np.array([
            [c_fn[0] + 2*c_fn[2]*x[0] + c_fn[3]*x[1],
             c_fn[1] + c_fn[3]*x[0] + 2*c_fn[4]*x[1]],
            [c_fo[0] + 2*c_fo[2]*x[0] + c_fo[3]*x[1],
             c_fo[1] + c_fo[3]*x[0] + 2*c_fo[4]*x[1]],
        ])
        try:
            delta = np.linalg.solve(J, [dFn - Fnp, dFo - Fop])
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta) < tol:
            break

    return float(x[0]), float(x[1])


# ── 自测 ──
if __name__ == '__main__':
    calibrate()

    # 随机验证
    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    ball = d['ball_center_500']
    cy = d['cyl_contact_y']
    cz = d['cyl_contact_z']
    cg = d['contact_geom'].sample_pts

    np.random.seed(42)
    errs = []
    for _ in range(50):
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
        dn_e, db_e = inverse(Fn_m, Fo_m, i)
        err = np.linalg.norm([dn_e - dn_t, db_e - db_t])
        errs.append(err)

    errs = np.array(errs)
    print(f'\n自测 {len(errs)} 点: 中位误差 {np.median(errs):.3f}mm  均值 {errs.mean():.3f}mm  max {errs.max():.3f}mm')
