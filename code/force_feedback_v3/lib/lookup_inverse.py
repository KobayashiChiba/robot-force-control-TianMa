"""
lookup_inverse.py — 查表逆推 (Fn,Fo) → (dn,db)

用法:
    from .lookup_inverse import build_table, inverse

    # 建表（一次性，约15-30秒）
    table = build_table()  # → dict: {'Fn','Fo','dn','db'}
    # 或: table = build_table(n_pos=50, R=3, Ng=31)

    # 逆推（直接查表）
    dn, db = inverse(Fn_meas, Fo_meas, table)

特点: 无拟合假设，精度由采样密度决定，查表 O(logN)
"""
import numpy as np
from scipy.spatial import KDTree


def build_table(cy=None, cz=None, ball_ref=None, contact_pts=None,
                n_pos=50, R=3.0, Ng=31, verbose=True):
    """在多个位置采样 (dn,db) 网格，建立全局查表。

    Args:
        cy, cz: 标准圆柱 CylinderDef
        ball_ref: 球刀参考轨迹 (N,3)
        contact_pts: 接触曲线点 (M,3)
        n_pos: 采样位置数
        R: dn/db 范围 ±R mm
        Ng: 每轴网格点数 (Ng×Ng=每组点数)

    Returns:
        dict: {'Fn':(K,), 'Fo':(K,), 'dn':(K,), 'db':(K,), 'tree':KDTree}
    """
    if cy is None or cz is None:
        from . import load_cylinders, load_ball_ref
        from .simulator import Simulator
        from .force_mechanics import compute_point_basis_ortho

        cy, cz = load_cylinders()
        ball_ref, _ = load_ball_ref()
        sim = Simulator(cy, cz)
        contact_pts = sim.contact_pts

    from .sphere_contact import sphere_contact_force
    from .force_mechanics import compute_point_basis_ortho

    Npts = len(ball_ref)
    dv = np.linspace(-R, R, Ng)
    import time
    t0 = time.perf_counter()

    _all_Fn, _all_Fo, _all_dn, _all_db = [], [], [], []
    sim = None

    for p_i, p in enumerate(np.linspace(0, 1, n_pos + 1)[:n_pos]):
        i = int(p * (Npts - 1))
        bc0 = ball_ref[i]

        # 找最近接触点 + 标架
        idx = np.argmin(np.linalg.norm(contact_pts - bc0, axis=1))
        Pc = contact_pts[idx]

        # 需要 Simulator 来获取 contact_geom → 标架
        if sim is None:
            from .simulator import Simulator as _Sim
            sim = _Sim(cy, cz)

        basis = compute_point_basis_ortho(Pc, sim.contact_geom)

        n, o = basis.normal, basis.ortho
        for dni in dv:
            for dbj in dv:
                pos = bc0 + dni * n + dbj * o
                F, area = sphere_contact_force(pos, cz, cy)
                if area < 0.01:
                    continue
                _all_dn.append(float(dni))
                _all_db.append(float(dbj))
                _all_Fn.append(float(np.dot(F, n)))
                _all_Fo.append(float(np.dot(F, o)))

        if verbose and (p_i + 1) % 10 == 0:
            print(f'  [{p_i+1}/{n_pos}] 已采样 {len(_all_dn)} 有效点')

    elapsed = time.perf_counter() - t0
    Fn_arr = np.array(_all_Fn)
    Fo_arr = np.array(_all_Fo)
    dn_arr = np.array(_all_dn)
    db_arr = np.array(_all_db)

    # 建 KD-tree: 在 (Fn, Fo) 空间中
    tree = KDTree(np.column_stack([Fn_arr, Fo_arr]))

    table = {
        'Fn': Fn_arr, 'Fo': Fo_arr,
        'dn': dn_arr, 'db': db_arr,
        'tree': tree,
        'n_pos': n_pos, 'R': R, 'Ng': Ng,
    }

    if verbose:
        print(f'查表建立完成: {len(Fn_arr)} 有效点, 耗时 {elapsed:.1f}s')
        print(f'  Fn: [{Fn_arr.min():.1f}, {Fn_arr.max():.1f}]N')
        print(f'  Fo: [{Fo_arr.min():.1f}, {Fo_arr.max():.1f}]N')

    return table


def inverse(Fn_meas, Fo_meas, table):
    """查表逆推 (Fn,Fo) → (dn,db)
    ...
    """
    # 无力时返回固定搜索步长（沿法向推入，负值→err_n为正→推入方向）
    if abs(Fn_meas) < 0.5:
        return -0.5, 0.0

    tree = table['tree']
    dist, idx = tree.query([Fn_meas, Fo_meas], k=1)
    idx = int(idx)

    dn = float(table['dn'][idx])
    db = float(table['db'][idx])

    # 浅接触保护：|Fn|<2N 时不反推 db
    if abs(Fn_meas) < 2.0:
        return dn, 0.0

    return dn, db


# ── 全局缓存（避免重复建表）──
_GLOBAL_TABLE = None


def get_table(force_rebuild=False, cache_path=None):
    """获取全局查表（惰性建表，缓存到磁盘）"""
    global _GLOBAL_TABLE
    if _GLOBAL_TABLE is not None and not force_rebuild:
        return _GLOBAL_TABLE

    import pickle, os
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'lookup_table.pkl')

    if not force_rebuild and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            _GLOBAL_TABLE = pickle.load(f)
        # KDTree 不能在 pickle 中保存（scipy 限制），重建
        _GLOBAL_TABLE['tree'] = KDTree(
            np.column_stack([_GLOBAL_TABLE['Fn'], _GLOBAL_TABLE['Fo']]))
        return _GLOBAL_TABLE

    _GLOBAL_TABLE = build_table()
    # 保存（去掉 tree 因为 scipy KDTree 不可 pickle）
    save = {k: v for k, v in _GLOBAL_TABLE.items() if k != 'tree'}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(save, f)
    return _GLOBAL_TABLE


def inverse_cached(Fn_meas, Fo_meas):
    """带缓存的查表逆推（推荐接口）"""
    return inverse(Fn_meas, Fo_meas, get_table())


# ── 自测 ──
if __name__ == '__main__':
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
    Npts = len(ball_ref)

    print('=== 查表逆推测试 ===')
    print(f'位置数={Npts}, 接触点={len(ct)}')

    # 建表
    table = build_table(cy, cz, ball_ref, ct, n_pos=50, R=3, Ng=31)

    # 往返验证
    np.random.seed(42)
    errs_dn, errs_db = [], []
    import time
    t0 = time.perf_counter()
    for _ in range(500):
        p = np.random.uniform(0, 1)
        i = min(int(p * (Npts - 1)), Npts - 1)
        bc0 = ball_ref[i]
        idx = np.argmin(np.linalg.norm(ct - bc0, axis=1))
        basis = compute_point_basis_ortho(ct[idx], sim.contact_geom)
        n, o = basis.normal, basis.ortho
        dn_t = np.random.uniform(-2.5, 2.5)
        db_t = np.random.uniform(-2.5, 2.5)
        pos = bc0 + dn_t * n + db_t * o
        F, area = sphere_contact_force(pos, cz, cy)
        if area < 0.01:
            continue
        Fn_m = np.dot(F, n)
        Fo_m = np.dot(F, o)
        dn_e, db_e = inverse(Fn_m, Fo_m, table)
        errs_dn.append(abs(dn_e - dn_t))
        errs_db.append(abs(db_e - db_t))

    elapsed = time.perf_counter() - t0
    errs_dn = np.array(errs_dn)
    errs_db = np.array(errs_db)
    print(f'\n往返验证 ({len(errs_dn)}点, {elapsed:.1f}s):')
    print(f'  dn: med={np.median(errs_dn):.4f} mean={errs_dn.mean():.4f} P90={np.percentile(errs_dn,90):.4f}mm')
    print(f'  db: med={np.median(errs_db):.4f} mean={errs_db.mean():.4f} P90={np.percentile(errs_db,90):.4f}mm')

    # 边界测试
    tests = [
        (-8.0, 0.0, '目标'),
        (-0.3, 0.0, '零接触'),
        (-1.5, 0.0, '浅接触'),
        (-12.0, 2.0, '深右上'),
        (-12.0, -2.0, '深右下'),
    ]
    print(f'\n边界测试:')
    for fn, fo, label in tests:
        dn, db = inverse(fn, fo, table)
        print(f'  {label:>6} ({fn:5.1f},{fo:5.1f}) → dn={dn:.3f} db={db:.3f}')
