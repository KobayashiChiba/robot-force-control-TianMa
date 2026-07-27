"""
force_field_quadratic.py — 力场全局二次模型

用 lib 的 compute_point_basis_ortho → {normal, ortho=t×n} 正交分解。
所有曲线位置混合标定，不分进度。

接口:
  calibrate()            → 全局拟合 + 保存 + 出验证图
  predict(dn, db)        → 预测 (Fn, Fo)
  inverse(Fn, Fo)        → 从测量力反推偏移量，Newton 迭代
"""

import sys, os, pickle
import numpy as np

_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, _sdir)

from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho

_COEF = None    # {'dn': [c0..c5], 'dfo': [c0..c5]}


def calibrate(save_name='force_field_global.npz'):
    """全局混合标定：20 个位置统一拟合 Fn(dn,db) 和 Fo(dn,db)。"""
    global _COEF

    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)

    ball = d['ball_center_500']
    cy = d['cyl_contact_y']
    cz = d['cyl_contact_z']
    geom = d['contact_geom']       # GeomV2，2000点接触曲线
    Npts = len(ball)

    R, Ng, N_pos = 2, 21, 20
    dn = np.linspace(-R, R, Ng)
    db = np.linspace(-R, R, Ng)

    all_Fn, all_Fo, AA = [], [], []

    for p in np.linspace(0, 0.99, N_pos):
        i = min(int(p * Npts), Npts - 1)
        bc0 = ball[i]
        idx = np.argmin(np.linalg.norm(geom.sample_pts - bc0, axis=1))
        Pc = geom.sample_pts[idx]
        basis = compute_point_basis_ortho(Pc, geom)
        n = basis.normal
        o = basis.ortho

        for dni in dn:
            for dbj in db:
                f, _ = sphere_contact_force(bc0 + dni * n + dbj * o, cz, cy)
                if np.linalg.norm(f) > 0.5:
                    all_Fn.append(np.dot(f, n))
                    all_Fo.append(np.dot(f, o))
                    AA.append([1.0, dni, dbj, dni ** 2, dni * dbj, dbj ** 2])

    all_Fn = np.array(all_Fn)
    all_Fo = np.array(all_Fo)
    AA = np.array(AA)

    c_fn, _, _, _ = np.linalg.lstsq(AA, all_Fn, rcond=None)
    c_fo, _, _, _ = np.linalg.lstsq(AA, all_Fo, rcond=None)

    _COEF = {'dn': list(c_fn), 'dfo': list(c_fo)}

    save_path = os.path.join(_sdir, '..', 'data', save_name)
    np.savez(save_path, c_fn=c_fn, c_fo=c_fo)

    # --- 验证图 ---
    Fn_pred = AA @ c_fn
    Fo_pred = AA @ c_fo
    err_n = np.abs(Fn_pred - all_Fn)
    err_o = np.abs(Fo_pred - all_Fo)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 行1：Fn / Fo / 残差
    for ax, title, pred, actual, err, color in [
        (axes[0, 0], 'Fn', Fn_pred, all_Fn, err_n, '#3498db'),
        (axes[0, 1], 'Fo', Fo_pred, all_Fo, err_o, '#e74c3c'),
    ]:
        ax.scatter(actual, pred, s=1, alpha=0.3, color=color)
        mn = min(actual.min(), pred.min())
        mx = max(actual.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=0.5)
        ax.set_xlabel('实际 (N)'); ax.set_ylabel('预测 (N)')
        ax.set_title(f'{title}  中位误差 {np.median(err):.3f}N')
        ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.hist(err_n, bins=50, alpha=0.6, color='#3498db', label=f'Fn')
    ax.hist(err_o, bins=50, alpha=0.6, color='#e74c3c', label=f'Fo')
    ax.legend(fontsize=8); ax.set_xlabel('绝对误差 (N)'); ax.set_title('残差分布')

    # 行2：力场热力图
    R2, Ng2 = 3, 50
    dn2 = np.linspace(-R2, R2, Ng2); db2 = np.linspace(-R2, R2, Ng2)
    DN, DB = np.meshgrid(dn2, db2)
    f_fn = np.zeros_like(DN); f_fo = np.zeros_like(DN)
    for ii in range(Ng2):
        for jj in range(Ng2):
            x = np.array([1, DN[ii, jj], DB[ii, jj], DN[ii, jj]**2, DN[ii, jj]*DB[ii, jj], DB[ii, jj]**2])
            f_fn[ii, jj] = x @ c_fn
            f_fo[ii, jj] = x @ c_fo

    ax = axes[1, 0]; im = ax.contourf(DN, DB, f_fn, levels=20, cmap='Blues')
    plt.colorbar(im, ax=ax, label='Fn (N)'); ax.set_xlabel('dn (mm)'); ax.set_ylabel('db (mm)')
    ax.set_title('Fn 力场 (标定)'); ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax = axes[1, 1]; im = ax.contourf(DN, DB, f_fo, levels=20, cmap='Reds')
    plt.colorbar(im, ax=ax, label='Fo (N)'); ax.set_xlabel('dn (mm)'); ax.set_ylabel('db (mm)')
    ax.set_title('Fo 力场 (标定)'); ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)

    ax = axes[1, 2]; ax.axis('off')
    td = [
        ['系数', 'Fn', 'Fo'],
        ['c0', f'{c_fn[0]:.3f}', f'{c_fo[0]:.3f}'],
        ['dn', f'{c_fn[1]:.3f}', f'{c_fo[1]:.3f}'],
        ['db', f'{c_fn[2]:.3f}', f'{c_fo[2]:.3f}'],
        ['dn^2', f'{c_fn[3]:.3f}', f'{c_fo[3]:.3f}'],
        ['dn*db', f'{c_fn[4]:.3f}', f'{c_fo[4]:.3f}'],
        ['db^2', f'{c_fn[5]:.3f}', f'{c_fo[5]:.3f}'],
        ['', '', ''],
        ['中位误差', f'{np.median(err_n):.3f}N', f'{np.median(err_o):.3f}N'],
    ]
    tbl = ax.table(cellText=td, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    ax.set_title(f'全局混合标定 {len(all_Fn)} 点', y=0.7)

    fig.suptitle('力场全局二次模型标定 (正交基底)', fontsize=14)
    fig.tight_layout()
    out = os.path.join(_sdir, '..', 'sim', 'output', 'force_field_calibration.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f'全局混合标定: {len(all_Fn)} 采样点')
    print(f'  Fn = {c_fn[0]:.3f} {c_fn[1]:+.3f}*dn {c_fn[2]:+.3f}*db {c_fn[3]:+.3f}*dn^2 {c_fn[4]:+.3f}*dn*db {c_fn[5]:+.3f}*db^2')
    print(f'  Fo = {c_fo[0]:.3f} {c_fo[1]:+.3f}*dn {c_fo[2]:+.3f}*db {c_fo[3]:+.3f}*dn^2 {c_fo[4]:+.3f}*dn*db {c_fo[5]:+.3f}*db^2')
    print(f'  Fn 中位误差 {np.median(err_n):.3f}N  Fo 中位误差 {np.median(err_o):.3f}N')
    print(f'已保存 {out}')

    return _COEF


def _load():
    global _COEF
    if _COEF is not None:
        return
    npz = np.load(os.path.join(_sdir, '..', 'data', 'force_field_global.npz'))
    _COEF = {'dn': list(npz['c_fn']), 'dfo': list(npz['c_fo'])}


def predict(dn, db):
    _load()
    c_fn = _COEF['dn']; c_fo = _COEF['dfo']
    x = np.array([1, dn, db, dn**2, dn*db, db**2])
    return float(x @ c_fn), float(x @ c_fo)


def inverse(Fn_meas, Fo_meas, max_iter=15, tol=1e-6):
    _load()
    if abs(Fn_meas) < 0.5:
        return 0.0, 0.0

    c_fn = _COEF['dn']; c_fo = _COEF['dfo']
    x = np.zeros(2)

    for _ in range(max_iter):
        dn, db = x
        Fnp = c_fn[0] + c_fn[1]*dn + c_fn[2]*db + c_fn[3]*dn**2 + c_fn[4]*dn*db + c_fn[5]*db**2
        Fop = c_fo[0] + c_fo[1]*dn + c_fo[2]*db + c_fo[3]*dn**2 + c_fo[4]*dn*db + c_fo[5]*db**2
        J = np.array([
            [c_fn[1] + 2*c_fn[3]*dn + c_fn[4]*db, c_fn[2] + c_fn[4]*dn + 2*c_fn[5]*db],
            [c_fo[1] + 2*c_fo[3]*dn + c_fo[4]*db, c_fo[2] + c_fo[4]*dn + 2*c_fo[5]*db],
        ])
        try:
            delta = np.linalg.solve(J, [Fn_meas - Fnp, Fo_meas - Fop])
        except np.linalg.LinAlgError:
            break
        x += delta
        if np.linalg.norm(delta) < tol:
            break

    return float(x[0]), float(x[1])


if __name__ == '__main__':
    calibrate()
