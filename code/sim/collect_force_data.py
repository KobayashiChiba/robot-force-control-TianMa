"""
collect_force_data.py — 在 20 个接触位置采样力场数据

每个位置在 ±2mm 范围内均匀网格采样 (dn, db)，
用 sphere_contact_force 计算真实 Fn/Fo，
输出 (Fn, Fo, dn, db, pos_idx, snippet) 数据集。
"""
import sys, os, pickle
import numpy as np

_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))

from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho

# ── 参数 ──
RANGE = 2.0          # dn/db 采样范围 ±2mm
N_GRID = 21          # 每轴网格点数 (21×21=441 每组)
N_POS = 20           # 接触位置数

out_dir = os.path.join(_sdir, 'output')


def main():
    # 加载模型
    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)

    cy = d['cyl_contact_y']
    cz = d['cyl_contact_z']
    geom = d['contact_geom']          # GeomV2, sample_pts 2000点
    ball_ref = d['ball_center_500']   # (500,3)

    # 在接触曲线上均匀取 N_POS 个位置（用接触曲线的弧长，不是球刀轨迹）
    # 球刀中心和接触曲线对齐——用 argmin 找每个 ball_ref 对应的接触点
    contact_all = geom.sample_pts   # (2000,3)
    N_ball = len(ball_ref)

    # 均匀间隔选 N_POS 个位置
    pos_indices = np.linspace(0, N_ball - 1, N_POS, dtype=int)
    print(f"采样位置: {N_POS} 个 / {N_ball} 球刀参考点")
    print(f"网格: {N_GRID}×{N_GRID} = {N_GRID*N_GRID} 每组")
    print(f"预计总量: {N_POS * N_GRID * N_GRID} 条")
    print("=" * 50)

    axis_vals = np.linspace(-RANGE, RANGE, N_GRID)
    DN, DB = np.meshgrid(axis_vals, axis_vals)

    all_data = []
    skipped = 0

    for pos_idx, bi in enumerate(pos_indices):
        # 球刀参考中心
        bc0 = ball_ref[bi]

        # 找最近的接触点
        ci = np.argmin(np.linalg.norm(contact_all - bc0, axis=1))
        Pc = contact_all[ci]

        # 计算标架 {n, o=t×n}
        basis = compute_point_basis_ortho(Pc, geom)
        n = basis.normal
        o = basis.ortho

        # 状态诊断：无偏移时的基准力
        f0, a0 = sphere_contact_force(bc0, cz, cy)
        Fn0 = np.dot(f0, n)
        Fo0 = np.dot(f0, o)

        valid_count = 0
        for i in range(N_GRID):
            for j in range(N_GRID):
                dni = DN[i, j]
                dbj = DB[i, j]
                pos_offset = bc0 + dni * n + dbj * o
                F_vec, area = sphere_contact_force(pos_offset, cz, cy)
                Fm = np.linalg.norm(F_vec)

                if Fm < 0.5:  # 完全脱接触
                    skipped += 1
                    continue

                Fn = np.dot(F_vec, n)
                Fo = np.dot(F_vec, o)

                all_data.append({
                    'pos_idx': pos_idx,
                    'ball_idx': int(bi),
                    'contact_idx': int(ci),
                    'dn': float(dni),
                    'db': float(dbj),
                    'Fn': float(Fn),
                    'Fo': float(Fo),
                    'Fm': float(Fm),
                    'area': float(area),
                })
                valid_count += 1

        pct = (bi + 1) / N_ball * 100
        print(f"  [{pos_idx+1:2d}/{N_POS}] ball_idx={bi:3d}  p={pct:5.1f}%  "
              f"Fn0={Fn0:+.2f}N  Fo0={Fo0:+.2f}N  valid={valid_count}/{N_GRID*N_GRID}")

    print(f"\n总计: {len(all_data)} 有效点 (跳过 {skipped} 个脱接触)")
    print(f"数据维度: {len(all_data)} 条 × 10 字段")

    # 保存
    out_path = os.path.join(out_dir, 'force_sweep_data.npz')
    arrays = {k: np.array([d[k] for d in all_data]) for k in all_data[0]}
    np.savez(out_path, **arrays)
    print(f"已保存 {out_path}")

    # 快速统计
    Fn_arr = arrays['Fn']
    Fo_arr = arrays['Fo']
    dn_arr = arrays['dn']
    db_arr = arrays['db']
    print(f"\n统计:")
    print(f"  Fn: [{Fn_arr.min():.1f}, {Fn_arr.max():.1f}] 均值={Fn_arr.mean():.1f}N")
    print(f"  Fo: [{Fo_arr.min():.1f}, {Fo_arr.max():.1f}] 均值={Fo_arr.mean():.1f}N")
    print(f"  dn: [{dn_arr.min():.1f}, {dn_arr.max():.1f}]mm")
    print(f"  db: [{db_arr.min():.1f}, {db_arr.max():.1f}]mm")


if __name__ == '__main__':
    main()
