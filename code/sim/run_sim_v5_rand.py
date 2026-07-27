"""
run_sim_v5_rand.py — V5 随机误差测试，独立 3D 图（含标准/误差接触曲线+球刀参考/实际轨迹）
"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import (
    ForceController, load_standard_cylinders, translate_cz
)
from sphere_contact import sphere_contact_force
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT = 0.005
N_STEPS = 3000


def run(cy_std, cz_std, cy_err, cz_err, label):
    ctrl = ForceController(cy_std, cz_std)
    pos = ctrl._ball_ref(0)
    traj, flog = [], []
    t0 = time.perf_counter()
    for step in range(N_STEPS):
        s_cur = step / (N_STEPS - 1)
        F_vec, _ = sphere_contact_force(pos, cz_err, cy_err)
        v_3d = ctrl.step(F_vec, s_cur, pos, N_STEPS, DT)
        pos = pos + v_3d * DT
        traj.append(pos.copy())
        flog.append(np.linalg.norm(F_vec))
    elapsed = time.perf_counter() - t0
    flog = np.array(flog)
    fm, fs = np.mean(flog[-500:]), np.std(flog[-500:])
    print(f"  [{label}] |F|={fm:.2f}±{fs:.2f}N  ({elapsed:.1f}s)")
    return np.array(traj), flog, fm, fs


def plot_3d(cy0, cz0, cy_err, cz_err, traj, flog, fm, fs, label, filename):
    # 各曲线
    contact_std = sample_intersection(cy0, cz0, n_samples=500).sample_pts
    contact_err = sample_intersection(cy_err, cz_err, n_samples=500).sample_pts

    # 误差圆柱下的球刀参考（误差圆柱交线按球刀半径偏移 ≈ 球心参考_err）
    # 用接触曲线沿法向偏移 R=4.0mm
    from force_mechanics_v2 import compute_point_basis_ortho
    cg_err = sample_intersection(cy_err, cz_err, n_samples=500)
    ball_ref_err = np.zeros_like(cg_err.sample_pts)
    for i in range(len(cg_err.sample_pts)):
        basis = compute_point_basis_ortho(cg_err.sample_pts[i], cg_err)
        ball_ref_err[i] = cg_err.sample_pts[i] - 4.0 * basis.normal

    # 标准球刀参考
    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        ball_ref = pickle.load(f)['ball_center_500']

    fig = plt.figure(figsize=(18, 12))
    ax3d = fig.add_subplot(2, 2, (1, 2), projection='3d')

    ax3d.plot(contact_std[:, 0], contact_std[:, 1], contact_std[:, 2],
              'gray', ls='--', lw=1, alpha=0.5, label='标准接触曲线')
    ax3d.plot(contact_err[:, 0], contact_err[:, 1], contact_err[:, 2],
              'red', ls='--', lw=1, alpha=0.5, label='误差接触曲线')
    ax3d.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2],
              'green', lw=0.6, alpha=0.3, label='标准球刀参考')
    ax3d.plot(ball_ref_err[:, 0], ball_ref_err[:, 1], ball_ref_err[:, 2],
              'orange', lw=0.5, alpha=0.3, ls=':', label='误差球刀参考')
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
              'blue', lw=1.2, label='力控轨迹')
    ax3d.scatter(*traj[0], c='cyan', s=40, marker='o', zorder=5, label='起点')
    ax3d.scatter(*traj[-1], c='blue', s=40, marker='s', zorder=5, label='终点')

    ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
    ax3d.set_title(f'{label} — 3D 轨迹'); ax3d.legend(fontsize=7, loc='upper right')

    # 力收敛
    ax_f = fig.add_subplot(2, 2, 3)
    ax_f.plot(flog, 'b-', lw=0.5)
    ax_f.axhline(8.0, color='gray', ls='--', lw=0.8)
    ax_f.axhline(fm, color='red', ls=':', lw=0.8)
    ax_f.set_ylabel('|F| (N)'); ax_f.set_title(f'力收敛 |F|={fm:.2f}±{fs:.2f}N')
    ax_f.grid(alpha=0.3)

    # 分布
    ax_h = fig.add_subplot(2, 2, 4)
    ax_h.hist(flog[-500:], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax_h.axvline(8.0, color='gray', ls='--', lw=1)
    ax_h.axvline(fm, color='red', ls='-', lw=1)
    ax_h.set_xlabel('|F| (N)'); ax_h.set_ylabel('频次')
    ax_h.set_title(f'{label} 力分布 (稳态)'); ax_h.grid(alpha=0.3, axis='y')

    fig.suptitle(f'V5 — {label}', fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f'  已保存 {filename}')
    plt.close(fig)


def main():
    cy0, cz0 = load_standard_cylinders()
    out_dir = os.path.join(_sdir, 'output')

    print(f"V5 随机误差测试 (10 组)")
    print("=" * 60)

    results = []
    for seed in range(10):
        np.random.seed(seed)
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        dz = np.random.uniform(-0.5, 0.5)
        label = f'RAND#{seed} ({dx:+.2f},{dy:+.2f},{dz:+.2f})'
        czr = translate_cz(cz0, dx=dx, dy=dy, dz=dz)

        print(f"\n[{label}]")
        traj, flog, fm, fs = run(cy0, cz0, cy0, czr, label)
        fname = os.path.join(out_dir, f'v5_rand{seed:02d}.png')
        plot_3d(cy0, cz0, cy0, czr, traj, flog, fm, fs, label, fname)
        results.append((label, fm, fs))

    print("\n" + "=" * 60)
    print("汇总:")
    for label, fm, fs in results:
        print(f"  {label:<40} {fm:6.2f}±{fs:.2f}N")


if __name__ == '__main__':
    main()
