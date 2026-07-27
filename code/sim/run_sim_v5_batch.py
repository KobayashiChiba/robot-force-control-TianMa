"""
run_sim_v5_batch.py — V5 批量误差测试，每组独立 3D 图

误差列表: 无误差, X±1.5, Y±1.5, Z±1.5, Z旋转±5°, 随机±0.5mm×3组
"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import (
    ForceController, load_standard_cylinders, translate_cz, rotate_cz
)
from sphere_contact import sphere_contact_force
from cylinder_geometry_v2 import sample_intersection
from cylinder_def import CylinderDef

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
    last500 = flog[-500:]
    print(f"  [{label}] |F|={np.mean(last500):.2f}±{np.std(last500):.2f}N  ({elapsed:.1f}s)")
    return np.array(traj), flog


def plot_3d(cy0, cz0, traj, flog, label, filename):
    """独立 3D 图：轨迹 + 力收敛 + 统计"""
    ball_ref = None
    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        ball_ref = pickle.load(f)['ball_center_500']
    geom0 = sample_intersection(cy0, cz0, n_samples=500)
    pts0 = geom0.sample_pts

    fig = plt.figure(figsize=(16, 12))

    # 3D 轨迹
    ax3d = fig.add_subplot(2, 2, (1, 2), projection='3d')
    ax3d.plot(pts0[:, 0], pts0[:, 1], pts0[:, 2], 'gray', ls='--', lw=1, alpha=0.3, label='接触曲线')
    ax3d.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2], 'green', lw=0.5, alpha=0.3, label='球刀参考')
    ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'blue', lw=1.2, label='力控轨迹')
    ax3d.scatter(*traj[0], c='cyan', s=40, marker='o', zorder=5, label='起点')
    ax3d.scatter(*traj[-1], c='blue', s=40, marker='s', zorder=5, label='终点')
    ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
    ax3d.set_title(f'{label} — 3D 轨迹'); ax3d.legend(fontsize=7)

    # 力收敛
    last500 = 500
    fm = np.mean(flog[-last500:])
    fs = np.std(flog[-last500:])

    ax_f = fig.add_subplot(2, 2, 3)
    ax_f.plot(flog, 'b-', lw=0.5)
    ax_f.axhline(8.0, color='gray', ls='--', lw=0.8)
    ax_f.axhline(fm, color='red', ls=':', lw=0.8)
    ax_f.set_ylabel('|F| (N)'); ax_f.set_title(f'力收敛 |F|={fm:.2f}±{fs:.2f}N')
    ax_f.grid(alpha=0.3)

    # 柱状图：最后 500 步分布
    ax_h = fig.add_subplot(2, 2, 4)
    ax_h.hist(flog[-last500:], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax_h.axvline(8.0, color='gray', ls='--', lw=1)
    ax_h.axvline(fm, color='red', ls='-', lw=1)
    ax_h.set_xlabel('|F| (N)'); ax_h.set_ylabel('频次')
    ax_h.set_title(f'{label} 力分布 (稳态)'); ax_h.grid(alpha=0.3, axis='y')

    fig.suptitle(f'V5 力控仿真 — {label}', fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f'  已保存 {filename}')
    plt.close(fig)


def main():
    cy0, cz0 = load_standard_cylinders()

    cases = [
        ('无误差', cy0, cz0),
        ('X+1.5mm', cy0, translate_cz(cz0, dx=1.5)),
        ('X-1.5mm', cy0, translate_cz(cz0, dx=-1.5)),
        ('Y+1.5mm', cy0, translate_cz(cz0, dy=1.5)),
        ('Y-1.5mm', cy0, translate_cz(cz0, dy=-1.5)),
        ('Z+1.5mm', cy0, translate_cz(cz0, dz=1.5)),
        ('Z旋转+5°', cy0, rotate_cz(cz0, 'z', 5)),
        ('Z旋转-5°', cy0, rotate_cz(cz0, 'z', -5)),
    ]

    # 加 3 组随机误差
    for seed in range(3):
        np.random.seed(seed)
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        dz = np.random.uniform(-0.5, 0.5)
        czr = translate_cz(cz0, dx=dx, dy=dy, dz=dz)
        cases.append((f'随机±0.5mm #{seed} ({dx:.2f},{dy:.2f},{dz:.2f})', cy0, czr))

    out_dir = os.path.join(_sdir, 'output')
    print(f"V5 批量误差测试 ({len(cases)} 组)")
    print("=" * 60)

    results = []
    for label, cy_e, cz_e in cases:
        print(f"\n[{label}]")
        traj, flog = run(cy0, cz0, cy_e, cz_e, label)
        fname = label.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_').replace('°', 'deg')
        fname = os.path.join(out_dir, f'v5_{fname}.png')
        plot_3d(cy0, cz0, traj, flog, label, fname)
        results.append((label, np.mean(flog[-500:]), np.std(flog[-500:])))

    print("\n" + "=" * 60)
    print("汇总:")
    print(f"{'误差工况':<30} {'|F|均值':>8} {'std':>8}")
    print("-" * 48)
    for label, fm, fs in results:
        print(f"{label:<30} {fm:7.2f}N {fs:7.2f}N")


if __name__ == '__main__':
    main()
