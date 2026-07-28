"""
run_sim_v5.py — V5 力控仿真运行脚本

用法: python run_sim_v5.py [误差类型] [误差大小]
  误差类型: x/y/z/rotate_x/rotate_y/rotate_z
  默认: x 1.5

示例:
  python run_sim_v5.py                  # X+1.5mm 平移
  python run_sim_v5.py rotate_z 5       # Z轴旋转5°
  python run_sim_v5.py x 3              # X+3mm
"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import (
    ForceController, load_standard_cylinders, translate_cz, rotate_cz
)
from sphere_contact import sphere_contact_force
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT = 0.005


def run(cy_std, cz_std, cy_err, cz_err, label, n_steps=3000):
    ctrl = ForceController(cy_std, cz_std)
    L = ctrl.L
    pos = ctrl.ball_ref[0]

    traj, flog = [], []
    t0 = time.perf_counter()

    for step in range(n_steps):
        s_cur = step / (n_steps - 1)
        F_vec, _ = sphere_contact_force(pos, cz_err, cy_err)
        v_3d = ctrl.step(F_vec, s_cur, pos, n_steps, DT)
        pos = pos + v_3d * DT
        traj.append(pos.copy())
        flog.append(np.linalg.norm(F_vec))

    elapsed = time.perf_counter() - t0
    flog = np.array(flog)
    last500 = flog[-500:] if len(flog) >= 500 else flog
    print(f"  [{label}] |F|={np.mean(last500):.2f}+/-{np.std(last500):.2f}N  ({elapsed:.1f}s)")
    return np.array(traj), flog


def plot(cy0, cz0, cy1, cz1, traj0, flog0, traj1, flog1, err_label):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14),
                              subplot_kw={'projection': None})
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')

    # 参考曲线
    geom0 = sample_intersection(cy0, cz0, n_samples=500)
    pts0 = geom0.sample_pts
    ball_ref = None
    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        ball_ref = pickle.load(f)['ball_center_500']

    ax3d.plot(pts0[:, 0], pts0[:, 1], pts0[:, 2], 'gray', ls='--', lw=1, alpha=0.5,
              label='接触曲线')
    ax3d.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2], 'green', lw=0.6, alpha=0.4,
              label='球刀参考')

    ax3d.plot(traj0[:, 0], traj0[:, 1], traj0[:, 2], 'blue', lw=1.2, label='力控(无误差)')
    ax3d.scatter(*traj0[0], c='cyan', s=40, marker='o', zorder=5)
    ax3d.scatter(*traj0[-1], c='blue', s=40, marker='s', zorder=5)

    ax3d.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], 'red', lw=1.2, label=f'力控({err_label})')
    ax3d.scatter(*traj1[0], c='orange', s=40, marker='o', zorder=5)
    ax3d.scatter(*traj1[-1], c='red', s=40, marker='s', zorder=5)

    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title('V5 3D轨迹'); ax3d.legend(fontsize=7)

    # 力时序
    ax_f = axes[0][1]
    ax_f.plot(flog0, 'b-', lw=0.8, alpha=0.7, label='无误差')
    ax_f.plot(flog1, 'r-', lw=0.8, alpha=0.7, label=err_label)
    ax_f.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax_f.set_ylabel('|F| (N)'); ax_f.set_title('力控收敛'); ax_f.grid(alpha=0.3); ax_f.legend(fontsize=7)

    # 无误差特写
    ax_f0 = axes[1][0]
    ax_f0.plot(flog0, 'b-', lw=0.5)
    ax_f0.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax_f0.set_ylabel('|F| (N)'); ax_f0.set_title('无误差 |F|'); ax_f0.grid(alpha=0.3)

    # 误差特写
    ax_f1 = axes[1][1]
    ax_f1.plot(flog1, 'r-', lw=0.5)
    ax_f1.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax_f1.set_ylabel('|F| (N)'); ax_f1.set_title(f'{err_label} |F|'); ax_f1.grid(alpha=0.3)

    fig.suptitle(f'V5 力控仿真 ({err_label})', fontsize=14)
    fig.tight_layout()
    out = os.path.join(_sdir, 'output', 'force_control_v5.png')
    fig.savefig(out, dpi=150)
    print(f'已保存 {out}')
    plt.close(fig)


def main():
    err_type = sys.argv[1] if len(sys.argv) > 1 else 'x'
    err_val = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5

    print(f"V5 力控仿真 ({err_type}={err_val})")
    print("=" * 50)

    cy0, cz0 = load_standard_cylinders()

    # 生成误差圆柱
    if err_type == 'x':
        cz_err = translate_cz(cz0, dx=err_val)
        label = f'X+{err_val}mm'
    elif err_type == 'y':
        cz_err = translate_cz(cz0, dy=err_val)
        label = f'Y+{err_val}mm'
    elif err_type == 'z':
        cz_err = translate_cz(cz0, dz=err_val)
        label = f'Z+{err_val}mm'
    elif err_type.startswith('rotate'):
        axis = err_type.split('_')[1]
        cz_err = rotate_cz(cz0, axis, err_val)
        label = f'{axis}旋转{err_val}°'
    else:
        print(f"未知误差类型: {err_type}")
        sys.exit(1)

    cy_err = cy0  # Y圆柱不做误差

    print(f"\n[1] 无误差仿真...")
    t0, f0 = run(cy0, cz0, cy0, cz0, '无误差')

    print(f"\n[2] {label}仿真...")
    t1, f1 = run(cy0, cz0, cy_err, cz_err, label)

    print(f"\n[3] 绘图...")
    plot(cy0, cz0, cy_err, cz_err, t0, f0, t1, f1, label)


if __name__ == '__main__':
    main()
