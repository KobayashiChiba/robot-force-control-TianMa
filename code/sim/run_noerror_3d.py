"""
run_noerror_3d.py — 无误差 V5 力控仿真 + 3D 图（含摩擦+噪声）
"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT = 0.005
MU = 0.2       # 库仑摩擦系数
SIGMA = 0.5    # 力噪声 std (N)
rng = np.random.RandomState(42)

# 加载
cy0, cz0 = load_standard_cylinders()
ctrl = ForceController(cy0, cz0)

# 参考曲线
geom0 = sample_intersection(cy0, cz0, n_samples=500)
pts0 = geom0.sample_pts
with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    ball_ref = pickle.load(f)['ball_center_500']

# 仿真
N_STEPS = 3000
traj, flog, flog_raw = [], [], []
flog_fric, flog_noise = [], []
pos = ctrl.ball_ref[0]
t0 = time.perf_counter()

v_prev = np.zeros(3)  # 初始无速度
for step in range(N_STEPS):
    s_cur = step / (N_STEPS - 1)
    
    # 原始接触力
    F_raw, _ = sphere_contact_force(pos, cz0, cy0)
    flog_raw.append(np.linalg.norm(F_raw))
    
    # 标架（与控制器一致：最近接触点）
    P_ct = ctrl._nearest_contact(pos)
    basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
    n, t = basis.normal, basis.tangent
    
    Fn = np.dot(F_raw, n)
    
    # 摩擦力：垂直于接触力方向（真实的接触切平面）
    f_dir = F_raw / np.linalg.norm(F_raw)
    v_tang = v_prev - np.dot(v_prev, f_dir) * f_dir
    v_norm = np.linalg.norm(v_tang)
    if v_norm > 1e-6:
        F_fric = MU * abs(Fn) * (-v_tang / v_norm)
    else:
        F_fric = np.zeros(3)
    
    # 高斯噪声（三方向独立）
    F_noise = rng.randn(3) * SIGMA
    
    F_vec = F_raw + F_fric + F_noise
    flog.append(np.linalg.norm(F_vec))
    flog_fric.append(np.linalg.norm(F_fric))
    flog_noise.append(np.linalg.norm(F_noise))
    
    v_3d = ctrl.step(F_vec, s_cur, pos, N_STEPS, DT)
    pos = pos + v_3d * DT
    v_prev = v_3d.copy()
    traj.append(pos.copy())

elapsed = time.perf_counter() - t0

traj = np.array(traj)
flog = np.array(flog)
flog_raw = np.array(flog_raw)
flog_fric = np.array(flog_fric)
flog_noise = np.array(flog_noise)
last500 = flog[-500:]
last500_raw = flog_raw[-500:]
print(f"|F_total| = {np.mean(last500):.2f} +/- {np.std(last500):.2f} N")
print(f"|F_raw|   = {np.mean(last500_raw):.2f} +/- {np.std(last500_raw):.2f} N")
print(f"摩擦      = {np.mean(flog_fric):.2f} +/- {np.std(flog_fric):.2f} N")
print(f"噪声      = {np.mean(flog_noise):.2f} +/- {np.std(flog_noise):.2f} N")
print(f"耗时: {elapsed:.1f}s")

gap = np.linalg.norm(traj[-1] - traj[0])
print(f"首尾距离 = {gap:.3f} mm  (参考弧长 {ctrl.L:.1f} mm)")

# === 3D 图 ===
fig = plt.figure(figsize=(16, 12))
ax3d = fig.add_subplot(221, projection='3d')
ax_f = fig.add_subplot(222)
ax_f_start = fig.add_subplot(223)
ax_f_detail = fig.add_subplot(224)

# 3D
ax3d.plot(pts0[:, 0], pts0[:, 1], pts0[:, 2], 'gray', ls='--', lw=1, alpha=0.5, label='接触曲线')
ax3d.plot(ball_ref[:, 0], ball_ref[:, 1], ball_ref[:, 2], 'green', lw=0.6, alpha=0.4, label='球刀参考')
ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'blue', lw=1.2, label='力控轨迹')
ax3d.scatter(*traj[0], c='cyan', s=60, marker='o', zorder=5, label='起点')
ax3d.scatter(*traj[-1], c='red', s=60, marker='s', zorder=5, label='终点')
ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
ax3d.set_title(f'V5 无误差 + 摩擦+噪声 (μ={MU}, σ={SIGMA}N)\n|F|={np.mean(last500):.2f}±{np.std(last500):.2f}N  首尾距={gap:.2f}mm')
ax3d.legend(fontsize=8)

# 力全程
ax_f.plot(flog_raw, 'gray', lw=0.5, alpha=0.5, label='原始力')
ax_f.plot(flog, 'b-', lw=0.8, label='含摩擦+噪声')
ax_f.axhline(8.0, color='gray', ls='--', lw=0.5)
ax_f.set_ylabel('|F| (N)'); ax_f.set_xlabel('Step')
ax_f.legend(fontsize=7); ax_f.grid(alpha=0.3)

# 力前100步
ax_f_start.plot(flog_raw[:100], 'gray', lw=0.8, alpha=0.5)
ax_f_start.plot(flog[:100], 'b-', lw=0.8)
ax_f_start.axhline(8.0, color='gray', ls='--', lw=0.5)
ax_f_start.set_ylabel('|F| (N)'); ax_f_start.set_xlabel('Step')
ax_f_start.set_title('力收敛: 前100步'); ax_f_start.grid(alpha=0.3)

# 力最后200步
ax_f_detail.plot(range(len(flog)-200, len(flog)), flog_raw[-200:], 'gray', lw=0.8, alpha=0.5, label='原始')
ax_f_detail.plot(range(len(flog)-200, len(flog)), flog[-200:], 'b-', lw=0.8, label='含摩擦+噪声')
ax_f_detail.axhline(8.0, color='gray', ls='--', lw=0.5)
ax_f_detail.set_ylabel('|F| (N)'); ax_f_detail.set_xlabel('Step')
ax_f_detail.set_title(f'力稳态: 最后200步 (μ={MU}, σ={SIGMA}N)')
ax_f_detail.legend(fontsize=7); ax_f_detail.grid(alpha=0.3)

fig.suptitle(f'V5 力控 — 无误差 + 摩擦(μ={MU}) + 噪声(σ={SIGMA}N)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_noerror_fric_noise.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
