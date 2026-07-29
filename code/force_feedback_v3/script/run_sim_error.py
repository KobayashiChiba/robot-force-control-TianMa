"""
run_sim_error.py — 有误差闭环仿真 + 出图

用法: python script/run_sim_error.py <seed>

误差模型:
  12参数 ±1mm: Z/Y 圆柱各两端点独立 XYZ 偏移，通过 generate_error_cylinders 生成。
  控制器不知情（用标准圆柱），simulator 用误差圆柱产生力。

停圈:
  角度法：绕 X 轴，YZ 投影用 contact_pts 均值中心，abs(累计Δθ) ≥ 2π 停。

出图 (1×4 布局):
  1. 3D 轨迹  — contact std(灰虚线) / contact actual(绿实线) / ball ref(蓝点线) / ball actual(红实线)
  2. YZ 投影  — 同上，as_pect='equal'
  3. Fn vs step — ylim [-20, 0]，target -8N 虚线
  4. Fo vs step — ylim [-5, 5]

图例命名:
  参考图:     ball std(标准球心) vs ball ref(无误差无摩擦无噪声参考轨迹)
  无误差仿真: ball ref(参考轨迹) vs ball actual(仿真+摩擦噪声)
  有误差仿真: ball ref(参考轨迹) vs ball actual(仿真+摩擦噪声+几何误差)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import (load_cylinders, load_ball_ref,
                                   perturb_endpoints, generate_error_cylinders)
from force_feedback_v3.lib.simulator import Simulator
from force_feedback_v3.lib.controller import ForceController
from force_feedback_v3.lib.cylinder_geometry import sample_intersection
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# ── 参数 ──
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
np.random.seed(seed)

cy, cz = load_cylinders()
ball_ref, L = load_ball_ref()

# 12参数误差：每个端点 ±1mm
rng = np.random.RandomState(seed)
cz_err, cy_err = generate_error_cylinders(cy, cz, rng)

sim = Simulator(cy, cz, mu=0.2, sigma=0.5)
ctrl = ForceController(ball_ref, L, sim.contact_geom)

DT = 0.005; N_STEPS = 2000; MAX_STEPS = 2 * N_STEPS
cy0 = np.mean(sim.contact_pts[:, 1])
cz0 = np.mean(sim.contact_pts[:, 2])

pos = ball_ref[0].copy(); v_prev = np.zeros(3)
log_pos = []; log_Fn = []; log_Fo = []; log_dn = []; log_do = []
log_dn_true = []; log_db_true = []; accum = 0.0
theta_prev = np.arctan2(pos[2]-cz0, pos[1]-cy0)

# 预计算误差接触曲线（仅用于标架和绘图）
contact_err_geom = sample_intersection(cy_err, cz_err, n_samples=2000)
contact_err_pts = contact_err_geom.sample_pts

R_EFF = R_BALL - 0.4  # 球刀半径 - 理想切入深度 = 3.8mm

for k in range(MAX_STEPS):
    F_meas, _, _, _, basis = sim.step(pos, v_prev, cy_err=cy_err, cz_err=cz_err)
    v_3d = ctrl.step(F_meas, pos, N_STEPS, DT)
    pos += v_3d * DT; v_prev = v_3d
    log_pos.append(pos.copy())
    log_Fn.append(np.dot(F_meas, basis.normal))
    log_Fo.append(np.dot(F_meas, basis.ortho))
    P_ref = ctrl._nearest_ball_ref(pos)
    log_dn.append(np.dot(pos - P_ref, basis.normal))
    log_do.append(np.dot(pos - P_ref, basis.ortho))

    # 真实偏移: 沿误差接触曲线法向量 n，球心在 -n 方向（空气侧）
    # 理想球心 = Pc + n*R_EFF, 当前球心偏理想 = dot(pos-Pc, n) + R_EFF
    i_err = np.argmin(np.linalg.norm(contact_err_pts - pos, axis=1))
    Pc_err = contact_err_pts[i_err]
    basis_err = compute_point_basis_ortho(Pc_err, contact_err_geom)
    n_e, o_e = basis_err.normal, basis_err.ortho
    log_dn_true.append(np.dot(pos - Pc_err, n_e) + R_EFF)
    log_db_true.append(np.dot(pos - Pc_err, o_e))

    theta = np.arctan2(pos[2]-cz0, pos[1]-cy0)
    dtheta = theta - theta_prev
    if dtheta > np.pi: dtheta -= 2*np.pi
    elif dtheta < -np.pi: dtheta += 2*np.pi
    accum += dtheta; theta_prev = theta
    if abs(accum) >= 2*np.pi: break

log_pos = np.array(log_pos); log_Fn = np.array(log_Fn); log_Fo = np.array(log_Fo)
log_dn = np.array(log_dn); log_do = np.array(log_do)
log_dn_true = np.array(log_dn_true); log_db_true = np.array(log_db_true)

n1 = np.sum((np.abs(log_dn) > 1.0) | (np.abs(log_do) > 1.0))
n2 = np.sum((np.abs(log_dn) > 2.0) | (np.abs(log_do) > 2.0))

# 误差圆柱下的真实接触曲线
sim_err = Simulator(cy_err, cz_err)
contact_actual = sim_err.contact_pts

# ── 加载参考 ──
ref = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_trajectory.npz'))
ref_ball = ref['ball_actual']

print(f'seed={seed}  {len(log_Fn)} steps, Fn={log_Fn[50:].mean():.2f}±{log_Fn[50:].std():.2f}  limit>1mm:{n1} >2mm:{n2}')

# ── 画图: 2×4 ──
fig = plt.figure(figsize=(20, 10))

# === 上排: 3D | YZ | XZ | XY ===
ax = fig.add_subplot(2, 4, 1, projection='3d')
ax.plot(sim.contact_pts[:,0], sim.contact_pts[:,1], sim.contact_pts[:,2],
        'gray', ls='--', lw=0.8, label='contact std')
ax.plot(contact_actual[:,0], contact_actual[:,1], contact_actual[:,2],
        'green', lw=1.2, alpha=0.5, label='contact actual')
ax.plot(ref_ball[:,0], ref_ball[:,1], ref_ball[:,2],
        'steelblue', ls=':', lw=1, label='ball ref')
ax.plot(log_pos[:,0], log_pos[:,1], log_pos[:,2],
        'coral', lw=1.2, label='ball actual')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f'3D (seed={seed})')
ax.legend(fontsize=5)
center = np.mean(sim.contact_pts, axis=0)
ax.set_xlim(center[0]-15, center[0]+15)
ax.set_ylim(center[1]-15, center[1]+15)
ax.set_zlim(center[2]-15, center[2]+15)
ax.set_box_aspect([1,1,1])

for col, (xs, ys, xl, yl, title) in enumerate([
    (1, 2, 'Y', 'Z', 'YZ'), (0, 2, 'X', 'Z', 'XZ'), (0, 1, 'X', 'Y', 'XY')
]):
    ax = fig.add_subplot(2, 4, 2+col)
    ax.plot(sim.contact_pts[:,xs], sim.contact_pts[:,ys], 'gray', ls='--', lw=0.8, label='std')
    ax.plot(contact_actual[:,xs], contact_actual[:,ys], 'green', lw=1.2, alpha=0.5, label='actual')
    ax.plot(ref_ball[:,xs], ref_ball[:,ys], 'steelblue', ls=':', lw=1, label='ref')
    ax.plot(log_pos[:,xs], log_pos[:,ys], 'coral', lw=1.2, label='ball')
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(fontsize=5); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# === 下排: Fn | Fo | dn_true | db_true ===
for col, (data, ylim, ylabel, title, color) in enumerate([
    (log_Fn, (-20, 0), 'Fn (N)', f'Fn (mean={log_Fn[50:].mean():.1f})', 'steelblue'),
    (log_Fo, (-5, 5), 'Fo (N)', f'Fo (std={log_Fo[50:].std():.2f})', 'coral'),
    (log_dn_true, (-3, 3), 'dn (mm)', f'True dn (mean={log_dn_true.mean():.2f})', 'darkgreen'),
    (log_db_true, (-3, 3), 'db (mm)', f'True db (mean={log_db_true.mean():.2f})', 'darkorange'),
]):
    ax = fig.add_subplot(2, 4, 5+col)
    ax.plot(data, color=color, lw=0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('step'); ax.set_ylabel(ylabel); ax.set_ylim(ylim)
    ax.set_title(title); ax.grid(True, alpha=0.3)

fig.suptitle(f'Error simulation (seed={seed})', fontsize=14)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', f'sim_seed{seed}.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
