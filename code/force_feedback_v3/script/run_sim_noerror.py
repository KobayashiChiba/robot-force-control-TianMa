"""
run_sim_noerror.py — 无误差闭环仿真 + 出图

几何误差: 无（cy_err=None, cz_err=None）
摩擦噪声: mu=0.2, sigma=0.5N
停圈: 角度法（绕X轴，YZ均值中心，abs(累计Δθ)≥2π）
出图: 1×4 (3D | YZ | Fn[-20,0] | Fo[-5,5])
图例: contact std(灰虚线) / contact actual(绿实线，重合) / ball ref(蓝点线=参考轨迹) / ball actual(红实线=仿真)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.simulator import Simulator
from force_feedback_v3.lib.controller import ForceController

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, L = load_ball_ref()
sim = Simulator(cy, cz, mu=0.2, sigma=0.5)
ctrl = ForceController(ball_ref, L, sim.contact_geom)

DT = 0.005
N_STEPS = 2000
MAX_STEPS = 2 * N_STEPS

# 角度法：绕X轴，YZ平面极角，中心=交线YZ投影均值
cy0 = np.mean(sim.contact_pts[:, 1])
cz0 = np.mean(sim.contact_pts[:, 2])

v_prev = np.zeros(3)
pos = ball_ref[0].copy()

log_pos = []; log_Fn = []; log_Fo = []
accum = 0.0
theta_prev = np.arctan2(pos[2] - cz0, pos[1] - cy0)

for k in range(MAX_STEPS):
    F_meas, _, _, _, basis = sim.step(pos, v_prev)
    v_3d = ctrl.step(F_meas, pos, N_STEPS, DT)
    pos = pos + v_3d * DT
    v_prev = v_3d

    log_pos.append(pos.copy())
    log_Fn.append(np.dot(F_meas, basis.normal))
    log_Fo.append(np.dot(F_meas, basis.ortho))

    theta = np.arctan2(pos[2] - cz0, pos[1] - cy0)
    dtheta = theta - theta_prev
    if dtheta > np.pi: dtheta -= 2 * np.pi
    elif dtheta < -np.pi: dtheta += 2 * np.pi
    accum += dtheta
    theta_prev = theta

    if abs(accum) >= 2 * np.pi:
        break

log_pos = np.array(log_pos)
log_Fn  = np.array(log_Fn)
log_Fo  = np.array(log_Fo)
steps  = np.arange(len(log_Fn))

print(f'一圈完成: {len(log_Fn)} steps / {len(log_Fn)*DT:.2f}s, 角度={np.degrees(accum):.1f}°')

# ── 加载参考轨迹 ──
ref = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_trajectory.npz'))
ref_ball = ref['ball_actual']

# ── 画图 ──
fig = plt.figure(figsize=(18, 12))

# 上排: 3D | Fn | Fo
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot(sim.contact_pts[:,0], sim.contact_pts[:,1], sim.contact_pts[:,2],
        'gray', ls='--', lw=0.8, label='contact std')
ax.plot(sim.contact_pts[:,0], sim.contact_pts[:,1], sim.contact_pts[:,2],
        'green', lw=1.2, alpha=0.5, label='contact actual')
ax.plot(ref_ball[:,0], ref_ball[:,1], ref_ball[:,2],
        'steelblue', ls=':', lw=1, label='ball ref')
ax.plot(log_pos[:,0], log_pos[:,1], log_pos[:,2],
        'coral', lw=1.2, label='ball actual')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('3D trajectory')
ax.legend(fontsize=6)
center = np.mean(sim.contact_pts, axis=0)
ax.set_xlim(center[0]-15, center[0]+15)
ax.set_ylim(center[1]-15, center[1]+15)
ax.set_zlim(center[2]-15, center[2]+15)
ax.set_box_aspect([1,1,1])

# Fn
ax = fig.add_subplot(2, 3, 2)
ax.plot(steps, log_Fn, color='steelblue', lw=1)
ax.axhline(-8, color='gray', ls='--', lw=1)
ax.set_xlabel('step'); ax.set_ylabel('Fn (N)'); ax.set_ylim(-20, 0)
ax.set_title('Fn vs step')
ax.grid(True, alpha=0.3)

# Fo
ax = fig.add_subplot(2, 3, 3)
ax.plot(steps, log_Fo, color='coral', lw=1)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('step'); ax.set_ylabel('Fo (N)'); ax.set_ylim(-5, 5)
ax.set_title('Fo vs step')
ax.grid(True, alpha=0.3)

# 下排: YZ | XZ | XY
for col, (xs, ys, xl, yl, title) in enumerate([
    (1, 2, 'Y', 'Z', 'YZ projection'),
    (0, 2, 'X', 'Z', 'XZ projection'),
    (0, 1, 'X', 'Y', 'XY projection'),
]):
    ax = fig.add_subplot(2, 3, 4+col)
    ax.plot(sim.contact_pts[:,xs], sim.contact_pts[:,ys],
            'gray', ls='--', lw=0.8, label='contact std')
    ax.plot(sim.contact_pts[:,xs], sim.contact_pts[:,ys],
            'green', lw=1.2, alpha=0.5, label='contact actual')
    ax.plot(ref_ball[:,xs], ref_ball[:,ys],
            'steelblue', ls=':', lw=1, label='ball ref')
    ax.plot(log_pos[:,xs], log_pos[:,ys],
            'coral', lw=1.2, label='ball actual')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title)
    ax.legend(fontsize=6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig.suptitle('No-error simulation', fontsize=14)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'sim_noerror.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
