"""
gen_reference.py — 生成参考轨迹（无误差无摩擦无噪声）

条件: mu=0, sigma=0, cy_err=None, cz_err=None
停圈: 角度法（绕X轴，YZ均值中心，abs(累计Δθ)≥2π）
输出: data/reference_trajectory.npz (ball_actual, Fn_ref, Fo_ref, N_STEPS, DT)
用途: 作为后续误差仿真的基准线（ball ref）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.simulator import Simulator
from force_feedback_v3.lib.controller import ForceController

cy, cz = load_cylinders()
ball_ref, L = load_ball_ref()
sim = Simulator(cy, cz, mu=0.0, sigma=0.0)  # 无摩擦无噪声
ctrl = ForceController(ball_ref, L, sim.contact_geom)

DT = 0.005
N_STEPS = 2000
MAX_STEPS = 2 * N_STEPS

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
    if dtheta > np.pi: dtheta -= 2*np.pi
    elif dtheta < -np.pi: dtheta += 2*np.pi
    accum += dtheta
    theta_prev = theta

    if abs(accum) >= 2*np.pi:
        break

log_pos = np.array(log_pos)
log_Fn  = np.array(log_Fn)
log_Fo  = np.array(log_Fo)

# ── 保存 ──
out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_trajectory.npz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez(out_path, ball_actual=log_pos, Fn_ref=log_Fn, Fo_ref=log_Fo, N_STEPS=N_STEPS, DT=DT)
print(f'✓ Saved: {out_path}')
print(f'  shape: {log_pos.shape}, Fn mean={log_Fn[50:].mean():.2f}±{log_Fn[50:].std():.2f}, steps={len(log_Fn)}')
