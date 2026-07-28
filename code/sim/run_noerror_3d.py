"""
run_noerror_3d.py — 无误差 V5 力控仿真 + 3D 图 (lib_v3)
"""
import sys, os, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
sys.path.insert(0, os.path.join(_sdir, '..'))
from lib_v3 import Simulator, ForceController, load_cylinders, load_ball_ref
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT = 0.005; N_STEPS = 3000; MU = 0.2; SIGMA = 0.5

cy, cz = load_cylinders(os.path.join(_sdir, '..', 'data'))
ball_ref, L = load_ball_ref(os.path.join(_sdir, '..', 'data'))
sim = Simulator(cy, cz, mu=MU, sigma=SIGMA, seed=42)
ctrl = ForceController(ball_ref, L, sim.contact_geom)

# 参考曲线
geom0 = sample_intersection(cy, cz, n_samples=500)
pts0 = geom0.sample_pts

pos = ball_ref[0]
traj, flog = [], []
v_prev = np.zeros(3)
t0 = time.perf_counter()

for step in range(N_STEPS):
    F_meas, F_raw, _, _, _ = sim.step(pos, v_prev)
    v_3d = ctrl.step(F_meas, pos, N_STEPS, DT)
    pos += v_3d * DT
    v_prev = v_3d.copy()
    traj.append(pos.copy())
    flog.append(np.linalg.norm(F_meas))

traj = np.array(traj); flog = np.array(flog)
last500 = flog[-500:]; elapsed = time.perf_counter() - t0
gap = np.linalg.norm(traj[-1] - traj[0])
print(f"|F| = {np.mean(last500):.2f} +/- {np.std(last500):.2f} N  ({elapsed:.1f}s)")
print(f"首尾距离 = {gap:.3f} mm  (弧长 {L:.1f} mm)")

# === 图 ===
fig = plt.figure(figsize=(16, 12))
ax3d = fig.add_subplot(221, projection='3d')
ax_f, ax_start, ax_last = fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)

ax3d.plot(pts0[:,0], pts0[:,1], pts0[:,2], 'gray', ls='--', lw=1, alpha=0.5, label='接触曲线')
ax3d.plot(ball_ref[:,0], ball_ref[:,1], ball_ref[:,2], 'green', lw=0.6, alpha=0.4, label='球刀参考')
ax3d.plot(traj[:,0], traj[:,1], traj[:,2], 'blue', lw=1.2, label='力控轨迹')
ax3d.scatter(*traj[0], c='cyan', s=60, zorder=5, label='起点')
ax3d.scatter(*traj[-1], c='red', s=60, zorder=5, label='终点')
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
ax3d.set_title(f'V5 无误差 (lib_v3)\n|F|={np.mean(last500):.2f}±{np.std(last500):.2f}N  首尾距={gap:.2f}mm')
ax3d.legend(fontsize=8)

for ax, seg, title in [(ax_f, flog, '力全程'), (ax_start, flog[:100], '前100步'), (ax_last, flog[-200:], '稳态200步')]:
    xs = range(len(seg)-len(seg), len(seg)) if seg is flog[-200:] else range(len(seg))
    ax.plot(xs, seg, 'b-', lw=0.5)
    ax.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax.set_title(title); ax.grid(alpha=0.3)

fig.suptitle('V5 力控仿真 — 无误差 (lib_v3)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_noerror_3d.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
