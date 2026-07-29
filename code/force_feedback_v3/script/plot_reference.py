"""
plot_reference.py — 参考轨迹出图

加载 reference_trajectory.npz，画 1×4 布局：
  1. 3D  — contact std(灰虚线) + ball std(蓝点线) + ball ref(红实线)
  2. YZ 投影
  3. Fn — ylim [-20,0]
  4. Fo — ylim [-5,5]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)

data = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'reference_trajectory.npz'))
log_pos = data['ball_actual']
log_Fn  = data['Fn_ref']
log_Fo  = data['Fo_ref']
steps   = np.arange(len(log_Fn))

fig = plt.figure(figsize=(24, 6))

# 1: 3D
ax = fig.add_subplot(1, 4, 1, projection='3d')
ax.plot(sim.contact_pts[:,0], sim.contact_pts[:,1], sim.contact_pts[:,2],
        'gray', ls='--', lw=0.8, label='contact std')
ax.plot(ball_ref[:,0], ball_ref[:,1], ball_ref[:,2],
        'steelblue', ls=':', lw=1, label='ball std')
ax.plot(log_pos[:,0], log_pos[:,1], log_pos[:,2],
        'coral', lw=1.2, label='ball ref')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Reference 3D')
ax.legend(fontsize=7)
center = np.mean(sim.contact_pts, axis=0)
for a, arr in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
    a(center[arr] - 15, center[arr] + 15)
ax.set_box_aspect([1,1,1])

# 2: YZ 投影
ax = fig.add_subplot(1, 4, 2)
ax.plot(sim.contact_pts[:,1], sim.contact_pts[:,2],
        'gray', ls='--', lw=0.8, label='contact std')
ax.plot(ball_ref[:,1], ball_ref[:,2], 'steelblue', ls=':', lw=1, label='ball std')
ax.plot(log_pos[:,1], log_pos[:,2], 'coral', lw=1.2, label='ball ref')
ax.set_xlabel('Y'); ax.set_ylabel('Z')
ax.set_title('YZ projection')
ax.legend(fontsize=7)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 3: Fn
ax = fig.add_subplot(1, 4, 3)
ax.plot(steps, log_Fn, color='steelblue', lw=1)
ax.axhline(-8, color='gray', ls='--', lw=1)
ax.set_xlabel('step'); ax.set_ylabel('Fn (N)')
ax.set_ylim(-20, 0)
ax.set_title(f'Fn ref (mean={log_Fn[50:].mean():.2f})')
ax.grid(True, alpha=0.3)

# 4: Fo
ax = fig.add_subplot(1, 4, 4)
ax.plot(steps, log_Fo, color='coral', lw=1)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel('step'); ax.set_ylabel('Fo (N)')
ax.set_ylim(-5, 5)
ax.set_title(f'Fo ref (std={log_Fo[50:].std():.3f})')
ax.grid(True, alpha=0.3)

fig.suptitle('Reference trajectory (mu=0, sigma=0, no error)', fontsize=14)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'output', 'reference.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
