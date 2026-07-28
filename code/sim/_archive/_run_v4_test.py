"""V4 无误差跑一次 + 3D轨迹图"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v4 import ForceController, run_sim

with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    d = pickle.load(f)
cy0, cz0, ball_ref = d['cyl_contact_y'], d['cyl_contact_z'], d['ball_center_500']

print("运行...")
traj, flog = run_sim(cy0, cz0, cy0, cz0, label="无误差", n_steps=3000)
print(f"|F|={np.mean(flog[-500:]):.2f}+/-{np.std(flog[-500:]):.2f}N")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ball_ref[:,0], ball_ref[:,1], ball_ref[:,2], 'gray', ls='--', lw=0.8, alpha=0.5, label='球心参考轨迹')
ax.plot(traj[:,0], traj[:,1], traj[:,2], 'red', lw=1.0, label='力控轨迹')
ax.scatter(*traj[0], c='blue', s=60, marker='o', zorder=5, label='起点')
ax.scatter(*traj[-1], c='green', s=60, marker='s', zorder=5, label='终点')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
ax.set_title(f'V4 无误差 |F|={np.mean(flog[-500:]):.1f}+/-{np.std(flog[-500:]):.1f}N')
ax.legend(fontsize=8)
ax.set_box_aspect([np.ptp(ball_ref[:,0]), np.ptp(ball_ref[:,1]), np.ptp(ball_ref[:,2])])

out = os.path.join(_sdir, 'output', 'v4_noerror.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
