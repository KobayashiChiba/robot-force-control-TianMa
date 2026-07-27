"""V4 平移 X+1.5 单组 + 四曲线3D对比"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v4 import run_sim
from cylinder_def import CylinderDef
from cylinder_geometry_v2 import sample_intersection

with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    d = pickle.load(f)
cy0, cz0, ball = d['cyl_contact_y'], d['cyl_contact_z'], d['ball_center_500']

cz_err = CylinderDef(p1=cz0.p1 + np.array([1.5, 0, 0]),
                     p2=cz0.p2 + np.array([1.5, 0, 0]),
                     radius=cz0.radius)

geom_std = sample_intersection(cy0, cz0, n_samples=500)
curve_std = geom_std.sample_pts
geom_err = sample_intersection(cy0, cz_err, n_samples=500)
curve_err = geom_err.sample_pts

traj, flog = run_sim(cy0, cz0, cy0, cz_err, label="X+1.5")
print(f"|F|={np.mean(flog[-500:]):.2f}+/-{np.std(flog[-500:]):.2f}N")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(curve_std[:,0], curve_std[:,1], curve_std[:,2], 'blue', lw=1.5, label='标准接触曲线')
ax.plot(curve_err[:,0], curve_err[:,1], curve_err[:,2], 'red', ls='--', lw=1.5, label='误差接触曲线 (X+1.5)')
ax.plot(ball[:,0], ball[:,1], ball[:,2], 'green', lw=0.8, alpha=0.6, label='标准球刀参考曲线')
ax.plot(traj[:,0], traj[:,1], traj[:,2], 'orange', lw=1.2, label='力控球刀轨迹')
ax.scatter(*traj[0], c='cyan', s=100, marker='o', zorder=5, label='起点')
ax.scatter(*traj[-1], c='magenta', s=100, marker='s', zorder=5, label='终点')

ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
ax.set_title(f'V4 平移 X+1.5  |F|={np.mean(flog[-500:]):.1f}N  M=0.1 D=2')
ax.legend(fontsize=8, loc='upper left')

all_pts = np.vstack([curve_std, curve_err, ball, traj])
ax.set_box_aspect([np.ptp(all_pts[:,0]), np.ptp(all_pts[:,1]), np.ptp(all_pts[:,2])])

out = os.path.join(_sdir, 'output', 'v4_x15_detail.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
