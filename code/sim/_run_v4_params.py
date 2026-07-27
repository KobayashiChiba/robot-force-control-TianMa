"""V4 X+1.5 多组参数3D对比"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v4 import ForceController
from sphere_contact import sphere_contact_force
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
diffs = np.diff(ball, axis=0)
L = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
DT = 0.005
N = 3000

params = [(0.1, 2.0), (0.05, 1.0), (0.02, 0.5), (0.01, 0.2)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(18, 16), subplot_kw={'projection': '3d'})

for idx, (M, D) in enumerate(params):
    ax = axes[idx // 2][idx % 2]
    ctrl = ForceController(cy0, cz0)
    ctrl.adm_n.M = M; ctrl.adm_n.D = D; ctrl.adm_n.vel = 0; ctrl.adm_n.dt = DT
    pos = ball[0].copy()
    traj = [pos.copy()]

    for step in range(N):
        s = step / (N - 1)
        F_vec, _ = sphere_contact_force(pos, cz_err, cy0)
        v_3d = ctrl.step(F_vec, s, L, N, DT)
        pos = pos + v_3d * DT
        traj.append(pos.copy())

    traj = np.array(traj)
    flog = np.array([np.linalg.norm(sphere_contact_force(p, cz_err, cy0)[0]) for p in traj[-500:]])

    ax.plot(curve_std[:,0], curve_std[:,1], curve_std[:,2], 'blue', lw=1, alpha=0.5, label='标准接触曲线')
    ax.plot(curve_err[:,0], curve_err[:,1], curve_err[:,2], 'red', ls='--', lw=1, alpha=0.5, label='误差接触曲线')
    ax.plot(ball[:,0], ball[:,1], ball[:,2], 'green', lw=0.6, alpha=0.4, label='球刀参考')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], 'orange', lw=1, label='力控轨迹')
    ax.scatter(*traj[0], c='cyan', s=60, marker='o', zorder=5)
    ax.scatter(*traj[-1], c='magenta', s=60, marker='s', zorder=5)
    ax.set_title(f'M={M} D={D}  |F|={np.mean(flog):.1f}+/-{np.std(flog):.1f}N', fontsize=11)
    ax.legend(fontsize=6)

fig.suptitle('V4 X+1.5 参数扫描', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v4_param_scan.png')
fig.savefig(out, dpi=120)
print(f'已保存 {out}')
plt.close(fig)
