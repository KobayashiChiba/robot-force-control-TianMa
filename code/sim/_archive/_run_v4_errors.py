"""V4 带误差测试"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v4 import ForceController, run_sim
from cylinder_def import CylinderDef

with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    d = pickle.load(f)
cy0, cz0, ball = d['cyl_contact_y'], d['cyl_contact_z'], d['ball_center_500']

def translate(cz, dx=0, dy=0, dz=0):
    t = np.array([dx, dy, dz])
    return CylinderDef(p1=cz.p1+t, p2=cz.p2+t, radius=cz.radius)

def rotate(cz, axis, deg):
    a = np.radians(deg)
    R = {'x': np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]),
         'y': np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]),
         'z': np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])}[axis]
    ctr = (cz.p1+cz.p2)/2; L = np.linalg.norm(cz.p2-cz.p1)
    d = R @ np.array([0,0,1])
    return CylinderDef(p1=ctr-L/2*d, p2=ctr+L/2*d, radius=cz.radius)

errors = [
    ("无误差",                cz0),
    ("平移 X+1.5",            translate(cz0, 1.5, 0, 0)),
    ("平移 Y+1.5",            translate(cz0, 0, 1.5, 0)),
    ("绕 X 旋转 2 deg",       rotate(cz0, 'x', 2.0)),
    ("绕 Y 旋转 2 deg",       rotate(cz0, 'y', 2.0)),
    ("X转1 deg + Y平移1",      translate(rotate(cz0, 'x', 1.0), 0, 1.0, 0)),
]

results = []
for name, cz_err in errors:
    traj, flog, fnlog = run_sim(cy0, cz0, cy0, cz_err, ball, label=name)
    results.append((name, traj, flog, fnlog))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

colors = ['gray','#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6']
fig = plt.figure(figsize=(20, 14))

ax1 = fig.add_subplot(231)
for i,(n,t,fl,_) in enumerate(results):
    ax1.plot(fl, color=colors[i], lw=0.8, alpha=0.8, label=n)
ax1.axhline(8.0, color='black', ls='--', lw=0.5)
ax1.set_ylabel('|F| (N)'); ax1.set_title('合力收敛'); ax1.grid(alpha=0.3); ax1.legend(fontsize=6)

ax2 = fig.add_subplot(232)
bp = ax2.boxplot([r[2][-500:] for r in results], tick_labels=[r[0] for r in results], patch_artist=True)
for i in range(len(results)): bp['boxes'][i].set_facecolor(colors[i])
ax2.set_ylabel('|F| (N)'); ax2.set_title('稳定段(末500步)'); ax2.tick_params(axis='x', rotation=30, labelsize=7)

ax3 = fig.add_subplot(233)
x = np.arange(len(results))
mf = [np.mean(r[2][-500:]) for r in results]
sf = [np.std(r[2][-500:]) for r in results]
ax3.bar(x, mf, color=colors)
ax3.errorbar(x, mf, yerr=sf, fmt='none', ecolor='black', capsize=4)
ax3.axhline(8.0, color='black', ls='--', lw=0.5)
ax3.set_xticks(x); ax3.set_xticklabels([r[0] for r in results], rotation=30, fontsize=7)
ax3.set_ylabel('|F|均值 (N)'); ax3.set_title('力均值+/-std')

for si, rng in [(234, range(3)), (235, range(3,6))]:
    ax = fig.add_subplot(si, projection='3d')
    ax.plot(ball[:,0], ball[:,1], ball[:,2], 'gray', ls='--', lw=0.5, alpha=0.4, label='球心参考')
    for i in rng:
        ax.plot(results[i][1][:,0], results[i][1][:,1], results[i][1][:,2],
                color=colors[i], lw=0.8, alpha=0.6, label=results[i][0])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title('3D轨迹'); ax.legend(fontsize=6)

ax6 = fig.add_subplot(236); ax6.axis('off')
td = [['测试组', '|F|均值', '|F| std', 'Fn均值']]
for n,t,fl,fnl in results:
    td.append([n, f'{np.mean(fl[-500:]):.1f}', f'{np.std(fl[-500:]):.1f}', f'{np.mean(fnl[-500:]):.1f}'])
tbl = ax6.table(cellText=td, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.5)
ax6.set_title('结果汇总', y=0.75)

fig.suptitle('V4 力控仿真 (D=20 目标偏移=0)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v4_errors.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
