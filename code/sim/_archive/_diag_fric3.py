"""摩擦力三分量分解"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT=0.005; MU=0.2; N=3000
cy0, cz0 = load_standard_cylinders()
ctrl = ForceController(cy0, cz0)
pos = ctrl.ball_ref[0]

log_Fn, log_Fo, log_Ft = [], [], []
log_fric_n, log_fric_o, log_fric_t = [], [], []
v_prev = np.zeros(3)

for step in range(N):
    sc = step/(N-1)
    F_raw,_ = sphere_contact_force(pos, cz0, cy0)
    P_ct = ctrl._nearest_contact(pos)
    b = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
    n, o, t = b.normal, b.ortho, b.tangent
    
    Fn_raw = np.dot(F_raw, n)
    
    f_dir = F_raw / np.linalg.norm(F_raw)
    v_tang = v_prev - np.dot(v_prev, f_dir) * f_dir
    vn_t = np.linalg.norm(v_tang)
    F_fric = MU*abs(Fn_raw)*(-v_tang/vn_t) if vn_t>1e-6 else np.zeros(3)
    
    # 分解到 {n, o, t}
    log_fric_n.append(np.dot(F_fric, n))
    log_fric_o.append(np.dot(F_fric, o))
    log_fric_t.append(np.dot(F_fric, t))
    
    v_3d = ctrl.step(F_raw + F_fric, sc, pos, N, DT)
    pos+=v_3d*DT; v_prev=v_3d.copy()

# 图
s,e = 1500,2000
xs = np.arange(e-s)

fig, axes = plt.subplots(2,2,figsize=(16,10))

ax1=axes[0][0]
ax1.plot(xs, np.array(log_fric_n[s:e]), 'blue', lw=1, label='n分量')
ax1.plot(xs, np.array(log_fric_o[s:e]), 'orange', lw=1, label='o分量')
ax1.plot(xs, np.array(log_fric_t[s:e]), 'green', lw=1, label='t分量')
ax1.set_title('摩擦力三分量 (⟂F_raw方向)'); ax1.set_ylabel('N')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2=axes[0][1]
ax2.hist(np.array(log_fric_n[s:e]), bins=40, alpha=0.6, label='n')
ax2.hist(np.array(log_fric_o[s:e]), bins=40, alpha=0.6, label='o')
ax2.hist(np.array(log_fric_t[s:e]), bins=40, alpha=0.6, label='t')
ax2.set_title('分布'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3=axes[1][0]
fn = np.array(log_fric_n[s:e])
fo = np.array(log_fric_o[s:e])
ft = np.array(log_fric_t[s:e])
ax3.bar(['n','o','t'], [np.std(fn),np.std(fo),np.std(ft)])
ax3.set_title('各方向 std (N)')

ax4=axes[1][1]
ax4.plot(xs, np.array(log_fric_n[s:e]), 'blue', lw=1)
ax4.plot(xs, np.array(log_fric_t[s:e]), 'green', lw=0.8)
ax4.set_title('n vs t 分量 (放大)'); ax4.set_ylabel('N')
ax4.legend(['n','t'],fontsize=8); ax4.grid(alpha=0.3)

fig.suptitle(f'摩擦力分解 (μ={MU}, ⟂F_raw方向, 稳态1500~2000步)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_friction_components.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)

print(f"n: mean={fn.mean():.4f} std={fn.std():.4f}")
print(f"o: mean={fo.mean():.4f} std={fo.std():.4f}")
print(f"t: mean={ft.mean():.4f} std={ft.std():.4f}")
