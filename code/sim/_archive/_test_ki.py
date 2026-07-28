"""Ki对比: 原Ki=0.3 vs 新Ki=0.5"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders, PID1D
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from cylinder_geometry_v2 import sample_intersection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT=0.005; MU=0.2; SIGMA=0.5; N=3000
cy0, cz0 = load_standard_cylinders()
rng = np.random.RandomState(42)

geom0 = sample_intersection(cy0, cz0, n_samples=500)
pts0 = geom0.sample_pts
with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    ball_ref = pickle.load(f)['ball_center_500']


def run(ki):
    ctrl = ForceController(cy0, cz0)
    ctrl.pid_n.Ki = ki
    pos = ctrl.ball_ref[0]
    flog = []
    v_prev = np.zeros(3)
    for step in range(N):
        sc = step/(N-1)
        F_raw,_ = sphere_contact_force(pos, cz0, cy0)
        P_ct = ctrl._nearest_contact(pos)
        b = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
        Fn = np.dot(F_raw, b.normal)
        vn = np.linalg.norm(v_prev)
        F_fric = MU*abs(Fn)*(-v_prev/vn) if vn>1e-6 else np.zeros(3)
        F_meas = F_raw + F_fric + rng.randn(3)*SIGMA
        v_3d = ctrl.step(F_meas, sc, pos, N, DT)
        pos+=v_3d*DT; v_prev=v_3d.copy()
        flog.append(np.linalg.norm(F_meas))
    flog=np.array(flog)
    return flog


print(f"Ki 对比 (μ={MU}, σ={SIGMA})")
print("="*40)
rng=np.random.RandomState(42); f03=run(0.3)
rng=np.random.RandomState(42); f05=run(0.5)
print(f"Ki=0.3: |F|={np.mean(f03[-500:]):.2f}+/-{np.std(f03[-500:]):.2f}N")
print(f"Ki=0.5: |F|={np.mean(f05[-500:]):.2f}+/-{np.std(f05[-500:]):.2f}N")

fig, axes = plt.subplots(2,2,figsize=(14,10))
ax_f=axes[0][0]; ax_conv=axes[0][1]; ax_last03=axes[1][0]; ax_last05=axes[1][1]

ax_f.plot(f03,'blue',lw=0.5,alpha=0.7,label=f'Ki=0.3')
ax_f.plot(f05,'red',lw=0.8,label=f'Ki=0.5')
ax_f.axhline(8,color='gray',ls='--'); ax_f.legend(fontsize=8); ax_f.grid(alpha=0.3)
ax_f.set_title(f'力全程 Ki对比'); ax_f.set_ylabel('|F| (N)')

ax_conv.plot(f03[:200],'blue',lw=0.8,alpha=0.7,label=f'Ki=0.3')
ax_conv.plot(f05[:200],'red',lw=0.8,label=f'Ki=0.5')
ax_conv.axhline(8,color='gray',ls='--'); ax_conv.legend(fontsize=8); ax_conv.grid(alpha=0.3)
ax_conv.set_title('收敛: 前200步')

for ax, flog, ki in [(ax_last03,f03,0.3),(ax_last05,f05,0.5)]:
    ax.plot(range(len(flog)-200,len(flog)),flog[-200:],lw=0.8)
    ax.axhline(8,color='gray',ls='--')
    ax.set_title(f'Ki={ki} 稳态最后200步 ({np.mean(flog[-500:]):.2f}±{np.std(flog[-500:]):.2f})')
    ax.grid(alpha=0.3)

fig.suptitle(f'V5 Ki对比 (μ={MU}, σ={SIGMA})',fontsize=14)
fig.tight_layout()
out=os.path.join(_sdir,'output','v5_ki_compare.png')
fig.savefig(out,dpi=150)
print(f'已保存 {out}')
plt.close(fig)
