"""诊断: 摩擦扰动的周期性"""
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

DT=0.005; MU=0.2; SIGMA=0.0; N=3000
cy0, cz0 = load_standard_cylinders()
rng = np.random.RandomState(42)
ctrl = ForceController(cy0, cz0)
pos = ctrl.ball_ref[0]

log_Fn, log_Ffric_n, log_vn, log_vt, log_dn = [], [], [], [], []
v_prev = np.zeros(3)

for step in range(N):
    sc = step/(N-1)
    F_raw,_ = sphere_contact_force(pos, cz0, cy0)
    P_ct = ctrl._nearest_contact(pos)
    b = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
    n, t = b.normal, b.tangent
    
    Fn_raw = np.dot(F_raw, n)
    v_norm = np.linalg.norm(v_prev)
    F_fric = MU*abs(Fn_raw)*(-v_prev/v_norm) if v_norm>1e-6 else np.zeros(3)
    Fn_fric = np.dot(F_fric, n)
    
    # 速度分量
    v_3d = ctrl.step(F_raw + F_fric, sc, pos, N, DT)
    vn = np.dot(v_3d, n)
    vt = np.dot(v_3d, t)
    
    P_ref = ctrl._nearest_ball_ref(pos)
    dn = np.dot(pos - P_ref, n)
    
    pos += v_3d*DT; v_prev=v_3d.copy()
    log_Fn.append(Fn_raw)
    log_Ffric_n.append(Fn_fric)
    log_vn.append(vn)
    log_vt.append(vt)
    log_dn.append(dn)

# 只看稳态段 (1500~2000步)
s,e = 1500,2000
xs = np.arange(e-s)

fig, axes = plt.subplots(2,2,figsize=(16,10))

ax1=axes[0][0]
ax1.plot(xs, np.array(log_Fn[s:e]), 'b-', lw=1, label='Fn_raw')
ax1.plot(xs, np.array(log_Ffric_n[s:e]), 'orange', lw=1, label='Fn_fric (n分量)')
ax1.axhline(-8, color='gray', ls='--')
ax1.set_ylabel('N'); ax1.set_title('Fn 分解: 接触力 vs 摩擦力n分量')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2=axes[0][1]
ax2.plot(xs, np.array(log_vn[s:e]), 'red', lw=1, label='vn (法向)')
ax2.plot(xs, np.array(log_vt[s:e]), 'green', lw=1, label='vt (切向)')
ax2.set_ylabel('mm/s'); ax2.set_title('速度分量')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3=axes[1][0]
ax3.plot(xs, np.array(log_Fn[s:e]), 'b-', lw=1, label='Fn')
ax3.plot(xs, np.array(log_vn[s:e]), 'red', lw=0.8, label='vn')
ax3.set_ylabel('Fn(N) / vn(mm/s)'); ax3.set_title('Fn vs vn 相位关系')
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

ax4=axes[1][1]
ax4.plot(xs, np.array(log_dn[s:e]), 'purple', lw=1)
ax4.set_ylabel('mm'); ax4.set_title('dn_actual'); ax4.grid(alpha=0.3)

# 自相关找周期
from scipy import signal
if False:
    fn_seg = np.array(log_Fn[s:e])
    fn_seg -= fn_seg.mean()
    corr = np.correlate(fn_seg, fn_seg, mode='full')
    corr = corr[len(corr)//2:]
    peaks = signal.find_peaks(corr[:200])[0]
    if len(peaks)>1:
        print(f"Fn自相关峰值间隔: {peaks[1]-peaks[0]} steps")

fig.suptitle('摩擦扰动诊断 (μ=0.2, 无噪声, 稳态1500~2000步)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_friction_diag.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)

# 统计
fn_seg = np.array(log_Fn[s:e])
print(f"Fn: mean={fn_seg.mean():.2f} std={fn_seg.std():.2f}")
fric_seg = np.array(log_Ffric_n[s:e])
print(f"Fn_fric: mean={fric_seg.mean():.3f} std={fric_seg.std():.3f}")
vn_seg = np.array(log_vn[s:e])
print(f"vn: mean={vn_seg.mean():.2f} std={vn_seg.std():.2f}")
