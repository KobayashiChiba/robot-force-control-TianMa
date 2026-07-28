"""V5 随机误差 3D图 (含摩擦+噪声)"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders, translate_cz
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
contact_std = sample_intersection(cy0, cz0, n_samples=500).sample_pts
with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    ball_ref = pickle.load(f)['ball_center_500']

def run(cy_err, cz_err, rng):
    ctrl = ForceController(cy0, cz0)
    pos = ctrl.ball_ref[0]
    traj, flog = [], []
    v_prev = np.zeros(3)
    for step in range(N):
        sc = step/(N-1)
        F_raw,_ = sphere_contact_force(pos, cz_err, cy_err)
        P_ct = ctrl._nearest_contact(pos)
        b = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
        Fn = np.dot(F_raw, b.normal)
        fn = np.linalg.norm(F_raw)
        if fn<1e-6: F_fric=np.zeros(3)
        else:
            fd=F_raw/fn; vt=v_prev-np.dot(v_prev,fd)*fd; vn_t=np.linalg.norm(vt)
            F_fric=MU*abs(Fn)*(-vt/vn_t) if vn_t>1e-6 else np.zeros(3)
        F_meas = F_raw+F_fric+rng.randn(3)*SIGMA
        v_3d = ctrl.step(F_meas, sc, pos, N, DT)
        pos+=v_3d*DT; v_prev=v_3d.copy()
        traj.append(pos.copy()); flog.append(np.linalg.norm(F_meas))
    return np.array(traj), np.array(flog)

out_dir = os.path.join(_sdir, 'output')
for seed in range(10):
    rng = np.random.RandomState(seed*100+42)
    np.random.seed(seed)
    dx,dy,dz = np.random.uniform(-0.5,0.5,3)
    czr = translate_cz(cz0, dx=dx, dy=dy, dz=dz)
    traj, flog = run(cy0, czr, rng)
    fm, fs = np.mean(flog[-500:]), np.std(flog[-500:])
    
    # 误差接触曲线
    cg = sample_intersection(cy0, czr, n_samples=500)
    contact_err = cg.sample_pts
    ball_ref_err = np.zeros_like(contact_err)
    for i in range(len(contact_err)):
        b2 = compute_point_basis_ortho(contact_err[i], cg)
        ball_ref_err[i] = contact_err[i] - 4.0*b2.normal
    
    fig = plt.figure(figsize=(16,10))
    ax3d = fig.add_subplot(221, projection='3d')
    ax_f = fig.add_subplot(222)
    ax_xy = fig.add_subplot(223)
    ax_xz = fig.add_subplot(224)
    
    ax3d.plot(contact_std[:,0],contact_std[:,1],contact_std[:,2],'gray',ls='--',lw=1,alpha=0.5,label='标准接触')
    ax3d.plot(contact_err[:,0],contact_err[:,1],contact_err[:,2],'red',ls='--',lw=0.8,alpha=0.5,label='误差接触')
    ax3d.plot(ball_ref[:,0],ball_ref[:,1],ball_ref[:,2],'green',lw=0.5,alpha=0.3,label='标准球刀参考')
    ax3d.plot(traj[:,0],traj[:,1],traj[:,2],'blue',lw=1.2,label='力控轨迹')
    ax3d.scatter(*traj[0],c='cyan',s=40,zorder=5); ax3d.scatter(*traj[-1],c='red',s=40,zorder=5)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.set_title(f'#{seed} ({dx:+.2f},{dy:+.2f},{dz:+.2f})'); ax3d.legend(fontsize=7)
    
    ax_f.plot(flog,'b-',lw=0.5); ax_f.axhline(8,color='gray',ls='--')
    ax_f.set_title(f'|F|={fm:.2f}±{fs:.2f}N'); ax_f.grid(alpha=0.3)
    
    ax_xy.plot(traj[:,0],traj[:,1],'blue',lw=0.8)
    ax_xy.plot(contact_std[:,0],contact_std[:,1],'gray',lw=0.5,alpha=0.3)
    ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y'); ax_xy.set_title('XY投影')
    ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3)
    
    ax_xz.plot(traj[:,0],traj[:,2],'blue',lw=0.8)
    ax_xz.plot(contact_std[:,0],contact_std[:,2],'gray',lw=0.5,alpha=0.3)
    ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z'); ax_xz.set_title('XZ投影')
    ax_xz.set_aspect('equal'); ax_xz.grid(alpha=0.3)
    
    fig.suptitle(f'V5 随机误差 #{seed} (μ={MU},σ={SIGMA})',fontsize=14)
    fig.tight_layout()
    fname = os.path.join(out_dir, f'v5_rand3d_{seed:02d}.png')
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f'#{seed} ({dx:+.2f},{dy:+.2f},{dz:+.2f}): |F|={fm:.2f}±{fs:.2f}N  → {fname}')
