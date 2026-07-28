"""诊断: 绕过_blend，直接PID-zero"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho

DT = 0.005
N_STEPS = 3000
cy0, cz0 = load_standard_cylinders()
ctrl = ForceController(cy0, cz0)
pos = ctrl._ball_ref(0)

traj, flog = [], []
for step in range(N_STEPS):
    s_cur = step / (N_STEPS - 1)
    F_vec, _ = sphere_contact_force(pos, cz0, cy0)
    
    P_ct = ctrl._nearest_contact(pos)
    basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
    n, o = basis.normal, basis.ortho
    
    Fn = np.dot(F_vec, n)
    Fn_f = ctrl.filt_fn.update(Fn)
    
    P_ref = ctrl._ball_ref(s_cur)
    dn_actual = np.dot(pos - P_ref, n)
    do_actual = np.dot(pos - P_ref, o)
    
    # 纯PID-zero: 追0位置偏移
    vn = ctrl.pid_n.step(-dn_actual)
    vb = ctrl.pid_o.step(-do_actual)
    
    v_fwd = ctrl.L / (N_STEPS * DT)
    v_3d = vn * n + vb * o + v_fwd * basis.tangent
    pos = pos + v_3d * DT
    traj.append(pos.copy())
    flog.append(np.linalg.norm(F_vec))

flog = np.array(flog)
last500 = flog[-500:]
gap = np.linalg.norm(np.array(traj[-1]) - np.array(traj[0]))
print(f"|F| = {np.mean(last500):.2f} +/- {np.std(last500):.2f} N")
print(f"首尾距离 = {gap:.3f} mm")
