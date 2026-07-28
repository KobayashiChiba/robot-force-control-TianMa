"""诊断: 排查PID-zero为什么崩了"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho

DT = 0.005
cy0, cz0 = load_standard_cylinders()
ctrl = ForceController(cy0, cz0)

# 检查关键参数
print(f"arc length L = {ctrl.L:.2f} mm")
print(f"ball_ref shape = {ctrl.ball_ref.shape}")
print(f"ball_ref[0] = {ctrl.ball_ref[0]}")
print(f"ball_ref[-1] = {ctrl.ball_ref[-1]}")
print(f"contact_pts shape = {ctrl.contact_pts.shape}")
print(f"pid_n: Kp={ctrl.pid_n.Kp}, Ki={ctrl.pid_n.Ki}")
print(f"pid_o: Kp={ctrl.pid_o.Kp}, Ki={ctrl.pid_o.Ki}")

# 初始位置处的力
pos0 = ctrl._ball_ref(0)
F0, _ = sphere_contact_force(pos0, cz0, cy0)
print(f"\n初始位置: {pos0}")
print(f"初始力: |F|={np.linalg.norm(F0):.2f}N, F_vec={F0}")

# 球刀沿参考轨迹走，看力变化
print("\n沿参考轨迹力采样:")
for s in [0.0, 0.2, 0.4, 0.6, 0.8]:
    pos = ctrl._ball_ref(s)
    F, _ = sphere_contact_force(pos, cz0, cy0)
    print(f"  s={s:.1f}: |F|={np.linalg.norm(F):.2f}N")

# PID-zero 单步诊断
ctrl2 = ForceController(cy0, cz0)
pos = ctrl2._ball_ref(0)
print("\nPID-zero 前10步:")
for step in range(10):
    s_cur = step / 2999
    F_vec, _ = sphere_contact_force(pos, cz0, cy0)
    P_ct = ctrl2._nearest_contact(pos)
    basis = compute_point_basis_ortho(P_ct, ctrl2.contact_geom)
    n, o = basis.normal, basis.ortho
    P_ref = ctrl2._ball_ref(s_cur)
    dn = np.dot(pos - P_ref, n)
    do = np.dot(pos - P_ref, o)
    vn = ctrl2.pid_n.step(-dn)
    vb = ctrl2.pid_o.step(-do)
    v_fwd = ctrl2.L / (3000 * DT)
    pos = pos + (vn*n + vb*o + v_fwd*basis.tangent) * DT
    print(f"  step{step}: dn={dn:+.4f} vn={vn:+.2f} pos={pos}")
