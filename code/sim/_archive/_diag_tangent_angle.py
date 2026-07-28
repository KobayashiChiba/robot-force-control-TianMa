"""量化: 接触曲线切线 vs 球刀参考切线 的夹角"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from force_mechanics_v2 import compute_point_basis_ortho

cy0, cz0 = load_standard_cylinders()
ctrl = ForceController(cy0, cz0)

angles = []
for i in range(0, 3000, 30):
    s = i / 2999.0
    pos = ctrl._ball_ref(s)
    t_ref = ctrl._ref_tangent(s)
    
    P_ct = ctrl._nearest_contact(pos)
    basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
    t_contact = basis.tangent
    
    # 夹角 (度)
    dot = np.clip(np.dot(t_ref, t_contact), -1.0, 1.0)
    angle = np.arccos(dot) * 180 / np.pi
    angles.append(angle)

angles = np.array(angles)
print(f"切线夹角: min={angles.min():.2f}°  max={angles.max():.2f}°  mean={angles.mean():.2f}°  std={angles.std():.2f}°")

# 转化为法向分量
v_fwd = ctrl.L / (3000 * 0.005)
drift_n = v_fwd * np.sin(np.radians(angles))
print(f"切向速度 {v_fwd:.1f} mm/s → 法向漂移: {drift_n.min():.3f} ~ {drift_n.max():.3f} mm/s")
print(f"PID vn 典型值: ~{ctrl.pid_n.Kp * 0.01:.2f} mm/s (dn=0.01mm)")
