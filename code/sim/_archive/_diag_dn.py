"""诊断: dn_actual 分布"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_control_sim_v5 import ForceController, load_standard_cylinders
from sphere_contact import sphere_contact_force

DT = 0.005
cy0, cz0 = load_standard_cylinders()
ctrl = ForceController(cy0, cz0)
pos = ctrl._ball_ref(0)

dn_list = []
for step in range(3000):
    s_cur = step / 2999
    F_vec, _ = sphere_contact_force(pos, cz0, cy0)
    
    P_ct = ctrl._nearest_contact(pos)
    from force_mechanics_v2 import compute_point_basis_ortho
    basis = compute_point_basis_ortho(P_ct, ctrl.contact_geom)
    n = basis.normal
    P_ref = ctrl._ball_ref(s_cur)
    dn_list.append(np.dot(pos - P_ref, n))
    
    v_3d = ctrl.step(F_vec, s_cur, pos, 3000, DT)
    pos = pos + v_3d * DT

dn_list = np.array(dn_list)
print(f"dn_actual: min={dn_list.min():.3f} max={dn_list.max():.3f} mean={dn_list.mean():.3f} std={dn_list.std():.3f}")
print(f">1mm: {(abs(dn_list)>1).sum()}/{len(dn_list)}")
print(f">2mm: {(abs(dn_list)>2).sum()}/{len(dn_list)}")
