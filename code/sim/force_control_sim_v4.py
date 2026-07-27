"""
force_control_sim_v4.py — 力控仿真

控制器 ForceController:
  step(F_vec, s_sim) → v_3d（世界坐标速度，mm/s）
  主循环: pos += v_3d * dt
"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from cylinder_def import CylinderDef
from cylinder_geometry_v2 import sample_intersection
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from force_field_fixed import inverse, predict

F_TARGET = -8.0
DT = 0.005


class Admittance1D:
    """返回速度 (mm/s)"""
    def __init__(self, M=0.1, D=2.0, dt=DT):
        self.M, self.D, self.dt = M, D, dt; self.vel = 0.0
    def compute(self, err):
        a = (err - self.D * self.vel) / self.M
        self.vel += a * self.dt
        if abs(self.vel) > 160.0: self.vel = np.sign(self.vel) * 160.0
        return self.vel
    def reset(self): self.vel = 0.0


class LowPass:
    def __init__(self, a=0.15): self.a, self.v, self.ok = a, 0.0, False
    def update(self, x):
        if not self.ok: self.v, self.ok = x, True
        else: self.v = self.a * x + (1 - self.a) * self.v
        return self.v


class ForceController:
    def __init__(self, cy_std, cz_std):
        self.cy, self.cz = cy_std, cz_std
        self.contact_geom = sample_intersection(cy_std, cz_std, n_samples=2000)
        self.contact_pts = self.contact_geom.sample_pts
        self.adm_n = Admittance1D()
        self.adm_b = Admittance1D()
        self.filt_fn = LowPass()
        self.filt_fo = LowPass()

    def contact_at(self, s):
        n = len(self.contact_pts)
        i = int((s % 1.0) * (n - 1))
        return self.contact_pts[max(0, min(i, n - 1))]

    def step(self, F_vec, s_sim, L, total_steps, dt=DT):
        """返回世界坐标速度 v_3d (mm/s)"""
        P_ct = self.contact_at(s_sim)
        basis = compute_point_basis_ortho(P_ct, self.contact_geom)
        n = basis.normal
        t = basis.tangent
        o = basis.ortho

        Fn = np.dot(F_vec, n)
        Fo = np.dot(F_vec, o)

        Fn_f = self.filt_fn.update(Fn)
        Fo_f = self.filt_fo.update(Fo)

        dn_eq, db_eq = inverse(Fn_f, Fo_f)
        vn = self.adm_n.compute(-dn_eq)
        vb = 0
        v_fwd = L / (total_steps * dt)

        return vn * n + vb * o + v_fwd * t


def run_sim(cy_std, cz_std, cy_err, cz_err, label="", n_steps=3000):
    ctrl = ForceController(cy_std, cz_std)

    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    ball_ref = d['ball_center_500']
    diffs = np.diff(ball_ref, axis=0)
    L = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

    pos = ball_ref[0].copy()

    traj, flog, fnlog = [], [], []
    t0 = time.perf_counter()

    for step in range(n_steps):
        s_cur = step / (n_steps - 1)
        F_vec, _ = sphere_contact_force(pos, cz_err, cy_err)
        v_3d = ctrl.step(F_vec, s_cur, L, n_steps)
        pos = pos + v_3d * DT
        traj.append(pos.copy())
        flog.append(np.linalg.norm(F_vec))

    elapsed = time.perf_counter() - t0
    flog = np.array(flog)
    print(f"  [{label}] |F|={np.mean(flog[-500:]):.2f}+/-{np.std(flog[-500:]):.2f}N  ({elapsed:.1f}s)")
    return np.array(traj), flog


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
