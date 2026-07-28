"""
lib_v3 — V5 力控仿真库

用法:
    from lib_v3 import Simulator, ForceController, load_cylinders, perturb_cylinder

    sim = Simulator(cy, cz, mu=0.2, sigma=0.5)
    ctrl = ForceController(ball_ref, L, sim.contact_geom)
"""
from .simulator import Simulator
from .controller import ForceController, PID1D, LowPass

# 便捷加载
import os, pickle
import numpy as np

def load_cylinders(data_dir=None):
    """加载标准圆柱 (cy_contact, cz_contact)"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(data_dir, 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['cyl_contact_y'], d['cyl_contact_z']


def load_ball_ref(data_dir=None):
    """加载球刀参考轨迹 → (ball_ref, L)"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    with open(os.path.join(data_dir, 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    ball_ref = d['ball_center_500']
    diffs = np.diff(ball_ref, axis=0)
    L = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    return ball_ref, L


def perturb_cylinder(cz, dx=0.0, dy=0.0, dz=0.0):
    """平移 Z 圆柱"""
    import numpy as np
    from cylinder_def import CylinderDef
    t = np.array([dx, dy, dz])
    return CylinderDef(p1=cz.p1 + t, p2=cz.p2 + t, radius=cz.radius)
