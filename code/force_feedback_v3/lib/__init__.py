"""
force_feedback_v3 — V5 力控仿真库

用法:
    from force_feedback_v3.lib import Simulator, ForceController, load_cylinders, perturb_cylinder

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
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    with open(os.path.join(data_dir, 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['cyl_contact_y'], d['cyl_contact_z']


def load_ball_ref(data_dir=None):
    """加载球刀参考轨迹 → (ball_ref, L)"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    with open(os.path.join(data_dir, 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    ball_ref = d['ball_center_500']
    diffs = np.diff(ball_ref, axis=0)
    L = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    return ball_ref, L


def perturb_cylinder(cz, dx=0.0, dy=0.0, dz=0.0):
    """平移 Z 圆柱（保留兼容）"""
    from .cylinder_def import CylinderDef
    t = np.array([dx, dy, dz])
    return CylinderDef(p1=cz.p1 + t, p2=cz.p2 + t, radius=cz.radius)


def perturb_endpoints(cyl, dp1, dp2):
    """圆柱端点偏移 → 新圆柱（轴线和位置都变）"""
    from .cylinder_def import CylinderDef
    return CylinderDef(p1=cyl.p1 + dp1, p2=cyl.p2 + dp2, radius=cyl.radius)


def generate_error_cylinders(cy, cz, rng):
    """生成一对误差圆柱：每个端点 ±1mm 随机偏移，共12参数"""
    dp1_z = rng.uniform(-1, 1, 3)  # Z圆柱端点1
    dp2_z = rng.uniform(-1, 1, 3)  # Z圆柱端点2
    dp1_y = rng.uniform(-1, 1, 3)  # Y圆柱端点1
    dp2_y = rng.uniform(-1, 1, 3)  # Y圆柱端点2
    return perturb_endpoints(cz, dp1_z, dp2_z), perturb_endpoints(cy, dp1_y, dp2_y)
