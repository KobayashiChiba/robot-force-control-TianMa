"""
force_control_sim_v5.py — 力控仿真 V5

架构：
  step(F_vec, s_sim, P_cur, dt) → v_3d（世界坐标速度，mm/s）
  主循环: pos += v_3d * dt

核心：
  1. 最近接触点标架 → 力分解
  2. 二次 inverse(Fn,Fo) → 逆推偏移
  3. PID 追逆推归零（力级闭环）
  4. 软限位: smoothstep 混合力反馈 & 位置弹簧
  5. 切向恒速推动
"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from cylinder_def import CylinderDef
from cylinder_geometry_v2 import sample_intersection
from force_mechanics_v2 import compute_point_basis_ortho
from force_field_quadratic import inverse

F_TARGET = -8.0
DT = 0.005
K_POS = 8.0  # 软限位位置弹簧增益


class PID1D:
    """PID 控制器，返回速度 (mm/s)"""
    def __init__(self, Kp=25.0, Ki=0.3, Kd=0.0, dt=DT):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.integral = 0.0
        self.last_err = 0.0
        self.first = True

    def step(self, err):
        if self.first:
            self.integral = 0.0
            self.last_err = err
            self.first = False

        self.integral += err * self.dt
        d_err = (err - self.last_err) / self.dt
        self.last_err = err

        out = self.Kp * err + self.Ki * self.integral + self.Kd * d_err
        return np.clip(out, -160.0, 160.0)


class LowPass:
    def __init__(self, a=1.0):  # a=1.0 = 无滤波
        self.a, self.v, self.ok = a, 0.0, False

    def update(self, x):
        if not self.ok:
            self.v, self.ok = x, True
        else:
            self.v = self.a * x + (1 - self.a) * self.v
        return self.v


class ForceController:
    def __init__(self, cy_std, cz_std):
        self.cy, self.cz = cy_std, cz_std
        self.contact_geom = sample_intersection(cy_std, cz_std, n_samples=2000)
        self.contact_pts = self.contact_geom.sample_pts

        # 加载参考轨迹 (球刀中心)
        with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
            d = pickle.load(f)
        self.ball_ref = d['ball_center_500']   # (N,3)
        diffs = np.diff(self.ball_ref, axis=0)
        self.L = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

        self.pid_n = PID1D(Kp=12.0, Ki=0.15, Kd=0.0, dt=DT)
        self.pid_o = PID1D(Kp=4.0, Ki=0.025, Kd=0.0, dt=DT)
        self.filt_fn = LowPass()
        self.filt_fo = LowPass()

    def _nearest_contact(self, P_cur):
        """找接触曲线上离当前位置最近的点"""
        dists = np.linalg.norm(self.contact_pts - P_cur, axis=1)
        return self.contact_pts[np.argmin(dists)]

    def _nearest_ball_ref(self, P_cur):
        """找ball_ref上离当前位置最近的参考点"""
        dists = np.linalg.norm(self.ball_ref - P_cur, axis=1)
        return self.ball_ref[np.argmin(dists)]

    def step(self, F_vec, s_sim, P_cur, total_steps, dt=DT):
        # 标架：找接触曲线上最近点
        P_ct = self._nearest_contact(P_cur)
        basis = compute_point_basis_ortho(P_ct, self.contact_geom)
        n = basis.normal
        o = basis.ortho
        t = basis.tangent

        # 力分解 + 滤波
        Fn = np.dot(F_vec, n)
        Fo = np.dot(F_vec, o)
        Fn_f = self.filt_fn.update(Fn)
        Fo_f = self.filt_fo.update(Fo)

        # 最近参考点（球刀在哪，参考点就在哪）
        P_ref = self._nearest_ball_ref(P_cur)
        dn_actual = np.dot(P_cur - P_ref, n)
        do_actual = np.dot(P_cur - P_ref, o)

        # 逆推 + 软限位混合
        # 0~1mm: 纯力反馈; 1~2mm: smoothstep过渡; >2mm: 纯位置弹簧
        dn_target, db_target = inverse(Fn_f, Fo_f)

        def _blend(d_abs, force_err, pos_err):
            if d_abs < 1.0:
                w = 0.0
            elif d_abs > 2.0:
                w = 1.0
            else:
                t_val = (d_abs - 1.0) / 1.0
                w = t_val * t_val * (3 - 2 * t_val)  # smoothstep
            return (1 - w) * force_err + w * pos_err

        err_n = _blend(abs(dn_actual), -dn_target, -K_POS * dn_actual)
        err_o = _blend(abs(do_actual), -db_target, -K_POS * do_actual)

        vn = self.pid_n.step(err_n)
        vb = self.pid_o.step(err_o)

        # 切向推动
        v_fwd = self.L / (total_steps * dt)
        v_raw = vn * n + vb * o + v_fwd * t

        return v_raw


# ============================================================
# 仿真运行（外部脚本用）
# ============================================================

def load_standard_cylinders():
    with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
        d = pickle.load(f)
    return d['cyl_contact_y'], d['cyl_contact_z']


def translate_cz(cz, dx=0, dy=0, dz=0):
    t = np.array([dx, dy, dz])
    return CylinderDef(p1=cz.p1 + t, p2=cz.p2 + t, radius=cz.radius)


def rotate_cz(cz, axis, deg):
    a = np.radians(deg)
    R = {
        'x': np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]),
        'y': np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]),
        'z': np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]),
    }[axis]
    ctr = (cz.p1 + cz.p2) / 2
    L = np.linalg.norm(cz.p2 - cz.p1)
    d = R @ np.array([0, 0, 1])
    return CylinderDef(p1=ctr - L/2 * d, p2=ctr + L/2 * d, radius=cz.radius)
