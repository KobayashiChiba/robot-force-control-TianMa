"""
force_control_sim_v5.py — 力控仿真 V5

架构：
  step(F_vec, s_sim, P_cur, dt) → v_3d（世界坐标速度，mm/s）
  主循环: pos += v_3d * dt

改动（V4→V5）:
  1. step 加 P_cur 参数 → 实时投影偏移
  2. 内部加 ball_ref → 参考轨迹点 + 切线方向
  3. Admittance1D → PID1D 追位置误差 (inverse 保留做非线性解耦)
  4. 切向推动改参考轨迹切线（不再是接触曲线切线）
  5. 硬限位 + anti-windup

保留:
  低通滤波 / compute_point_basis_ortho / 真实力 sphere_contact_force
  速度输出模式 / inverse 解耦 / o 方向暂关

配合 run_sim_v5.py 使用
"""
import sys, os, pickle, numpy as np, time
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from cylinder_def import CylinderDef
from cylinder_geometry_v2 import sample_intersection
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from force_field_fixed import inverse

F_TARGET = -8.0
DT = 0.005
LIMIT_N = 2.5
LIMIT_O = 1.0


class PID1D:
    """PID 控制器，返回速度 (mm/s)"""
    def __init__(self, Kp=25.0, Ki=0.3, Kd=0.0, dt=DT):
        self.Kp, self.Ki, self.Kd, self.dt = Kp, Ki, Kd, dt
        self.integral = 0.0
        self.last_err = 0.0
        self.first = True
        self.frozen = False

    def step(self, err):
        if self.first:
            self.integral = 0.0
            self.last_err = err
            self.first = False

        # anti-windup: 限位激活时冻结积分
        if not self.frozen:
            self.integral += err * self.dt

        d_err = (err - self.last_err) / self.dt
        self.last_err = err

        out = self.Kp * err + self.Ki * self.integral + self.Kd * d_err
        return np.clip(out, -160.0, 160.0)

    def reset(self):
        self.integral = 0.0
        self.last_err = 0.0
        self.first = True
        self.frozen = False


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
        # 弧长参数
        diffs = np.diff(self.ball_ref, axis=0)
        segs = np.sqrt(np.sum(diffs**2, axis=1))
        self.arc = np.concatenate([[0.0], np.cumsum(segs)])
        self.L = self.arc[-1]

        self.pid_n = PID1D()
        self.pid_o = PID1D(Kp=8.0, Ki=0.05, Kd=0.0, dt=DT)
        self.filt_fn = LowPass()
        self.filt_fo = LowPass()

    def _ball_ref(self, s):
        """s ∈ [0, 1) → 球刀中心参考点 (线性插值)"""
        s = s % 1.0
        target = s * self.L
        i = np.searchsorted(self.arc, target, side='right') - 1
        i = max(0, min(i, len(self.ball_ref) - 2))
        t = (target - self.arc[i]) / max(1e-12, self.arc[i+1] - self.arc[i])
        t = max(0.0, min(1.0, t))
        return (1 - t) * self.ball_ref[i] + t * self.ball_ref[i+1]

    def _ref_tangent(self, s):
        """参考轨迹切线方向 (单位向量)"""
        ds = 0.001
        p0 = self._ball_ref(s)
        p1 = self._ball_ref(s + ds)
        t = p1 - p0
        nrm = np.linalg.norm(t)
        return t / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])

    def _nearest_contact(self, P_cur):
        """找接触曲线上离当前位置最近的点"""
        dists = np.linalg.norm(self.contact_pts - P_cur, axis=1)
        return self.contact_pts[np.argmin(dists)]

    def step(self, F_vec, s_sim, P_cur, total_steps, dt=DT):
        # 标架：找当前位置在接触曲线上最近的点来计算
        P_ct = self._nearest_contact(P_cur)
        basis = compute_point_basis_ortho(P_ct, self.contact_geom)
        n = basis.normal
        o = basis.ortho

        # 力分解 + 滤波
        Fn = np.dot(F_vec, n)
        Fo = np.dot(F_vec, o)
        Fn_f = self.filt_fn.update(Fn)
        Fo_f = self.filt_fo.update(Fo)

        # 实时偏移（相对参考轨迹）
        P_ref = self._ball_ref(s_sim)
        dn_actual = np.dot(P_cur - P_ref, n)
        do_actual = np.dot(P_cur - P_ref, o)

        # PID 直接追逆推值——逆推归零时力=目标
        # dn_actual 用于 db 反推（Fo ∝ dn_actual·db）
        dn_target, db_target = inverse(Fn_f, Fo_f, dn_actual)
        vn = self.pid_n.step(dn_target)
        vb = self.pid_o.step(db_target - do_actual)

        # 切向推动（参考轨迹切线）
        t_ref = self._ref_tangent(s_sim)
        v_fwd = self.L / (total_steps * dt)
        vt = v_fwd

        v_raw = vn * n + vb * o + vt * t_ref

        # === 硬限位 ===
        P_next = P_cur + v_raw * dt
        d_n = np.dot(P_next - P_ref, n)
        d_o = np.dot(P_next - P_ref, o)

        if abs(d_n) > LIMIT_N:
            sign = 1.0 if d_n > 0 else -1.0
            vn_clamped = (sign * LIMIT_N - np.dot(P_cur - P_ref, n)) / dt
            v_raw += (vn_clamped - vn) * n
            self.pid_n.frozen = True
        else:
            self.pid_n.frozen = False

        if abs(d_o) > LIMIT_O:
            sign = 1.0 if d_o > 0 else -1.0
            vo_clamped = (sign * LIMIT_O - np.dot(P_cur - P_ref, o)) / dt
            v_raw += (vo_clamped - vb) * o
            self.pid_o.frozen = True
        else:
            self.pid_o.frozen = False

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
