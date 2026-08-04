"""
controller.py — V5 力控控制器

核心三步：
  1. 最近接触点标架 → 力分解 (Fn, Fo)
  2. quadratic inverse(Fn,Fo) → 逆推偏移 (dn_target, db_target)
  3. PID 追逆推归零 + smoothstep 软限位 → 速度输出

用法:
    ctrl = ForceController(ball_ref, L, contact_geom)
    v_3d = ctrl.step(F_vec, P_cur, total_steps, dt)
"""
import numpy as np
from .force_field_physical_v2 import inverse
from .force_mechanics import compute_point_basis_ortho


# === 默认参数 ===
F_TARGET = -8.0      # 目标法向力 (N)
DT = 0.005           # 仿真步长 (s)
K_POS = 8.0          # 软限位位置弹簧增益
SOFT_LO = 2.0        # 软限位起始 (mm)
SOFT_HI = 3.0        # 软限位饱和 (mm)


class PID1D:
    """一维 PID 控制器，输出速度 (mm/s)"""
    def __init__(self, Kp, Ki=0.0, Kd=0.0, dt=DT):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
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

    def reset(self):
        self.integral = 0.0
        self.last_err = 0.0
        self.first = True


class LowPass:
    """一阶低通滤波，a=1.0 为无滤波"""
    def __init__(self, a=1.0):
        self.a, self.v, self.ok = a, 0.0, False

    def update(self, x):
        if not self.ok:
            self.v, self.ok = x, True
        else:
            self.v = self.a * x + (1 - self.a) * self.v
        return self.v


class ForceController:
    """V5 力控控制器

    Parameters
    ----------
    ball_ref : (N,3) array
        球刀中心参考轨迹
    L : float
        参考轨迹弧长 (mm)
    contact_geom : Geom
        接触曲线几何（sample_intersection 输出）
    kp_n, ki_n : float
        n 方向 PID 增益
    kp_o, ki_o : float
        o 方向 PID 增益
    k_pos : float
        软限位弹簧增益
    soft_lo, soft_hi : float
        软限位过渡区间 (mm)
    filt_a : float
        力低通滤波系数 (1.0=无滤波)
    """

    def __init__(self, ball_ref, L, contact_geom,
                 kp_n=25.0, ki_n=0.3,
                 kp_o=4.0, ki_o=0.025,
                 k_pos=K_POS, soft_lo=SOFT_LO, soft_hi=SOFT_HI,
                 filt_a=1.0):
        self.ball_ref = ball_ref
        self.L = L
        self.contact_geom = contact_geom
        self.contact_pts = contact_geom.sample_pts

        self.k_pos = k_pos
        self.soft_lo = soft_lo
        self.soft_hi = soft_hi

        self.pid_n = PID1D(Kp=kp_n, Ki=ki_n, dt=DT)
        self.pid_o = PID1D(Kp=kp_o, Ki=ki_o, dt=DT)
        self.filt_fn = LowPass(a=filt_a)
        self.filt_fo = LowPass(a=filt_a)

    # === 内部 ===

    def _nearest_contact(self, P_cur):
        dists = np.linalg.norm(self.contact_pts - P_cur, axis=1)
        return self.contact_pts[np.argmin(dists)]

    def _nearest_ball_ref(self, P_cur):
        dists = np.linalg.norm(self.ball_ref - P_cur, axis=1)
        return self.ball_ref[np.argmin(dists)]

    def _blend(self, d_abs, force_err, pos_err):
        """smoothstep 混合力反馈与位置弹簧"""
        lo, hi = self.soft_lo, self.soft_hi
        if d_abs < lo:
            w = 0.0
        elif d_abs > hi:
            w = 1.0
        else:
            t = (d_abs - lo) / (hi - lo)
            w = t * t * (3 - 2 * t)
        return (1 - w) * force_err + w * pos_err

    # === 核心 ===

    def step(self, F_vec, P_cur, total_steps, dt=DT):
        """计算一步控制输出

        Parameters
        ----------
        F_vec : (3,) array
            当前力向量 (N) — 来自 Simulator.step()
        P_cur : (3,) array
            球刀当前位置 (mm)
        total_steps : int
            总仿真步数（用于计算切向速度）
        dt : float
            步长 (s)

        Returns
        -------
        v_3d : (3,) array
            世界坐标速度 (mm/s)
        """
        # 1. 标架
        P_ct = self._nearest_contact(P_cur)
        basis = compute_point_basis_ortho(P_ct, self.contact_geom)
        n, o, t = basis.normal, basis.ortho, basis.tangent

        # 2. 力分解 + 滤波
        Fn = np.dot(F_vec, n)
        Fo = np.dot(F_vec, o)
        Fn_f = self.filt_fn.update(Fn)
        Fo_f = self.filt_fo.update(Fo)

        # 3. 最近参考点 → 局部偏移
        P_ref = self._nearest_ball_ref(P_cur)
        dn_actual = np.dot(P_cur - P_ref, n)
        do_actual = np.dot(P_cur - P_ref, o)

        # 4. 逆推 + 软限位混合
        dn_target, db_target = inverse(Fn_f, Fo_f)
        err_n = self._blend(abs(dn_actual), -dn_target, -self.k_pos * dn_actual)
        err_o = self._blend(abs(do_actual), -db_target, -self.k_pos * do_actual)

        # 5. PID → 速度
        vn = self.pid_n.step(err_n)
        vb = self.pid_o.step(err_o)

        # 6. 切向推动 + 合成
        v_fwd = self.L / (total_steps * dt)
        v_3d = vn * n + vb * o + v_fwd * t

        return v_3d
