"""
simulator.py — 力控仿真环境

封装接触力 + 摩擦力 + 随机噪声，统一接口，参数可配。
"""
import pickle, numpy as np
from .sphere_contact import sphere_contact_force
from .force_mechanics import compute_point_basis_ortho
from .cylinder_geometry import sample_intersection


class Simulator:
    """力控仿真器：接触力 + 库仑摩擦 + 高斯噪声

    Parameters
    ----------
    cy, cz : CylinderDef
        标准圆柱几何（控制器用的参考几何）
    mu : float
        库仑摩擦系数，默认 0.2
    sigma : float
        力噪声标准差 (N)，默认 0.5
    seed : int or None
        随机种子，None 表示不固定
    """

    def __init__(self, cy, cz, mu=0.2, sigma=0.5, seed=None):
        self.cy, self.cz = cy, cz
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

        # 预计算接触曲线（用于标架）
        self.contact_geom = sample_intersection(cy, cz, n_samples=2000)
        self.contact_pts = self.contact_geom.sample_pts

    # === 配置 ===

    def set_friction(self, mu):
        """设置摩擦系数"""
        self.mu = mu

    def set_noise(self, sigma):
        """设置噪声标准差"""
        self.sigma = sigma

    def reset_rng(self, seed=None):
        """重置随机状态"""
        self.rng = np.random.RandomState(seed)

    # === 核心 ===

    def step(self, pos, v_prev, cy_err=None, cz_err=None):
        """计算一步仿真力

        Parameters
        ----------
        pos : (3,) array
            球刀当前位置 (mm)
        v_prev : (3,) array
            上一帧速度 (mm/s)，用于摩擦方向
        cy_err, cz_err : CylinderDef or None
            误差圆柱（模拟真实工件）。None 则用标准圆柱。

        Returns
        -------
        F_meas : (3,) array
            含接触力+摩擦+噪声的总力 (N)
        F_raw : (3,) array
            纯接触力 (N)
        basis : Basis
            当前接触点的正交标架 {normal, tangent, ortho}
        """
        cy = cy_err if cy_err is not None else self.cy
        cz = cz_err if cz_err is not None else self.cz

        # 1. 接触力
        F_raw, _ = sphere_contact_force(pos, cz, cy)

        # 2. 标架（用于力分解，控制器也用到）
        P_ct = self._nearest_contact(pos)
        basis = compute_point_basis_ortho(P_ct, self.contact_geom)

        # 3. 摩擦力 — 垂直于接触力方向
        f_norm = np.linalg.norm(F_raw)
        if f_norm < 1e-6 or np.linalg.norm(v_prev) < 1e-6:
            F_fric = np.zeros(3)
        else:
            # 摩擦面法向 = 接触力方向
            f_dir = F_raw / f_norm
            # 速度投影到摩擦切平面
            v_tang = v_prev - np.dot(v_prev, f_dir) * f_dir
            vt_norm = np.linalg.norm(v_tang)
            if vt_norm < 1e-6:
                F_fric = np.zeros(3)
            else:
                Fn = np.dot(F_raw, basis.normal)
                F_fric = self.mu * abs(Fn) * (-v_tang / vt_norm)

        # 4. 噪声
        F_noise = self.rng.randn(3) * self.sigma

        F_meas = F_raw + F_fric + F_noise
        return F_meas, F_raw, F_fric, F_noise, basis

    def _nearest_contact(self, P_cur):
        """找接触曲线上离当前位置最近的点"""
        dists = np.linalg.norm(self.contact_pts - P_cur, axis=1)
        return self.contact_pts[np.argmin(dists)]

    # === 便捷方法 ===

    def load_ball_ref(self, data_path):
        """加载球刀中心参考轨迹

        Parameters
        ----------
        data_path : str
            force_model.pkl 路径

        Returns
        -------
        ball_ref : (N,3) array
        L : float
            参考轨迹弧长 (mm)
        """
        with open(data_path, 'rb') as f:
            d = pickle.load(f)
        ball_ref = d['ball_center_500']
        diffs = np.diff(ball_ref, axis=0)
        L = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
        return ball_ref, L
