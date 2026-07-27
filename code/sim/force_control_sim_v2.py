"""
force_control_sim_v2.py — 力控仿真（融合V2力场模型 + 导纳控制）

========== 信号链路 ==========

[仿真真实力]
  球心实际位置 → 算偏移(dn, db) → force_field.predict(dn, db) → F_vec(3D, 世界坐标)
  
[传感器→控制器]
  F_vec → compute_point_basis(Pc) → 局部标架(n, b, t)
        → decompose_force → (Ft, Fn, Fo)
        → force_field.inverse(Fn, Fo, p) → (dn_est, db_est)

[位置修正]
  导纳控制：ΔF = desired_Fn - Fn_meas → Δdn = admittance(ΔF)
  复法向修正：Δdb = -K_fo * Fo_meas
  位置更新：pos += Δdn·n + Δdb·b + tangent_speed·t·dt

最终效果：力反馈闭环 → 球心自动贴回标准位置

========== 仿真对比 ==========
  - 无误差仿真：理想圆柱 + 力控 → 验证闭环收敛
  - 有误差仿真：Z轴倾斜 → 观察力控如何应对几何误差
"""

import sys, os
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
sys.path.insert(0, os.path.join(_sdir, '..', 'sim'))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import pickle
import warnings
warnings.filterwarnings("ignore")

from cylinder_def import CylinderDef
from cylinder_geometry_v2 import sample_intersection, GeomV2
from contact_frame_v2 import compute_frame
from force_mechanics_v2 import decompose_force, Basis, ForceDecomp
from force_field_quadratic import calibrate as calib_quad, inverse as inv_quad, predict as pred_quad, get_base

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 低通滤波器
# ============================================================
class LowPassFilter:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.filtered = np.zeros(3)
        self.initialized = False

    def update(self, value):
        if not self.initialized:
            self.filtered = np.array(value, dtype=float)
            self.initialized = True
        else:
            self.filtered = self.alpha * np.array(value) + (1 - self.alpha) * self.filtered
        return self.filtered


# ============================================================
# 2. 导纳控制器（一维，法向）
# ============================================================
class AdmittanceController:
    def __init__(self, mass=0.6, damping=70.0, dt=0.005):
        self.M, self.D, self.dt = mass, damping, dt
        self.vel = 0.0

    def compute(self, force_error):
        accel = (force_error - self.D * self.vel) / self.M
        self.vel += accel * self.dt
        delta = self.vel * self.dt
        max_delta = 0.8
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
            self.vel = delta / self.dt
        return delta


# ============================================================
# 3. 辅助
# ============================================================
def quat_to_rotmat(q):
    return R.from_quat(q).as_matrix()

def rotmat_to_quat(Rm):
    return R.from_matrix(Rm).as_quat()


# ============================================================
# 4. 力场模型封装（基于 force_field_quadratic）
# ============================================================
class ForceFieldModel:
    """力场模型：给定偏移(dn, db)，返回3D力向量"""
    def __init__(self):
        self.initialized = False

    def _ensure_calib(self):
        if not self.initialized:
            try:
                cal_data = np.load(os.path.join(_sdir, '..', 'data', 'force_field_quadratic.npz'))
                self.c_dfn = cal_data['c_dfn']
                self.c_dfo = cal_data['c_dfo']
                idxs = cal_data['base_indices']
                fns = cal_data['base_Fn0']
                fos = cal_data['base_Fo0']
                self._base = [(int(idxs[i]), float(fns[i]), float(fos[i])) for i in range(len(idxs))]
            except FileNotFoundError:
                calib_quad()
                return self._ensure_calib()
            self.initialized = True

    def get_base(self, p_idx):
        self._ensure_calib()
        best = min(self._base, key=lambda x: abs(x[0] - p_idx))
        return best[1], best[2]

    def predict_3d(self, dn, db, n_vec, b_vec, p_idx=0):
        """返回世界坐标下的3D力向量"""
        self._ensure_calib()
        Fn0, Fo0 = self.get_base(p_idx)
        c_fn = self.c_dfn
        c_fo = self.c_dfo
        dFn = c_fn[0]*dn + c_fn[1]*db + c_fn[2]*dn**2 + c_fn[3]*dn*db + c_fn[4]*db**2
        dFo = c_fo[0]*dn + c_fo[1]*db + c_fo[2]*dn**2 + c_fo[3]*dn*db + c_fo[4]*db**2
        return (Fn0 + dFn) * n_vec + (Fo0 + dFo) * b_vec

    def inverse(self, Fn_meas, Fo_meas, p_idx=0):
        """从测量值逆推偏移"""
        return inv_quad(Fn_meas, Fo_meas, p_idx)


# ============================================================
# 5. 力控仿真主函数
# ============================================================
def run_force_control(cyl_y, cyl_z, tool_radius=4.0, n_steps=3000, dt=0.005,
                      desired_fn=-2.0, preload_n=0.08, filt_alpha=0.15,
                      K_fo=0.15, label=""):
    """
    Args:
        desired_fn: 期望法向力 (负值)
        preload_n: 初始预压量 (mm, 沿法向)
        K_fo: 复法向力修正增益
    """
    # ── 参考轨迹 ──
    ref_ideal = sample_intersection(cyl_y, cyl_z, n_samples=500, N_curve=250).sample_pts
    if np.linalg.norm(ref_ideal[-1] - ref_ideal[0]) > 1e-9:
        ref_ideal = np.vstack([ref_ideal, ref_ideal[0:1]])

    offset_pts = []
    for P in ref_ideal:
        frame = compute_frame(P, cyl_y, cyl_z)
        n_inner = -frame.normal
        offset_pts.append(P + tool_radius * n_inner)
    ref_offset = np.array(offset_pts)
    ref_offset[-1] = ref_offset[0].copy()

    diff = np.diff(ref_offset, axis=0)
    chord_len = np.linalg.norm(diff, axis=1)
    total_arc = np.sum(chord_len)
    s = np.linspace(0, total_arc, len(ref_offset))
    fx = interp1d(s, ref_offset[:, 0], kind='linear')
    fy = interp1d(s, ref_offset[:, 1], kind='linear')
    fz = interp1d(s, ref_offset[:, 2], kind='linear')

    # ── 初始化 ──
    adm = AdmittanceController(mass=0.6, damping=70.0, dt=dt)
    tang_speed = total_arc / (n_steps * dt)
    flt = LowPassFilter(alpha=filt_alpha)
    fmodel = ForceFieldModel()

    P0_offset = ref_offset[0]
    frame0 = compute_frame(ref_ideal[0], cyl_y, cyl_z)
    n0 = -frame0.normal
    t0 = frame0.tangent

    robot_pos = P0_offset + preload_n * n0
    y0 = np.cross(t0, frame0.normal)
    y0 /= np.linalg.norm(y0)
    z0 = np.cross(t0, y0)
    Rmat0 = np.column_stack([t0, y0, z0])
    quat = rotmat_to_quat(Rmat0)
    prev_pos = robot_pos.copy()

    trajectory = []
    force_log = []
    dn_log = []
    db_log = []

    for step in range(n_steps):
        t_frac = step / (n_steps - 1)
        s_cur = t_frac * total_arc
        P_ref_offset = np.array([fx(s_cur), fy(s_cur), fz(s_cur)])

        idx_ideal = int(t_frac * (len(ref_ideal) - 1))
        P_ideal = ref_ideal[idx_ideal]
        frame = compute_frame(P_ideal, cyl_y, cyl_z)
        t_vec = frame.tangent
        n_inner = -frame.normal
        b_vec = frame.radial_z
        b_vec /= np.linalg.norm(b_vec) if np.linalg.norm(b_vec) > 1e-12 else 1.0

        # ── 算实际偏移 (dn, db) — 仿真真实力 ──
        delta_pos = robot_pos - P_ref_offset
        dn_real = np.dot(delta_pos, n_inner)
        db_real = np.dot(delta_pos, b_vec)

        # ── 力模型 → 传感器读数 F_vec(3D) ──
        F_vec_true = fmodel.predict_3d(dn_real, db_real, n_inner, b_vec, idx_ideal)
        F_vec_filt = flt.update(F_vec_true)

        # ── 力分解 → (Fn_meas, Fo_meas) ──
        Fn_meas = np.dot(F_vec_filt, n_inner)
        Fo_meas = np.dot(F_vec_filt, b_vec)

        # ── 逆推 → (dn_est, db_est) ──
        try:
            dn_est, db_est = fmodel.inverse(Fn_meas, Fo_meas, idx_ideal)
        except Exception:
            dn_est, db_est = 0.0, 0.0

        # ── 导纳控制 ──
        force_error = desired_fn - Fn_meas
        delta_n_cmd = adm.compute(force_error)

        # ── 位置更新 ──
        delta_t = t_vec * tang_speed * dt
        delta_n_vec = n_inner * delta_n_cmd
        delta_b_vec = b_vec * (-K_fo * Fo_meas)

        # P兜底：向参考轨迹靠拢
        delta_pos_corr = (P_ref_offset - robot_pos) * 0.06

        robot_pos += delta_t + delta_n_vec + delta_b_vec + delta_pos_corr
        prev_pos = robot_pos.copy()

        # ── 姿态跟随切线 ──
        y_new = np.cross(t_vec, frame.radial_z)
        y_new /= np.linalg.norm(y_new) if np.linalg.norm(y_new) > 1e-12 else 1.0
        z_new = np.cross(t_vec, y_new)
        Rmat_new = np.column_stack([t_vec, y_new, z_new])
        quat = rotmat_to_quat(Rmat_new)

        trajectory.append(robot_pos.copy())
        force_log.append(Fn_meas)
        dn_log.append(dn_real)
        db_log.append(db_real)

    trajectory = np.array(trajectory)
    force_log = np.array(force_log)
    dn_log = np.array(dn_log)
    db_log = np.array(db_log)

    print(f'  [{label}] |Fn|均值={np.mean(np.abs(force_log)):.2f}N  std={np.std(force_log):.2f}N  '
          f'dn_max={np.max(np.abs(dn_log)):.3f}mm  db_max={np.max(np.abs(db_log)):.3f}mm')

    return trajectory, force_log, dn_log, db_log, ref_ideal, ref_offset


# ============================================================
# 6. 对比绘图
# ============================================================
def plot_comparison(ref0, off0, traj0, ref1, off1, traj1,
                    cy0, cz0, cy1, cz1,
                    force0, force1, dn0, db0, dn1, db1, tilt_angle):
    fig = plt.figure(figsize=(18, 14))

    # ── 3D 轨迹 ──
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(ref0[:,0], ref0[:,1], ref0[:,2], 'gray', ls='--', lw=1, alpha=0.5, label='理想交线')
    ax.plot(ref1[:,0], ref1[:,1], ref1[:,2], 'gray', ls=':',  lw=1, alpha=0.5, label=f'倾斜{int(tilt_angle)}°交线')
    ax.plot(traj0[:,0], traj0[:,1], traj0[:,2], 'blue', lw=1.5, label='力控轨迹(无误差)')
    ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], 'red',  lw=1.5, label=f'力控轨迹(倾斜{int(tilt_angle)}°)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D轨迹对比')
    ax.legend(fontsize=7)

    # ── 力曲线 ──
    ax2 = fig.add_subplot(222)
    ax2.plot(force0, 'b-', lw=0.8, label='无误差')
    ax2.plot(force1, 'r-', lw=0.8, label=f'倾斜{int(tilt_angle)}°')
    ax2.axhline(-2.0, color='gray', ls='--', lw=0.5)
    ax2.set_ylabel('Fn (N)'); ax2.set_xlabel('仿真步数')
    ax2.set_title(f'法向力 (目标=-2N)'); ax2.legend()
    ax2.grid(alpha=0.3)

    # ── dn(db)偏移 ──
    ax3 = fig.add_subplot(223)
    ax3.plot(dn0, 'b-', lw=0.6, alpha=0.7, label='dn (无误差)')
    ax3.plot(dn1, 'r-', lw=0.6, alpha=0.7, label=f'dn (倾斜{int(tilt_angle)}°)')
    ax3.axhline(0, color='gray', lw=0.3)
    ax3.set_ylabel('dn (mm)'); ax3.set_title('法向偏移')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(224)
    ax4.plot(db0, 'b-', lw=0.6, alpha=0.7, label='db (无误差)')
    ax4.plot(db1, 'r-', lw=0.6, alpha=0.7, label=f'db (倾斜{int(tilt_angle)}°)')
    ax4.axhline(0, color='gray', lw=0.3)
    ax4.set_ylabel('db (mm)'); ax4.set_xlabel('仿真步数')
    ax4.set_title('复法向偏移')
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    fig.suptitle(f'V2力场模型 + 导纳控制 去毛刺力控仿真 (Z轴倾斜{int(tilt_angle)}°)', fontsize=14)
    fig.tight_layout()
    plt.show(block=True)


# ============================================================
# 7. 生成圆柱
# ============================================================
def generate_cylinders(tilt_angle_deg=0.0):
    cyl_y = CylinderDef(p1=np.array([0, -20, 0]), p2=np.array([0, 20, 0]), radius=10.0)
    angle = np.radians(tilt_angle_deg)
    dir_z = np.array([np.sin(angle), 0, np.cos(angle)])
    center = np.array([27, 0, 0])
    length = 40.0
    p1 = center - (length / 2) * dir_z
    p2 = center + (length / 2) * dir_z
    cyl_z = CylinderDef(p1=p1, p2=p2, radius=20.0)
    return cyl_y, cyl_z


# ============================================================
# 8. 主程序
# ============================================================
def main():
    TILT = 5.0
    print(f"V2力场模型力控仿真 (Z轴倾斜 {int(TILT)}°)")
    print("=" * 50)

    print("\n[1] 力场标定...")
    calib_quad()

    print("\n[2] 无误差仿真...")
    cy0, cz0 = generate_cylinders(0.0)
    traj0, f0, dn0, db0, ref0, off0 = run_force_control(
        cy0, cz0, desired_fn=-2.0, preload_n=0.08, label='无误差')

    print(f"\n[3] 倾斜 {int(TILT)}° 仿真...")
    cy1, cz1 = generate_cylinders(TILT)
    traj1, f1, dn1, db1, ref1, off1 = run_force_control(
        cy1, cz1, desired_fn=-2.0, preload_n=0.08, label=f'倾斜{int(TILT)}°')

    print("\n[4] 生成对比图...")
    plot_comparison(ref0, off0, traj0, ref1, off1, traj1,
                    cy0, cz0, cy1, cz1,
                    f0, f1, dn0, db0, dn1, db1, TILT)


if __name__ == "__main__":
    main()
