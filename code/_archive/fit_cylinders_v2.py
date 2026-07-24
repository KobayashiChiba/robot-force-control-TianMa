import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. 读取数据
# ============================================================
path = r'C:\Users\KCserver\AppData\Local\hermes\cache\documents\doc_f08acc09a0d8_球刀中心点及轮廓轨迹点.xlsx'
df = pd.read_excel(path)
px, py, pz = df['x'].values, df['y'].values, df['z'].values
pts = np.column_stack([px, py, pz])
N = len(px)
print(f'实测点数: {N}')

out_dir = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

# ============================================================
# 2. 原始方法（投影圆拟合）做对比基准
# ============================================================
Y0_mean = py.mean()

# Y方向: XZ平面圆拟合
Y1 = px**2 + pz**2
A1, B1, C1 = np.linalg.lstsq(np.column_stack([px, pz, np.ones_like(px)]), Y1, rcond=None)[0]
X0_cyl1_old, Z0_cyl1_old = A1/2, B1/2
r1_old = np.sqrt(C1 + X0_cyl1_old**2 + Z0_cyl1_old**2)

# Z方向: XY平面圆拟合
Y2 = px**2 + py**2 - 2*py*Y0_mean
A2, B2 = np.linalg.lstsq(np.column_stack([px, np.ones_like(px)]), Y2, rcond=None)[0]
X0_cyl2_old = A2/2
r2_old = np.sqrt(B2 + X0_cyl2_old**2 + Y0_mean**2)

print('\n===== 原始方法（投影圆拟合）=')
print(f'Y圆柱: X₀={X0_cyl1_old:.4f}, Z₀={Z0_cyl1_old:.4f}, r₁={r1_old:.4f}')
res1_old = np.sqrt((px - X0_cyl1_old)**2 + (pz - Z0_cyl1_old)**2) - r1_old
print(f'  RMS={np.sqrt(np.mean(res1_old**2)):.4f}, max|r|={np.max(np.abs(res1_old)):.4f}')
print(f'Z圆柱: X₀={X0_cyl2_old:.4f}, Y₀={Y0_mean:.4f}, r₂={r2_old:.4f}')
res2_old = np.sqrt((px - X0_cyl2_old)**2 + (py - Y0_mean)**2) - r2_old
print(f'  RMS={np.sqrt(np.mean(res2_old**2)):.4f}, max|r|={np.max(np.abs(res2_old)):.4f}')

# ============================================================
# 3. 完整圆柱拟合（非线性最小二乘）
# ============================================================
def cylinder_residuals(params, pts):
    """残差: 点到圆柱面的有向距离"""
    theta, phi, x0, y0, z0, r = params
    a = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                  np.cos(theta)])
    p0 = np.array([x0, y0, z0])
    d = pts - p0
    cross = np.cross(d, a)
    dist = np.sqrt(np.sum(cross**2, axis=1))
    return dist - r

def fit_cylinder(pts, init_theta, init_phi, init_x0, init_y0, init_z0, init_r):
    """完整圆柱拟合"""
    x0_vec = [init_theta, init_phi, init_x0, init_y0, init_z0, init_r]
    lb = [-np.pi, -np.pi, -200, -200, -200, 1]
    ub = [np.pi, np.pi, 200, 200, 200, 50]
    result = least_squares(cylinder_residuals, x0_vec, args=(pts,),
                           bounds=(lb, ub), method='trf',
                           loss='linear', ftol=1e-12, xtol=1e-12, max_nfev=10000)
    theta, phi, x0, y0, z0, r = result.x
    a = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                  np.cos(theta)])
    # 确保方向向上（方便比较）
    # 计算残差
    d = pts - np.array([x0, y0, z0])
    cross = np.cross(d, a)
    dist = np.sqrt(np.sum(cross**2, axis=1))
    res = dist - r
    return a, np.array([x0, y0, z0]), r, res, result

# --- 拟合Y圆柱 ---
# 初始值: 从原始方法 + 方向∥Y (θ=π/2, φ=π/2)
init_theta1 = np.pi/2  # ∥Y → θ=90°
init_phi1 = np.pi/2    # φ=90° → ay正方向
print('\n===== 完整圆柱拟合 =====')
print('拟合Y圆柱 (初始方向∥Y)...')
a1, p01, r1, res1, res1_full = fit_cylinder(
    pts, init_theta1, init_phi1,
    X0_cyl1_old, Y0_mean, Z0_cyl1_old, r1_old
)
print(f'  轴线方向: ({a1[0]:.6f}, {a1[1]:.6f}, {a1[2]:.6f})')
print(f'  轴线上点: ({p01[0]:.3f}, {p01[1]:.3f}, {p01[2]:.3f})')
print(f'  半径: {r1:.4f}')
print(f'  RMS残差: {np.sqrt(np.mean(res1**2)):.4f}')
print(f'  最大残差: {np.max(np.abs(res1)):.4f}')

# 检查方向是否接近∥Y
ideal_y = np.array([0, 1, 0])
dot_y1 = np.abs(np.dot(a1, ideal_y))
angle_y1 = np.degrees(np.arccos(np.clip(dot_y1, -1, 1)))
print(f'  与Y轴夹角: {angle_y1:.2f}°')

# --- 拟合Z圆柱 ---
# 初始值: 从原始方法 + 方向∥Z (θ=0)
init_theta2 = 0      # ∥Z → θ=0°
init_phi2 = 0
print('\n拟合Z圆柱 (初始方向∥Z)...')
a2, p02, r2, res2, res2_full = fit_cylinder(
    pts, init_theta2, init_phi2,
    X0_cyl2_old, Y0_mean, -60, r2_old
)
print(f'  轴线方向: ({a2[0]:.6f}, {a2[1]:.6f}, {a2[2]:.6f})')
print(f'  轴线上点: ({p02[0]:.3f}, {p02[1]:.3f}, {p02[2]:.3f})')
print(f'  半径: {r2:.4f}')
print(f'  RMS残差: {np.sqrt(np.mean(res2**2)):.4f}')
print(f'  最大残差: {np.max(np.abs(res2)):.4f}')

ideal_z = np.array([0, 0, 1])
dot_z2 = np.abs(np.dot(a2, ideal_z))
angle_z2 = np.degrees(np.arccos(np.clip(dot_z2, -1, 1)))
print(f'  与Z轴夹角: {angle_z2:.2f}°')

# 检查正交性
ortho_check = np.abs(np.dot(a1, a2))
print(f'\n  两圆柱轴线正交性: dot={ortho_check:.6f} (0=正交)')

# ============================================================
# 4a. 用完整拟合参数计算交线曲线
# ============================================================
# 虽然轴线有微小偏差，但近似∥Y和∥Z，用投影法算交线
# Y圆柱在XZ平面截面圆心(X0_c1, Z0_c1)，半径r1
# Z圆柱在XY平面截面圆心(X0_c2, Y0_c2)，半径r2
X0_c1, Z0_c1 = p01[0], p01[2]
X0_c2, Y0_c2 = p02[0], p02[1]

common_min = max(X0_c1 - r1, X0_c2 - r2)
common_max = min(X0_c1 + r1, X0_c2 + r2)
print(f'公共X范围: [{common_min:.2f}, {common_max:.2f}]')

tt = np.linspace(common_min, common_max, 200)
dz = np.sqrt(np.maximum(0, r1**2 - (tt - X0_c1)**2))
z_plus = Z0_c1 + dz
z_minus = Z0_c1 - dz
dy = np.sqrt(np.maximum(0, r2**2 - (tt - X0_c2)**2))
y_plus = Y0_c2 + dy
y_minus = Y0_c2 - dy
curves = [
    (tt,       y_plus,  z_plus),
    (tt[::-1], y_minus[::-1], z_plus[::-1]),
    (tt,       y_plus,  z_minus),
    (tt[::-1], y_minus[::-1], z_minus[::-1]),
]

# ============================================================
# 4. 对比
print('\n===== 对比 =====')
print(f'Y圆柱RMS: 原始={np.sqrt(np.mean(res1_old**2)):.4f} → 完整={np.sqrt(np.mean(res1**2)):.4f}')
print(f'Z圆柱RMS: 原始={np.sqrt(np.mean(res2_old**2)):.4f} → 完整={np.sqrt(np.mean(res2**2)):.4f}')
rms_improve1 = (np.sqrt(np.mean(res1_old**2)) - np.sqrt(np.mean(res1**2))) / np.sqrt(np.mean(res1_old**2)) * 100
rms_improve2 = (np.sqrt(np.mean(res2_old**2)) - np.sqrt(np.mean(res2**2))) / np.sqrt(np.mean(res2_old**2)) * 100
print(f'Y圆柱RMS改善: {rms_improve1:.1f}%')
print(f'Z圆柱RMS改善: {rms_improve2:.1f}%')

# ============================================================
# 5. 生成圆柱面（用于可视化）
# ============================================================
def make_cylinder_full(axis_dir, axis_point, r, length=80, n_theta=60, n_t=30):
    """通用圆柱面生成"""
    a = np.array(axis_dir) / np.linalg.norm(axis_dir)
    p0 = np.array(axis_point)
    # 找一个垂直于a的向量
    if abs(a[0]) < 0.9:
        u = np.array([1, 0, 0])
    else:
        u = np.array([0, 1, 0])
    v = np.cross(a, u)
    v = v / np.linalg.norm(v)
    u = np.cross(v, a)
    u = u / np.linalg.norm(u)
    theta = np.linspace(0, 2*np.pi, n_theta)
    t_vals = np.linspace(-length/2, length/2, n_t)
    T, Th = np.meshgrid(t_vals, theta)
    X = p0[0] + a[0]*T + r*np.cos(Th)*u[0] + r*np.sin(Th)*v[0]
    Y = p0[1] + a[1]*T + r*np.cos(Th)*u[1] + r*np.sin(Th)*v[1]
    Z = p0[2] + a[2]*T + r*np.cos(Th)*u[2] + r*np.sin(Th)*v[2]
    return X, Y, Z

# 确定范围
mr = max(px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()) / 2
mx, my, mz = (px.max()+px.min())/2, (py.max()+py.min())/2, (pz.max()+pz.min())/2
Ylen = max(py) - min(py) + 60
Zlen = max(pz) - min(pz) + 60
# 分别控制圆柱长度
Y_cyl_len = max(py) - min(py) + 40
Z_cyl_len = max(pz) - min(pz) + 40

# ============================================================
# 6. 计算理论交线曲线
# ============================================================
# 用完整圆柱参数重建交线
# 需要求解两圆柱的公共曲线
# 从圆柱2(Z原方向)的方程解出参数 → 代入圆柱1
# 完整圆柱没有简单的代数解，用数值方法

# 先用原始方法算交线（近似）
common_min = max(X0_cyl1_old - r1_old, X0_cyl2_old - r2_old)
common_max = min(X0_cyl1_old + r1_old, X0_cyl2_old + r2_old)
t = np.linspace(common_min, common_max, 200)
dz = np.sqrt(np.maximum(0, r1_old**2 - (t - X0_cyl1_old)**2))
z_plus = Z0_cyl1_old + dz
z_minus = Z0_cyl1_old - dz
dy = np.sqrt(np.maximum(0, r2_old**2 - (t - X0_cyl2_old)**2))
y_plus = Y0_mean + dy
y_minus = Y0_mean - dy
curves_old = [
    (t, y_plus, z_plus), (t[::-1], y_minus[::-1], z_plus[::-1]),
    (t, y_plus, z_minus), (t[::-1], y_minus[::-1], z_minus[::-1]),
]

# ============================================================
# 7. 生成圆柱面 + 4幅目标图 + 2×2拼合
# ============================================================
X1c, Y1c, Z1c = make_cylinder_full(a1, p01, r1, length=Y_cyl_len)
X2c, Y2c, Z2c = make_cylinder_full(a2, p02, r2, length=Z_cyl_len)

# ---- 图1: 3D 带圆柱 + 实测点 ----
fig1 = plt.figure(figsize=(14, 12))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X1c, Y1c, Z1c, alpha=0.12, color='steelblue', edgecolor='none',
                 label=f'Cyl-Y (r₁={r1:.2f}mm)')
ax1.plot_surface(X2c, Y2c, Z2c, alpha=0.12, color='seagreen', edgecolor='none',
                 label=f'Cyl-Z (r₂={r2:.2f}mm)')
ax1.scatter(px, py, pz, c='red', s=20, alpha=0.8, label='Measured points (81)')
ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
ax1.set_title(f'Fitted Cylinders with Measured Points  |  RMS: Y={np.sqrt(np.mean(res1**2)):.3f}mm, Z={np.sqrt(np.mean(res2**2)):.3f}mm', fontsize=12)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_xlim(mx-mr*1.2, mx+mr*1.2); ax1.set_ylim(my-mr*1.2, my+mr*1.2); ax1.set_zlim(mz-mr*1.2, mz+mr*1.2)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(f'{out_dir}/fig1_cylinders_3d.png', dpi=150)
print('图1 saved (Cylinders 3D)')

# ---- 图2: 3D 相交曲线 + 实测点 ----
fig2 = plt.figure(figsize=(14, 12))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(px, py, pz, c='red', s=20, alpha=0.8, label='Measured points (81)')
colors = ['blue', 'cyan', 'magenta', 'orange']
for i, (xs, ys, zs) in enumerate(curves):
    ax2.plot(xs, ys, zs, color=colors[i], linewidth=2.5,
             label='Fitted intersection' if i == 0 else None)
ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
ax2.set_title('Fitted Intersection Curve vs Measured Points', fontsize=13)
ax2.legend(fontsize=9, loc='upper left')
ax2.set_xlim(mx-mr*1.2, mx+mr*1.2); ax2.set_ylim(my-mr*1.2, my+mr*1.2); ax2.set_zlim(mz-mr*1.2, mz+mr*1.2)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(f'{out_dir}/fig2_curve_3d.png', dpi=150)
print('图2 saved (Intersection Curve 3D)')

# ---- 图3: XZ投影（Y圆柱截面）----
fig3, ax3 = plt.subplots(figsize=(10, 9))
theta_plot = np.linspace(0, 2*np.pi, 200)
circle_x = X0_c1 + r1 * np.cos(theta_plot)
circle_z = Z0_c1 + r1 * np.sin(theta_plot)
ax3.plot(circle_x, circle_z, 'b-', linewidth=2, label=f'Cyl-Y section (r₁={r1:.3f}mm)')
ax3.scatter(px, pz, c='red', s=20, alpha=0.7, label='Measured points')
ax3.scatter([X0_c1], [Z0_c1], c='blue', s=60, marker='+', linewidths=2.5,
            label=f'Center ({X0_c1:.1f}, {Z0_c1:.1f})')
ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Z (mm)')
ax3.set_title('XZ Projection — Y-Cylinder Section (∥Y)', fontsize=13)
ax3.grid(True, alpha=0.3); ax3.axis('equal'); ax3.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(f'{out_dir}/fig3_xz_projection.png', dpi=150)
print('图3 saved (XZ Projection)')

# ---- 图4: XY投影（Z圆柱截面）----
fig4, ax4 = plt.subplots(figsize=(10, 9))
circle_x2 = X0_c2 + r2 * np.cos(theta_plot)
circle_y2 = Y0_c2 + r2 * np.sin(theta_plot)
ax4.plot(circle_x2, circle_y2, 'g-', linewidth=2, label=f'Cyl-Z section (r₂={r2:.3f}mm)')
ax4.scatter(px, py, c='red', s=20, alpha=0.7, label='Measured points')
ax4.scatter([X0_c2], [Y0_c2], c='green', s=60, marker='+', linewidths=2.5,
            label=f'Center ({X0_c2:.1f}, {Y0_c2:.1f})')
ax4.set_xlabel('X (mm)'); ax4.set_ylabel('Y (mm)')
ax4.set_title('XY Projection — Z-Cylinder Section (∥Z)', fontsize=13)
ax4.grid(True, alpha=0.3); ax4.axis('equal'); ax4.legend(fontsize=9)
fig4.tight_layout()
fig4.savefig(f'{out_dir}/fig4_xy_projection.png', dpi=150)
print('图4 saved (XY Projection)')

# ---- 图5: 2×2拼合图 ----
fig_collage = plt.figure(figsize=(18, 15))

# A: 3D圆柱+点
axA = fig_collage.add_subplot(2, 2, 1, projection='3d')
axA.plot_surface(X1c, Y1c, Z1c, alpha=0.12, color='steelblue', edgecolor='none')
axA.plot_surface(X2c, Y2c, Z2c, alpha=0.12, color='seagreen', edgecolor='none')
axA.scatter(px, py, pz, c='red', s=10, alpha=0.7)
axA.set_title('A: Fitted Cylinders + Measured Points', fontsize=11)
axA.set_xlim(mx-mr*1.2, mx+mr*1.2); axA.set_ylim(my-mr*1.2, my+mr*1.2); axA.set_zlim(mz-mr*1.2, mz+mr*1.2)
axA.grid(True, alpha=0.2)

# B: 交线+点
axB = fig_collage.add_subplot(2, 2, 2, projection='3d')
axB.scatter(px, py, pz, c='red', s=10, alpha=0.7)
c_colors = ['blue', 'cyan', 'magenta', 'orange']
for i, (xs, ys, zs) in enumerate(curves):
    axB.plot(xs, ys, zs, color=c_colors[i], linewidth=2.0)
axB.set_title('B: Intersection Curve + Measured Points', fontsize=11)
axB.set_xlim(mx-mr*1.2, mx+mr*1.2); axB.set_ylim(my-mr*1.2, my+mr*1.2); axB.set_zlim(mz-mr*1.2, mz+mr*1.2)
axB.grid(True, alpha=0.2)

# C: XZ投影
axC = fig_collage.add_subplot(2, 2, 3)
axC.plot(circle_x, circle_z, 'b-', linewidth=1.5, label=f'Cyl-Y r₁={r1:.2f}')
axC.scatter(px, pz, c='red', s=8, alpha=0.5)
axC.scatter([X0_c1], [Z0_c1], c='blue', s=40, marker='+', linewidths=2)
axC.set_title('C: XZ Projection (Y-Cyl Section)', fontsize=11)
axC.set_xlabel('X (mm)'); axC.set_ylabel('Z (mm)')
axC.axis('equal'); axC.legend(fontsize=8); axC.grid(True, alpha=0.3)

# D: XY投影
axD = fig_collage.add_subplot(2, 2, 4)
axD.plot(circle_x2, circle_y2, 'g-', linewidth=1.5, label=f'Cyl-Z r₂={r2:.2f}')
axD.scatter(px, py, c='red', s=8, alpha=0.5)
axD.scatter([X0_c2], [Y0_c2], c='green', s=40, marker='+', linewidths=2)
axD.set_title('D: XY Projection (Z-Cyl Section)', fontsize=11)
axD.set_xlabel('X (mm)'); axD.set_ylabel('Y (mm)')
axD.axis('equal'); axD.legend(fontsize=8); axD.grid(True, alpha=0.3)

fig_collage.suptitle('Cylinder Fit Results (Full 3D Least Squares)', fontsize=14, y=0.98)
fig_collage.tight_layout()
fig_collage.savefig(f'{out_dir}/fig_collage_full_fit.png', dpi=150)
print('图5 (collage) saved')

print('\n✅ 所有图片已保存到 code/ 目录')
