import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. 读取数据
# ============================================================
path = r'C:\Users\KCserver\AppData\Local\hermes\cache\documents\doc_f08acc09a0d8_球刀中心点及轮廓轨迹点.xlsx'
df = pd.read_excel(path)
# 接触点（工件表面上的实测点）
px, py, pz = df['x'].values, df['y'].values, df['z'].values
N = len(px)
print(f'实测点数: {N}')

# ============================================================
# 2. 圆柱拟合
# ============================================================
# 两个正交圆柱：一个∥Y，一个∥Z
# 交线Y方向对称 → Y₀ = mean of all Y
Y0 = py.mean()
print(f'Y₀ = {Y0:.4f}')

# --- Y方向圆柱 (轴∥Y) ---
# XZ平面圆拟合: (x - X₀)² + (z - Z₀)² = r₁²
# 展开: x² + z² = 2X₀·x + 2Z₀·z + (r₁² - X₀² - Z₀²)
Y_cyl1 = px**2 + pz**2
A1, B1, C1 = np.linalg.lstsq(
    np.column_stack([px, pz, np.ones_like(px)]), Y_cyl1, rcond=None
)[0]
X0_cyl1 = A1 / 2
Z0_cyl1 = B1 / 2
r1 = np.sqrt(C1 + X0_cyl1**2 + Z0_cyl1**2)
res1 = np.sqrt((px - X0_cyl1)**2 + (pz - Z0_cyl1)**2) - r1
print(f'Y方向圆柱: X₀={X0_cyl1:.4f}, Z₀={Z0_cyl1:.4f}, r₁={r1:.4f}')
print(f'  RMS残差={np.sqrt(np.mean(res1**2)):.4f}, max|r|={np.max(np.abs(res1)):.4f}')

# --- Z方向圆柱 (轴∥Z) ---
# XY平面圆拟合: (x - X₀)² + (y - Y₀)² = r₂²
# Y₀已知 → 固定圆心Y坐标的圆拟合
# 展开: x² + y² - 2yY₀ = 2X₀·x + (r₂² - X₀² - Y₀²)
Y_cyl2 = px**2 + py**2 - 2*py*Y0
A2, B2 = np.linalg.lstsq(np.column_stack([px, np.ones_like(px)]), Y_cyl2, rcond=None)[0]
X0_cyl2 = A2 / 2
r2 = np.sqrt(B2 + X0_cyl2**2 + Y0**2)
res2 = np.sqrt((px - X0_cyl2)**2 + (py - Y0)**2) - r2
print(f'Z方向圆柱: X₀={X0_cyl2:.4f}, Y₀={Y0:.4f}, r₂={r2:.4f}')
print(f'  RMS残差={np.sqrt(np.mean(res2**2)):.4f}, max|r|={np.max(np.abs(res2)):.4f}')

# ============================================================
# 3. 生成圆柱面（用于可视化）
# ============================================================
def make_cylinder(axis, cx, cy, cz, r, length=80, n_theta=60, n_t=30):
    """生成圆柱面的 X, Y, Z meshgrid"""
    idx = {'X': 0, 'Y': 1, 'Z': 2}
    ai = idx[axis]
    ri = [i for i in range(3) if i != ai]
    center = [cx, cy, cz]
    theta = np.linspace(0, 2 * np.pi, n_theta)
    t = np.linspace(center[ai] - length/2, center[ai] + length/2, n_t)
    T, Th = np.meshgrid(t, theta)
    coords = [None, None, None]
    coords[ai] = T
    coords[ri[0]] = center[ri[0]] + r * np.cos(Th)
    coords[ri[1]] = center[ri[1]] + r * np.sin(Th)
    return tuple(coords)

# Y方向圆柱: 轴∥Y, 过点(X₀, ?, Z₀) → 画长盖过图范围
Y1_len = max(py) - min(py) + 60
X1c, Y1c, Z1c = make_cylinder('Y', X0_cyl1, Y0, Z0_cyl1, r1, length=Y1_len)

# Z方向圆柱: 轴∥Z, 过点(X₀, Y₀, ?) → 画长盖过图范围
Z2_len = max(pz) - min(pz) + 60
X2c, Y2c, Z2c = make_cylinder('Z', X0_cyl2, Y0, -60, r2, length=Z2_len)

# ============================================================
# 4. 计算理论交线曲线
# ============================================================
# Y方向圆柱(轴∥Y): (x - X₀)² + (z - Z₀)² = r₁², Y自由
# Z方向圆柱(轴∥Z): (x - X₀₂)² + (y - Y₀)² = r₂², Z自由
#
# 从圆柱2(Z方向)解出 y: y = Y₀ ± sqrt(r₂² - (x - X₀₂)²)
# 代入圆柱1: (x - X₀₁)² + (z - Z₀₁)² = r₁², Y自由
# 所以参数化用 x 作为公共坐标
# 从圆柱1(Y方向)解出 z: z = Z₀₁ ± sqrt(r₁² - (x - X₀₁)²)
# 从圆柱2(Z方向)解出 y: y = Y₀ ± sqrt(r₂² - (x - X₀₂)²)

common_min = max(X0_cyl1 - r1, X0_cyl2 - r2)
common_max = min(X0_cyl1 + r1, X0_cyl2 + r2)
print(f'公共X范围: [{common_min:.2f}, {common_max:.2f}]')

t = np.linspace(common_min, common_max, 200)

# 从圆柱1(Y方向)解出 z
dz = np.sqrt(np.maximum(0, r1**2 - (t - X0_cyl1)**2))
z_plus = Z0_cyl1 + dz
z_minus = Z0_cyl1 - dz

# 从圆柱2(Z方向)解出 y
dy = np.sqrt(np.maximum(0, r2**2 - (t - X0_cyl2)**2))
y_plus = Y0 + dy
y_minus = Y0 - dy

# 四段曲线: (±z, ±y)
curves = [
    (t,       y_plus,  z_plus),    # (+, +)
    (t[::-1], y_minus[::-1], z_plus[::-1]),  # (-, +)  
    (t,       y_plus,  z_minus),   # (+, -)
    (t[::-1], y_minus[::-1], z_minus[::-1]),  # (-, -)
]

# ============================================================
# 5. 绘图
# ============================================================

# ---- 图1: 测量点 + 两个圆柱 3D ----
fig1 = plt.figure(figsize=(14, 12))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X1c, Y1c, Z1c, alpha=0.15, color='steelblue', edgecolor='none',
                 label=f'Cyl-Y (r₁={r1:.2f})')
ax1.plot_surface(X2c, Y2c, Z2c, alpha=0.15, color='seagreen', edgecolor='none',
                 label=f'Cyl-Z (r₂={r2:.2f})')
ax1.scatter(px, py, pz, c='red', s=15, alpha=0.8, label='Measured points')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('Measured Points with Fitted Cylinders', fontsize=13)
ax1.legend(fontsize=9)
# 等比例
mr = max(px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()) / 2
mx, my, mz = (px.max()+px.min())/2, (py.max()+py.min())/2, (pz.max()+pz.min())/2
ax1.set_xlim(mx-mr*1.2, mx+mr*1.2)
ax1.set_ylim(my-mr*1.2, my+mr*1.2)
ax1.set_zlim(mz-mr*1.2, mz+mr*1.2)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(r'C:\Users\KCserver\projects\formal\机器人末端力控\code\fig1_points_and_cylinders.png', dpi=150)
print('图1 saved')

# ---- 图2: 测量点 + 交线曲线 对比 ----
fig2 = plt.figure(figsize=(14, 12))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(px, py, pz, c='red', s=15, alpha=0.8, label='Measured points')
colors = ['blue', 'cyan', 'magenta', 'orange']
for i, (xs, ys, zs) in enumerate(curves):
    ax2.plot(xs, ys, zs, color=colors[i], linewidth=2.0,
             label='Fitted intersection' if i == 0 else None)
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')
ax2.set_title('Measured Points vs Fitted Intersection Curve', fontsize=13)
ax2.legend(fontsize=9)
ax2.set_xlim(mx-mr*1.2, mx+mr*1.2)
ax2.set_ylim(my-mr*1.2, my+mr*1.2)
ax2.set_zlim(mz-mr*1.2, mz+mr*1.2)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(r'C:\Users\KCserver\projects\formal\机器人末端力控\code\fig2_measured_vs_curve.png', dpi=150)
print('图2 saved')

# ---- 图3: Y投影图 (XZ平面) ----
fig3, ax3 = plt.subplots(figsize=(10, 9))
# 圆柱截面圆 (XZ平面)
theta = np.linspace(0, 2*np.pi, 200)
circle_x = X0_cyl1 + r1 * np.cos(theta)
circle_z = Z0_cyl1 + r1 * np.sin(theta)
ax3.plot(circle_x, circle_z, 'b-', linewidth=1.5, label=f'Cyl-Y section (r₁={r1:.2f})')
ax3.scatter(px, pz, c='red', s=15, alpha=0.7, label='Measured points')
ax3.scatter([X0_cyl1], [Z0_cyl1], c='blue', s=50, marker='+', linewidths=2, label=f'Center ({X0_cyl1:.1f}, {Z0_cyl1:.1f})')
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Z (mm)')
ax3.set_title('Y-Projection: XZ Plane (Cyl ∥ Y)', fontsize=13)
ax3.grid(True, alpha=0.3)
ax3.axis('equal')
ax3.legend(fontsize=9)
fig3.tight_layout()
fig3.savefig(r'C:\Users\KCserver\projects\formal\机器人末端力控\code\fig3_xz_projection.png', dpi=150)
print('图3 saved')

# ---- 图4: Z投影图 (XY平面) ----
fig4, ax4 = plt.subplots(figsize=(10, 9))
# 圆柱截面圆 (XY平面)
circle_x2 = X0_cyl2 + r2 * np.cos(theta)
circle_y2 = Y0 + r2 * np.sin(theta)
ax4.plot(circle_x2, circle_y2, 'g-', linewidth=1.5, label=f'Cyl-Z section (r₂={r2:.2f})')
ax4.scatter(px, py, c='red', s=15, alpha=0.7, label='Measured points')
ax4.scatter([X0_cyl2], [Y0], c='green', s=50, marker='+', linewidths=2, label=f'Center ({X0_cyl2:.1f}, {Y0:.1f})')
ax4.set_xlabel('X (mm)')
ax4.set_ylabel('Y (mm)')
ax4.set_title('Z-Projection: XY Plane (Cyl ∥ Z)', fontsize=13)
ax4.grid(True, alpha=0.3)
ax4.axis('equal')
ax4.legend(fontsize=9)
fig4.tight_layout()
fig4.savefig(r'C:\Users\KCserver\projects\formal\机器人末端力控\code\fig4_xy_projection.png', dpi=150)
print('图4 saved')

# ============================================================
# 6. 输出参数
# ============================================================
print('\n' + '='*60)
print('圆柱拟合结果')
print('='*60)

print(f'\nY方向圆柱 (轴∥Y):')
print(f'  轴线 X₀ = {X0_cyl1:.3f} mm')
print(f'  轴线 Z₀ = {Z0_cyl1:.3f} mm')
print(f'  半径 r₁ = {r1:.3f} mm')
print(f'  RMS残差 = {np.sqrt(np.mean(res1**2)):.4f} mm')
print(f'  最大残差:   {np.max(np.abs(res1)):.4f} mm')

print(f'\nZ方向圆柱 (轴∥Z):')
print(f'  轴线 X 坐标: X₀ = {X0_cyl2:.3f} mm')
print(f'  轴线 Y 坐标: Y₀ = {Y0:.3f} mm')
print(f'  半径:       r₂ = {r2:.3f} mm')
print(f'  RMS残差:    {np.sqrt(np.mean(res2**2)):.4f} mm')
print(f'  最大残差:   {np.max(np.abs(res2)):.4f} mm')

print(f'\n设计值对比:')
print(f'  Y圆柱设计半径 10mm → 实测 {r1:.2f}mm (偏差 {r1-10:+.2f}mm)')
print(f'  Z圆柱设计半径 20mm → 实测 {r2:.2f}mm (偏差 {r2-20:+.2f}mm)')
print(f'  Y圆柱轴线X₀设计 0mm → 实测 {X0_cyl1:.2f}mm')
print(f'  Y圆柱轴线Z₀设计 0mm → 实测 {Z0_cyl1:.2f}mm')
print(f'  Z圆柱轴线X₀设计 27mm → 实测 {X0_cyl2:.2f}mm')
print('\n所有图片已保存到 code/ 目录')
