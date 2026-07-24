"""L0+L1 联调：拟合 → 交线 → 3D可视化"""
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cylinder_fitting_v2 import fit_cylinders_from_points
from cylinder_geometry_v2 import sample_intersection

# ============================================================
# 1. 加载实测数据
# ============================================================
df = pd.read_excel('code/data/球刀中心点及轮廓轨迹点.xlsx')
contact_pts = df[['x', 'y', 'z']].values
ball_pts = df[['X', 'Y', 'Z']].values

print(f"接触点: {len(contact_pts)} 个")
print(f"球刀中心点: {len(ball_pts)} 个")

# ============================================================
# 2. L0: 拟合圆柱
# ============================================================
cyls, details = fit_cylinders_from_points(contact_pts, 'Y', 'Z')
cyl_y, cyl_z = cyls

print(f"\n=== 拟合结果 ===")
print(f"Y圆柱: r={cyl_y.radius:.4f}mm  RMS={details[0]['rms']:.4f}mm  "
      f"p1=[{cyl_y.p1[0]:.2f},{cyl_y.p1[1]:.2f},{cyl_y.p1[2]:.2f}]  "
      f"p2=[{cyl_y.p2[0]:.2f},{cyl_y.p2[1]:.2f},{cyl_y.p2[2]:.2f}]")
print(f"Z圆柱: r={cyl_z.radius:.4f}mm  RMS={details[1]['rms']:.4f}mm  "
      f"p1=[{cyl_z.p1[0]:.2f},{cyl_z.p1[1]:.2f},{cyl_z.p1[2]:.2f}]  "
      f"p2=[{cyl_z.p2[0]:.2f},{cyl_z.p2[1]:.2f},{cyl_z.p2[2]:.2f}]")

# ============================================================
# 3. L1: 计算交线
# ============================================================
geom = sample_intersection(cyl_y, cyl_z, n_samples=500)
curve = geom.sample_pts
print(f"\n交线采样: {len(curve)} 点")

# ============================================================
# 4. 可视化
# ============================================================
fig = plt.figure(figsize=(16, 6))

# --- 子图1: 3D 全景 ---
ax1 = fig.add_subplot(131, projection='3d')
# 交线
ax1.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', linewidth=1.5, label='交线 (V2)')
# 接触点
ax1.scatter(contact_pts[:, 0], contact_pts[:, 1], contact_pts[:, 2],
            c='red', s=15, alpha=0.7, label='实测接触点')
# Y圆柱轴线
ax1.plot([cyl_y.p1[0], cyl_y.p2[0]], [cyl_y.p1[1], cyl_y.p2[1]], [cyl_y.p1[2], cyl_y.p2[2]],
         'g-', linewidth=2, label=f'Y轴 (r={cyl_y.radius:.1f})')
# Z圆柱轴线
ax1.plot([cyl_z.p1[0], cyl_z.p2[0]], [cyl_z.p1[1], cyl_z.p2[1]], [cyl_z.p1[2], cyl_z.p2[2]],
         'm-', linewidth=2, label=f'Z轴 (r={cyl_z.radius:.1f})')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('3D全景')
ax1.legend(fontsize=7)

# --- 子图2: XZ投影 (Y圆柱截面) ---
ax2 = fig.add_subplot(132)
# 交线投影
ax2.plot(curve[:, 0], curve[:, 2], 'b-', linewidth=1.5, label='交线投影')
# 接触点
ax2.scatter(contact_pts[:, 0], contact_pts[:, 2], c='red', s=15, alpha=0.7, label='实测')
# Y圆柱轴线投影 (在XZ平面是点)
ax2.plot(cyl_y.p1[0], cyl_y.p1[2], 'go', markersize=8, label=f'Y轴心')
# 圆
theta = np.linspace(0, 2*np.pi, 200)
ax2.plot(cyl_y.p1[0] + cyl_y.radius*np.cos(theta),
         cyl_y.p1[2] + cyl_y.radius*np.sin(theta),
         'g--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Z (mm)')
ax2.set_title('XZ投影 (Y圆柱截面)')
ax2.axis('equal')
ax2.legend(fontsize=7)

# --- 子图3: XY投影 (Z圆柱截面) ---
ax3 = fig.add_subplot(133)
ax3.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=1.5, label='交线投影')
ax3.scatter(contact_pts[:, 0], contact_pts[:, 1], c='red', s=15, alpha=0.7, label='实测')
ax3.plot(cyl_z.p1[0], cyl_z.p1[1], 'mo', markersize=8, label=f'Z轴心')
theta = np.linspace(0, 2*np.pi, 200)
ax3.plot(cyl_z.p1[0] + cyl_z.radius*np.cos(theta),
         cyl_z.p1[1] + cyl_z.radius*np.sin(theta),
         'm--', linewidth=0.8, alpha=0.5)
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_title('XY投影 (Z圆柱截面)')
ax3.axis('equal')
ax3.legend(fontsize=7)

plt.tight_layout()
output_path = 'code/lib_v2/output_l0_l1_pipeline.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {output_path}")

print("\n✅ L0+L1 联调完成")
