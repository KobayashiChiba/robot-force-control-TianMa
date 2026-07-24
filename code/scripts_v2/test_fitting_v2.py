"""快速验证 cylinder_fitting_v2.py"""
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
from cylinder_fitting_v2 import fit_cylinders_from_points, make_geom
from cylinder_geometry import resample_curve

# 用理论数据验证
np.random.seed(42)
# 模拟 Y 圆柱 (r=10) 和 Z 圆柱 (r=20) 交线上的散点
# 交线大致范围在 x≈50~70, y≈55~75, z≈-50~-30
n_pts = 100
t_common = np.linspace(55, 75, n_pts)  # Y 范围即公共坐标
# Y圆柱: 轴心 (51.5, *, -39.7) r=9.0  → Y轴，轴心XZ平面 (51.5, -39.7)
c1_x, c1_z = 51.5, -39.7
r1 = 9.0
# 随机选取正负分支
s1 = np.random.choice([1, -1], n_pts)
x_vals = c1_x + s1 * np.sqrt(np.maximum(0, r1**2 - (t_common - 65)**2))
z_vals = c1_z + np.random.randn(n_pts) * 0.05  # 加噪声 (Z不明显)

pts = np.column_stack([x_vals, t_common, z_vals])

print("=== 拟合测试 ===")
cyls, details = fit_cylinders_from_points(pts, 'Y', 'Z')

for i, (cyl, d) in enumerate(zip(cyls, details)):
    print(f"\n圆柱 {i+1}:")
    print(f"  nearest_axis: {cyl.nearest_axis}")
    print(f"  p1:           [{cyl.p1[0]:.3f}, {cyl.p1[1]:.3f}, {cyl.p1[2]:.3f}]")
    print(f"  p2:           [{cyl.p2[0]:.3f}, {cyl.p2[1]:.3f}, {cyl.p2[2]:.3f}]")
    print(f"  direction:    [{cyl.direction[0]:.4f}, {cyl.direction[1]:.4f}, {cyl.direction[2]:.4f}]")
    print(f"  radius:       {cyl.radius:.3f}  (expected: {r1 if i==0 else 0.001})")
    print(f"  rms:          {d['rms']:.4f} mm")
    print(f"  max_err:      {d['max_err']:.4f} mm")

# 测试 make_geom 兼容
print("\n=== Geom 兼容测试 ===")
geom = make_geom(cyls)
print(f"  axis1={geom.axis1}, r1={geom.r1}, c1={geom.c1}")
print(f"  axis2={geom.axis2}, r2={geom.r2}, c2={geom.c2}")

# 测试 resample_curve
geom_r = resample_curve(geom, n_samples=50)
print(f"\n=== resample_curve ===")
print(f"  n_samples: {geom_r.n_samples}")
print(f"  sample_pts shape: {geom_r.sample_pts.shape}")
print(f"  first pt: {geom_r.sample_pts[0]}")
print(f"  last pt:  {geom_r.sample_pts[-1]}")

print("\n✅ 全部通过")
