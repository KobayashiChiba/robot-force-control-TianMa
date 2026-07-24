"""用实测数据验证 cylinder_fitting_v2.py"""
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
import pandas as pd
from cylinder_fitting_v2 import fit_cylinders_from_points, make_geom
from cylinder_geometry import resample_curve

# 读取实测数据
df = pd.read_excel('code/data/球刀中心点及轮廓轨迹点.xlsx')
print("列名:", list(df.columns))

# 接触点: x, y, z (小写); 球刀中心: X, Y, Z (大写)
pts = df[['x', 'y', 'z']].values
print(f"\n接触点数量: {len(pts)}")

# 拟合
cyls, details = fit_cylinders_from_points(pts, 'Y', 'Z')

for i, (cyl, d) in enumerate(zip(cyls, details)):
    print(f"\n圆柱 {i+1} ({cyl.nearest_axis}方向):")
    print(f"  p1:       [{cyl.p1[0]:.3f}, {cyl.p1[1]:.3f}, {cyl.p1[2]:.3f}]")
    print(f"  p2:       [{cyl.p2[0]:.3f}, {cyl.p2[1]:.3f}, {cyl.p2[2]:.3f}]")
    print(f"  spacing:  {np.linalg.norm(cyl.p2 - cyl.p1):.1f} mm")
    print(f"  radius:   {cyl.radius:.3f} mm")
    print(f"  rms:      {d['rms']:.4f} mm")
    print(f"  max_err:  {d['max_err']:.4f} mm")

# 对照旧版
from cylinder_fitting import fit_cylinders_from_points as fit_old
params_old, geom_old = fit_old(pts, 'Y', 'Z')
print("\n=== 旧版对照 ===")
for i, p in enumerate(params_old):
    print(f"  圆柱{i+1}: axis={p.axis}, axis_point={[f'{x:.3f}' for x in p.axis_point]}, radius={p.radius:.3f}")

# 验证半径一致
print("\n=== 半径对比 ===")
for i, (cyl, p) in enumerate(zip(cyls, params_old)):
    print(f"  圆柱{i+1}: v2={cyl.radius:.4f} vs 旧={p.radius:.4f}  diff={abs(cyl.radius-p.radius):.6f}")

print("\n✅ 验证完成")
