"""验证 cylinder_geometry_v2 — 对比旧版结果"""
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---- 加载实测数据 & 拟合 ----
from cylinder_fitting_v2 import fit_cylinders_from_points

df = pd.read_excel('code/data/球刀中心点及轮廓轨迹点.xlsx')
pts = df[['x', 'y', 'z']].values

cyls, details = fit_cylinders_from_points(pts, 'Y', 'Z')
cyl_y, cyl_z = cyls

print("=== 圆柱参数 ===")
print(f"Y: r={cyl_y.radius:.3f}, p1={[f'{x:.3f}' for x in cyl_y.p1]}")
print(f"Z: r={cyl_z.radius:.3f}, p1={[f'{x:.3f}' for x in cyl_z.p1]}")

# ---- V2 交线 ----
from cylinder_geometry_v2 import sample_intersection as si_v2

geom_v2 = si_v2(cyl_y, cyl_z, n_samples=500)
print(f"\n=== V2 ===")
print(f"  pts: {geom_v2.sample_pts.shape}")
print(f"  first: [{geom_v2.sample_pts[0,0]:.3f}, {geom_v2.sample_pts[0,1]:.3f}, {geom_v2.sample_pts[0,2]:.3f}]")
closure = np.linalg.norm(geom_v2.sample_pts[0] - geom_v2.sample_pts[-1])
print(f"  closure: {closure:.4f} mm")

# ---- 旧版交线 ----
from cylinder_fitting import fit_cylinders_from_points as fit_old
from cylinder_geometry import sample_intersection as si_old

_, geom_old = fit_old(pts, 'Y', 'Z')
geom_old_s = si_old('Y', geom_old.c1, geom_old.r1,
                    'Z', geom_old.c2, geom_old.r2, n_samples=500)
print(f"\n=== 旧版 ===")
print(f"  pts: {geom_old_s.sample_pts.shape}")

# ---- 逐点对比 ----
tree_old = cKDTree(geom_old_s.sample_pts)
dists, _ = tree_old.query(geom_v2.sample_pts)
print(f"\n=== 对比 ===")
print(f"  V2->旧 max:  {dists.max():.4f} mm")
print(f"  V2->旧 mean: {dists.mean():.4f} mm")

# 长度对比
def curve_len(pts):
    d = np.diff(pts, axis=0)
    return np.sum(np.linalg.norm(d, axis=1)) + np.linalg.norm(pts[0] - pts[-1])

print(f"  总长 V2: {curve_len(geom_v2.sample_pts):.2f} mm")
print(f"  总长 旧: {curve_len(geom_old_s.sample_pts):.2f} mm")

if dists.max() < 0.1:
    print("\n✅ 完全一致")
else:
    print(f"\n⚠️ 偏差: {dists.max():.4f} mm")
