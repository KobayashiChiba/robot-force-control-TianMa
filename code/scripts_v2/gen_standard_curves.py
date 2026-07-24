"""生成标准曲线文件 — V2 版本

输出 standard_curves_v2.pkl，含:
    - ball_center_geom : GeomV2 — 球刀中心标准交线
    - contact_geom     : GeomV2 — 标准接触交线
    - ball_radius      : float  — 球刀半径 4mm
    - cyl_ball_y, cyl_ball_z : CylinderDef — 球刀中心拟合圆柱
    - cyl_contact_y, cyl_contact_z : CylinderDef — 接触曲线拟合圆柱
"""
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
import pandas as pd
import pickle

from cylinder_fitting_v2 import fit_cylinders_from_points
from cylinder_geometry_v2 import sample_intersection

BALL_RADIUS = 4.0  # mm
N_SAMPLES   = 500

# ============================================================
# 1. 球刀中心标准曲线（Z 修正 +4.815mm 后拟合）
# ============================================================
df_ball = pd.read_excel('code/data/球刀中心点_修正后.xlsx')
ball_pts = df_ball[['X_shifted', 'Y_shifted', 'Z_shifted']].values
print(f"球刀中心点: {len(ball_pts)} 个")

cyls_ball, details_ball = fit_cylinders_from_points(ball_pts, 'Y', 'Z')
cyl_ball_y, cyl_ball_z = cyls_ball
print(f"  球刀中心 Y圆柱: r={cyl_ball_y.radius:.3f}  RMS={details_ball[0]['rms']:.4f}")
print(f"  球刀中心 Z圆柱: r={cyl_ball_z.radius:.3f}  RMS={details_ball[1]['rms']:.4f}")

ball_center_geom = sample_intersection(cyl_ball_y, cyl_ball_z, n_samples=N_SAMPLES)
print(f"  交线: {ball_center_geom.n_samples} 点, 闭合差={np.linalg.norm(ball_center_geom.sample_pts[0] - ball_center_geom.sample_pts[-1]):.4f}mm")

# ============================================================
# 2. 标准接触曲线
# ============================================================
df_contact = pd.read_excel('code/data/球刀中心点及轮廓轨迹点.xlsx')
contact_pts = df_contact[['x', 'y', 'z']].values
print(f"\n接触点: {len(contact_pts)} 个")

cyls_contact, details_contact = fit_cylinders_from_points(contact_pts, 'Y', 'Z')
cyl_contact_y, cyl_contact_z = cyls_contact
print(f"  接触 Y圆柱: r={cyl_contact_y.radius:.3f}  RMS={details_contact[0]['rms']:.4f}")
print(f"  接触 Z圆柱: r={cyl_contact_z.radius:.3f}  RMS={details_contact[1]['rms']:.4f}")

contact_geom = sample_intersection(cyl_contact_y, cyl_contact_z, n_samples=N_SAMPLES)
print(f"  交线: {contact_geom.n_samples} 点, 闭合差={np.linalg.norm(contact_geom.sample_pts[0] - contact_geom.sample_pts[-1]):.4f}mm")

# ============================================================
# 3. 保存
# ============================================================
output = {
    'ball_center_geom': ball_center_geom,
    'contact_geom':     contact_geom,
    'ball_radius':      BALL_RADIUS,
    'cyl_ball_y':       cyl_ball_y,
    'cyl_ball_z':       cyl_ball_z,
    'cyl_contact_y':    cyl_contact_y,
    'cyl_contact_z':    cyl_contact_z,
    'fit_details_ball':    details_ball,
    'fit_details_contact': details_contact,
}

out_path = 'code/data/standard_curves_v2.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(output, f)

print(f"\n✅ 已保存: {out_path}")
print(f"   ball_radius = {BALL_RADIUS} mm")
print(f"   ball_center: Y r={cyl_ball_y.radius:.1f}, Z r={cyl_ball_z.radius:.1f}")
print(f"   contact:     Y r={cyl_contact_y.radius:.1f}, Z r={cyl_contact_z.radius:.1f}")
