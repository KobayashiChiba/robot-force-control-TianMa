"""验证 sphere_contact.py _inside_cyl 修复"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import pickle
import numpy as np

# ── 加载标准曲线 ──────────────────────
with open('../data/standard_curves_v3.pkl', 'rb') as f:
    data = pickle.load(f)

cyl_z = data['cyl_contact_z']   # CylinderDef
cyl_y = data['cyl_contact_y']
contact_pts = data['contact_geom'].pts  # (500,3)

print(f"Z圆柱: r={cyl_z.radius}, axis_dir={cyl_z.p2 - cyl_z.p1}")
print(f"Y圆柱: r={cyl_y.radius}, axis_dir={cyl_y.p2 - cyl_y.p1}")

# ── 算接触标架 + 球刀中心 ─────────────
from contact_frame_v2 import compute_frame

mid = len(contact_pts) // 2
pc = contact_pts[mid]
frame = compute_frame(pc, cyl_y, cyl_z, cyl_y.radius, cyl_z.radius)
ball_center = pc + frame.normal * 4.0

print(f"\n接触点(中间): {np.round(pc, 3)}")
print(f"法向量: {np.round(frame.normal, 4)}")
print(f"球刀中心(approx): {np.round(ball_center, 3)}")

# ── 算力 ──────────────────────────────
from sphere_contact import sphere_contact_force, _inside_cyl

F, area = sphere_contact_force(ball_center, cyl_z, cyl_y)
print(f"\n力: {np.round(F, 3)}, |F|={np.linalg.norm(F):.2f}N, area={area:.2f}mm²")

# ── 验证接触曲线点确实在圆柱面上 ──────
def radial_dist(pts, cyl):
    axis = cyl.p2 - cyl.p1
    L = np.linalg.norm(axis)
    d = axis / L
    v = pts - cyl.p1
    t = np.clip(np.dot(v, d), 0, L)
    ax_pts = cyl.p1 + t[:, None] * d
    return np.linalg.norm(pts - ax_pts, axis=1)

r_z = radial_dist(contact_pts, cyl_z)
r_y = radial_dist(contact_pts, cyl_y)

err_z = np.abs(r_z - cyl_z.radius)
err_y = np.abs(r_y - cyl_y.radius)

print(f"\n接触曲线 500点 径向距离误差:")
print(f"  Z: mean={np.mean(err_z):.6f} max={np.max(err_z):.6f}")
print(f"  Y: mean={np.mean(err_y):.6f} max={np.max(err_y):.6f}")

if np.max(err_z) < 0.1 and np.max(err_y) < 0.1:
    print("✅ 标准圆柱验证通过")
else:
    print("❌ 偏差过大!")
