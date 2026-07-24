"""L0+L1+L2 全链路验证"""
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cylinder_fitting_v2 import fit_cylinders_from_points
from cylinder_geometry_v2 import sample_intersection
from force_mechanics_v2 import compute_point_basis, decompose_force


def _set_equal_3d_aspect(ax, pts):
    """强制 3D 轴等比例尺，防止曲线变形。"""
    x_mid = (pts[:, 0].min() + pts[:, 0].max()) / 2
    y_mid = (pts[:, 1].min() + pts[:, 1].max()) / 2
    z_mid = (pts[:, 2].min() + pts[:, 2].max()) / 2
    half_range = max(
        np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])
    ) / 2 * 1.1
    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)

# ============================================================
# L0: 拟合
# ============================================================
df = pd.read_excel('code/data/球刀中心点及轮廓轨迹点.xlsx')
contact_pts = df[['x', 'y', 'z']].values

cyls, details = fit_cylinders_from_points(contact_pts, 'Y', 'Z')
cyl_y, cyl_z = cyls
print(f"L0: Y r={cyl_y.radius:.3f} RMS={details[0]['rms']:.4f}  Z r={cyl_z.radius:.3f} RMS={details[1]['rms']:.4f}")

# ============================================================
# L1: 交线
# ============================================================
geom = sample_intersection(cyl_y, cyl_z, n_samples=500)
curve = geom.sample_pts
print(f"L1: {len(curve)} pts, 闭合 {np.linalg.norm(curve[0]-curve[-1]):.4f}mm")

# ============================================================
# L2: 力分解 — 在每个采样点计算基底 + 分解虚构力
# ============================================================
n_test = 50
step = len(curve) // n_test
indices = range(0, len(curve), step)

F_test = np.array([5.0, -1.0, -8.0])  # 虚构外力

tangents, normals, verticals = [], [], []
coeffs_t, coeffs_n, coeffs_v = [], [], []
errors = []

for i in indices:
    P = curve[i]
    basis = compute_point_basis(P, geom)
    decomp = decompose_force(F_test, basis)

    tangents.append(basis.tangent)
    normals.append(basis.normal)
    verticals.append(basis.vertical)
    coeffs_t.append(decomp.coeffs[0])
    coeffs_n.append(decomp.coeffs[1])
    coeffs_v.append(decomp.coeffs[2])
    errors.append(decomp.error)

tangents = np.array(tangents)
normals = np.array(normals)
verticals = np.array(verticals)

print(f"\nL2: {n_test} 采样点力分解")
print(f"  分解误差 max: {max(errors):.2e}")
print(f"  coeffs_t: [{np.mean(coeffs_t):.2f} ± {np.std(coeffs_t):.2f}]")
print(f"  coeffs_n: [{np.mean(coeffs_n):.2f} ± {np.std(coeffs_n):.2f}]")
print(f"  coeffs_v: [{np.mean(coeffs_v):.2f} ± {np.std(coeffs_v):.2f}]")

# 检查 t·n 和 t·rz（应为 0 当轴对齐）
t_dot_n = [abs(np.dot(tangents[j], normals[j])) for j in range(n_test)]
t_dot_rz = [abs(np.dot(tangents[j], verticals[j])) for j in range(n_test)]
print(f"  |t·n|  max: {max(t_dot_n):.2e}")
print(f"  |t·rz| max: {max(t_dot_rz):.2e}")

# ============================================================
# 可视化：交线上的法向量箭头
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

# 子图1: 全景
ax = axes[0]
ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'gray', linewidth=0.8, alpha=0.5, label='curve')
# 每隔几个点画标架
show_every = 5
for j in range(0, n_test, show_every):
    i = indices[j]
    P = curve[i]
    t, n, v = tangents[j], normals[j], verticals[j]
    scale = 3.0
    ax.quiver(*P, *(t*scale), color='r', linewidth=0.5)
    ax.quiver(*P, *(n*scale), color='g', linewidth=0.5)
    ax.quiver(*P, *(v*scale), color='b', linewidth=0.5)
# 图例
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0], color='r', label='t (切向)'),
    Line2D([0],[0], color='g', label='n (法向)'),
    Line2D([0],[0], color='b', label='rz (Z径向)'),
])
ax.set_title('Local frames along intersection curve')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
_set_equal_3d_aspect(ax, curve)

# 子图2: 力分解（挑一个点展示）
idx_show = len(indices) // 2
i = indices[idx_show]
P = curve[i]
basis = compute_point_basis(P, geom)
decomp = decompose_force(F_test, basis)

ax2 = axes[1]
ax2.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'gray', linewidth=0.5, alpha=0.3)
ax2.scatter(*P, c='k', s=30)

s = 8.0
ax2.quiver(*P, *(decomp.Ft_vec*s), color='r', linewidth=2, label=f'Ft ({decomp.coeffs[0]:.1f})')
ax2.quiver(*P, *(decomp.Fn_vec*s), color='g', linewidth=2, label=f'Fn ({decomp.coeffs[1]:.1f})')
ax2.quiver(*P, *(decomp.Fv_vec*s), color='b', linewidth=2, label=f'Fv ({decomp.coeffs[2]:.1f})')
ax2.quiver(*P, *(F_test*s), color='k', linewidth=3, label=f'F [{F_test[0]},{F_test[1]},{F_test[2]}]')
ax2.legend(fontsize=7)
ax2.set_title(f'Force decomposition at P=[{P[0]:.1f},{P[1]:.1f},{P[2]:.1f}]')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
_set_equal_3d_aspect(ax2, curve)

plt.tight_layout()
out = 'code/lib_v2/output_l0_l1_l2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图: {out}")
print("✅ L0+L1+L2 全链路通过")
