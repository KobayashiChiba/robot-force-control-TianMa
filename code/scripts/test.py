import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ★ 可配置参数 — 修改这里即可自由调整几何体 ★
# ============================================================

# --- 圆柱1 ---
AXIS1 = 'Y'               # 轴线方向: 'X', 'Y', 'Z'
CX1, CY1, CZ1 = 0, 0, 0   # 轴线上一点
R1 = 10                    # 半径
LEN1 = 100                 # 显示长度

# --- 圆柱2 ---
AXIS2 = 'Z'               # 轴线方向: 'X', 'Y', 'Z'
CX2, CY2, CZ2 = 27, 0, 0  # 轴线上一点
R2 = 20                    # 半径
LEN2 = 100                 # 显示长度

# --- 显示 ---
CYL_ALPHA = 0.1           # 圆柱面透明度
AXIS_RANGE_X = (-2, 18)
AXIS_RANGE_Y = (-10, 10)
AXIS_RANGE_Z = (-10, 10)
NORMAL_SAMPLE = 15        # 每隔多少个采样点画一个法向量
NORMAL_LENGTH = 2.5       # 法向量箭头长度
OUTPUT_FILE = 'intersection_curve.png'

# ============================================================
# 工具函数
# ============================================================
IDX = {'X': 0, 'Y': 1, 'Z': 2}

def make_cylinder(axis, cx, cy, cz, r, length, n_theta=80, n_t=40):
    """生成圆柱面的 X, Y, Z meshgrid。axis: 'X'|'Y'|'Z'"""
    ai = IDX[axis]                        # 轴向索引
    ri = [i for i in range(3) if i != ai]  # 两个径向索引
    center = [cx, cy, cz]

    theta = np.linspace(0, 2 * np.pi, n_theta)
    t = np.linspace(center[ai] - length/2, center[ai] + length/2, n_t)
    T, Th = np.meshgrid(t, theta)

    coords = [None, None, None]
    coords[ai] = T
    coords[ri[0]] = center[ri[0]] + r * np.cos(Th)
    coords[ri[1]] = center[ri[1]] + r * np.sin(Th)
    return tuple(coords)  # X, Y, Z


def axis_line(axis, cx, cy, cz, length):
    """返回轴线两端点坐标: (x1,x2), (y1,y2), (z1,z2)"""
    ai = IDX[axis]
    center = [cx, cy, cz]
    p1 = list(center)
    p2 = list(center)
    p1[ai] -= length / 2
    p2[ai] += length / 2
    return (p1[0], p2[0]), (p1[1], p2[1]), (p1[2], p2[2])


# ============================================================
# 生成圆柱面
# ============================================================
X1, Y1, Z1 = make_cylinder(AXIS1, CX1, CY1, CZ1, R1, LEN1)
X2, Y2, Z2 = make_cylinder(AXIS2, CX2, CY2, CZ2, R2, LEN2)

# ============================================================
# 求解交线（通用版，适用于任意两个不同轴向的圆柱）
# ============================================================
if AXIS1 == AXIS2:
    raise ValueError("AXIS1 and AXIS2 must be different (parallel cylinders not supported)")

# 确定三个坐标的角色
ai1, ai2 = IDX[AXIS1], IDX[AXIS2]            # 各自的轴向索引
common = ({0, 1, 2} - {ai1, ai2}).pop()      # 公共径向坐标（两圆柱共用的径向）
# 对于圆柱1: 径向 = {common, ai2}  → 从圆柱1可解出 ai2
# 对于圆柱2: 径向 = {common, ai1}  → 从圆柱2可解出 ai1
other1 = ai2  # 圆柱1的另一径向 = 圆柱2的轴向
other2 = ai1  # 圆柱2的另一径向 = 圆柱1的轴向

c1 = [CX1, CY1, CZ1]
c2 = [CX2, CY2, CZ2]

# 公共坐标的取值范围
t_min = max(c1[common] - R1, c2[common] - R2)
t_max = min(c1[common] + R1, c2[common] + R2)

if t_min >= t_max:
    raise ValueError(f"No intersection! common-coord range empty: [{t_min:.2f}, {t_max:.2f}]")

coord_names = ['X', 'Y', 'Z']
print(f"Common coordinate: {coord_names[common]}, range: [{t_min:.2f}, {t_max:.2f}]")

N = 250
t_up   = np.linspace(t_min, t_max, N)
t_down = np.linspace(t_max, t_min, N)

# 从圆柱1解出 other1 (取正负号)
p1_up   = c1[other1] + np.sqrt(R1**2 - (t_up   - c1[common])**2)
p1_down = c1[other1] + np.sqrt(R1**2 - (t_down - c1[common])**2)
n1_up   = c1[other1] - np.sqrt(R1**2 - (t_up   - c1[common])**2)
n1_down = c1[other1] - np.sqrt(R1**2 - (t_down - c1[common])**2)

# 从圆柱2解出 other2 (取正负号)
p2_up   = c2[other2] + np.sqrt(R2**2 - (t_up   - c2[common])**2)
p2_down = c2[other2] + np.sqrt(R2**2 - (t_down - c2[common])**2)
n2_up   = c2[other2] - np.sqrt(R2**2 - (t_up   - c2[common])**2)
n2_down = c2[other2] - np.sqrt(R2**2 - (t_down - c2[common])**2)

# 四段曲线独立绘制，避免拼接顺序错误导致多余连线
#   段: (other1符号, other2符号, t方向)
branches = [
    (p1_up,   p2_up,   t_up),     # (+, +, ↑)
    (p1_down, n2_down, t_down),   # (+, -, ↓)
    (n1_up,   n2_up,   t_up),     # (-, -, ↑)
    (n1_down, p2_down, t_down),   # (-, +, ↓)
]
curves = []
for v1, v2, t_vals in branches:
    pt = [None, None, None]
    pt[common] = t_vals
    pt[other1] = v1
    pt[other2] = v2
    curves.append((np.asarray(pt[0]), np.asarray(pt[1]), np.asarray(pt[2])))

# ============================================================
# 计算打磨法向量
# ============================================================
# 圆柱内部 = 空气区域，圆柱外部 = 金属
# 相交曲线上每点有两个圆柱面法向量 n1, n2（均指向金属/外侧）
# 打磨法向量 = (n1 + n2) / |n1 + n2|，即二者角平分线方向，指向金属

axis_dir = {'X': np.array([1., 0., 0.]),
            'Y': np.array([0., 1., 0.]),
            'Z': np.array([0., 0., 1.])}
d1 = axis_dir[AXIS1]
d2 = axis_dir[AXIS2]
C1 = np.array([CX1, CY1, CZ1])
C2 = np.array([CX2, CY2, CZ2])

normals_p, normals_v = [], []  # 起点和方向
for xs, ys, zs in curves:
    for i in range(0, len(xs), NORMAL_SAMPLE):
        P = np.array([xs[i], ys[i], zs[i]])
        # 圆柱1 径向向外法向量
        r1 = P - C1 - np.dot(P - C1, d1) * d1
        n1 = r1 / np.linalg.norm(r1)
        # 圆柱2 径向向外法向量
        r2 = P - C2 - np.dot(P - C2, d2) * d2
        n2 = r2 / np.linalg.norm(r2)
        # 打磨法向量（角平分线，指向金属）
        n_grind = n1 + n2
        n_grind /= np.linalg.norm(n_grind)
        normals_p.append(P)
        normals_v.append(n_grind * NORMAL_LENGTH)

normals_p = np.array(normals_p)
normals_v = np.array(normals_v)

# ============================================================
# 绘图
# ============================================================
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 圆柱面
ax.plot_surface(X1, Y1, Z1, alpha=CYL_ALPHA, color='steelblue', edgecolor='none',
                label=f'Cyl-{AXIS1} (R1={R1})')
ax.plot_surface(X2, Y2, Z2, alpha=CYL_ALPHA, color='seagreen', edgecolor='none',
                label=f'Cyl-{AXIS2} (R2={R2})')

# 相交曲线：四段各自独立绘制，自然闭合处自会相连
for i, (xs, ys, zs) in enumerate(curves):
    ax.plot(xs, ys, zs, color='darkred', linewidth=2.5,
            label='Intersection curve' if i == 0 else None)

# 打磨法向量箭头
ax.quiver(normals_p[:, 0], normals_p[:, 1], normals_p[:, 2],
          normals_v[:, 0], normals_v[:, 1], normals_v[:, 2],
          color='darkorange', alpha=0.85, linewidth=1.2,
          label=f'Grinding normal (every {NORMAL_SAMPLE} pts)')

# 轴线
ax1_line = axis_line(AXIS1, CX1, CY1, CZ1, LEN1)
ax2_line = axis_line(AXIS2, CX2, CY2, CZ2, LEN2)
ax.plot(*ax1_line, color='navy', linewidth=1.5, linestyle='--', alpha=0.5,
        label=f'Cyl-{AXIS1} axis')
ax.plot(*ax2_line, color='darkgreen', linewidth=1.5, linestyle='--', alpha=0.5,
        label=f'Cyl-{AXIS2} axis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Two Orthogonal Cylinders and Their Intersection\n'
             f'Cyl-{AXIS1}: R={R1}, center=({CX1},{CY1},{CZ1})  |  '
             f'Cyl-{AXIS2}: R={R2}, center=({CX2},{CY2},{CZ2})',
             fontsize=12)

ax.set_xlim(*AXIS_RANGE_X)
ax.set_ylim(*AXIS_RANGE_Y)
ax.set_zlim(*AXIS_RANGE_Z)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')  # 先保存再显示
print(f"Figure saved to {OUTPUT_FILE}")
# plt.show()  # 关闭交互窗口，避免headless卡住
