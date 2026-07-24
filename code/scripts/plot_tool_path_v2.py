"""工具中心路径双视角可视化
- 左：原视角（3D自由视角）
- 右：X轴正面视角
- Z轴箭头加长（XY的3倍），标注含义
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
path = r"C:\Users\KCserver\AppData\Local\hermes\cache\documents\doc_b0c789e481b3_球刀中心点及轮廓轨迹点.xlsx"
df = pd.read_excel(path)

# 球心轨迹
cx, cy, cz = df['X'].values, df['Y'].values, df['Z'].values
# 接触点
px, py, pz = df['x'].values, df['y'].values, df['z'].values
# 四元数
q1, q2, q3, q4 = df['Q1'].values, df['Q2'].values, df['Q3'].values, df['Q4'].values

def quat_to_rot(q):
    """四元数 → 旋转矩阵"""
    q1, q2, q3, q4 = q
    R = np.array([
        [1-2*(q3**2+q4**2), 2*(q2*q3-q1*q4), 2*(q2*q4+q1*q3)],
        [2*(q2*q3+q1*q4), 1-2*(q2**2+q4**2), 2*(q3*q4-q1*q2)],
        [2*(q2*q4-q1*q3), 2*(q3*q4+q1*q2), 1-2*(q2**2+q3**2)]
    ])
    return R

# ---- 创建两个子图 ----
fig = plt.figure(figsize=(22, 10))

# 左图：原视角（参考原图角度）
ax1 = fig.add_subplot(121, projection='3d')
ax1.view_init(elev=20, azim=-60)

# 右图：从X轴方向看
ax2 = fig.add_subplot(122, projection='3d')
ax2.view_init(elev=0, azim=0)   # 面向X轴

for ax, title in [(ax1, 'Original View'), (ax2, 'Front View (viewing along X-axis)')]:

    # ---- 球心轨迹（彩色渐变） ----
    colors = plt.cm.jet(np.linspace(0, 1, len(cx)))
    for i in range(len(cx)-1):
        ax.plot(cx[i:i+2], cy[i:i+2], cz[i:i+2], color=colors[i], linewidth=2.0)

    # ---- 起始/结束标记 ----
    ax.scatter(*[cx[0], cy[0], cz[0]], color='lime', s=80, marker='o', zorder=5, label='Start')
    ax.scatter(*[cx[-1], cy[-1], cz[-1]], color='red', s=80, marker='*', zorder=5, label='End')

    # ---- 姿态坐标架（每8个点画一个） ----
    step = 8
    arrow_short = 2.5      # XY 箭头长度
    arrow_long  = 7.5      # Z 轴箭头长度（3倍）
    for i in range(0, len(cx), step):
        R = quat_to_rot([q1[i], q2[i], q3[i], q4[i]])
        origin = np.array([cx[i], cy[i], cz[i]])
        # 红 = X, 绿 = Y（短箭头）
        ax.quiver(*origin, *R[:, 0]*arrow_short, color='red', alpha=0.7, linewidth=1.2)
        ax.quiver(*origin, *R[:, 1]*arrow_short, color='green', alpha=0.7, linewidth=1.2)
        # 蓝 = Z（长箭头，3倍）
        ax.quiver(*origin, *R[:, 2]*arrow_long, color='blue', alpha=0.8, linewidth=1.8)

    # ---- 接触点轨迹（虚线） ----
    ax.plot(px, py, pz, color='gray', linewidth=1.5, linestyle='--', alpha=0.6, label='Contact path')

    # ---- 球心→接触点连线 ----
    for i in range(0, len(cx), step):
        ax.plot([cx[i], px[i]], [cy[i], py[i]], [cz[i], pz[i]],
                color='orange', alpha=0.25, linewidth=0.8)

    # ---- 轴标签 ----
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # ---- 轴等比例 ----
    max_range = max(cx.max()-cx.min(), cy.max()-cy.min(), cz.max()-cz.min()) / 2
    mid_x, mid_y, mid_z = (cx.max()+cx.min())/2, (cy.max()+cy.min())/2, (cz.max()+cz.min())/2
    ax.set_xlim(mid_x - max_range*1.15, mid_x + max_range*1.15)
    ax.set_ylim(mid_y - max_range*1.15, mid_y + max_range*1.15)
    ax.set_zlim(mid_z - max_range*1.15, mid_z + max_range*1.15)

    ax.set_title(title, fontsize=13, fontweight='bold')

# ---- 添加图例说明 ----
# 自定义图例条目
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

legend_elements = [
    Line2D([], [], color='red', linewidth=2, label='Tool center path'),
    Line2D([], [], color='gray', linewidth=1.5, linestyle='--', label='Contact path'),
    Line2D([], [], color='orange', linewidth=1, alpha=0.5, label='Tool axis direction'),
    plt.scatter([], [], color='lime', s=50, marker='o', label='Start'),
    plt.scatter([], [], color='red', s=50, marker='*', label='End'),
]

# 图注已移至正文说明，不在图上标注

ax1.legend(handles=legend_elements, fontsize=8, loc='upper left')
ax2.legend(handles=legend_elements, fontsize=8, loc='upper left')

plt.suptitle('Tool Path with Orientation Frames — Two Views',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = r'C:\Users\KCserver\projects\formal\机器人末端力控\code\tool_path_dual_view.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")
