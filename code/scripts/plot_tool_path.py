import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
path = r"C:\Users\KCserver\AppData\Local\hermes\cache\documents\doc_2d04c29bc292_球刀中心点及轮廓轨迹点.xlsx"
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

fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# ---- 球心轨迹 ----
# 颜色渐变：起点→终点（彩虹）
colors = plt.cm.jet(np.linspace(0, 1, len(cx)))
for i in range(len(cx)-1):
    ax.plot(cx[i:i+2], cy[i:i+2], cz[i:i+2], color=colors[i], linewidth=2.0)

# ---- 起始/结束标记 ----
ax.scatter(*[cx[0], cy[0], cz[0]], color='lime', s=80, marker='o', zorder=5, label='Start')
ax.scatter(*[cx[-1], cy[-1], cz[-1]], color='red', s=80, marker='*', zorder=5, label='End')

# ---- 姿态坐标架（每N个点画一个） ----
step = 8
arrow_len = 3.0
for i in range(0, len(cx), step):
    R = quat_to_rot([q1[i], q2[i], q3[i], q4[i]])
    origin = np.array([cx[i], cy[i], cz[i]])
    # 工具坐标系: x=红, y=绿, z=蓝
    ax.quiver(*origin, *R[:, 0]*arrow_len, color='red', alpha=0.7, linewidth=1.0)
    ax.quiver(*origin, *R[:, 1]*arrow_len, color='green', alpha=0.7, linewidth=1.0)
    ax.quiver(*origin, *R[:, 2]*arrow_len, color='blue', alpha=0.7, linewidth=1.0)

# ---- 接触点轨迹（虚线） ----
ax.plot(px, py, pz, color='gray', linewidth=1.5, linestyle='--', alpha=0.6, label='Contact path')

# ---- 连接球心→接触点（少量连线示意刀具方向） ----
for i in range(0, len(cx), step):
    ax.plot([cx[i], px[i]], [cy[i], py[i]], [cz[i], pz[i]],
            color='orange', alpha=0.25, linewidth=0.8)

# ---- 轴标签和范围 ----
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Tool Center Path with Orientation Frames\n'
             '(RGB: tool coordinate system | dashed: contact path | orange: tool axis)',
             fontsize=12)

# ---- 轴等比例 ----
max_range = max(cx.max()-cx.min(), cy.max()-cy.min(), cz.max()-cz.min()) / 2
mid_x, mid_y, mid_z = (cx.max()+cx.min())/2, (cy.max()+cy.min())/2, (cz.max()+cz.min())/2
ax.set_xlim(mid_x - max_range*1.1, mid_x + max_range*1.1)
ax.set_ylim(mid_y - max_range*1.1, mid_y + max_range*1.1)
ax.set_zlim(mid_z - max_range*1.1, mid_z + max_range*1.1)

# ---- 图例 ----
legend_elements = [
    plt.Line2D([], [], color='red', linewidth=2, label='Tool center path'),
    plt.Line2D([], [], color='gray', linewidth=1.5, linestyle='--', label='Contact path'),
    plt.scatter([], [], color='lime', s=50, marker='o', label='Start'),
    plt.scatter([], [], color='red', s=50, marker='*', label='End'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

plt.tight_layout()
output_path = r'C:\Users\KCserver\projects\formal\机器人末端力控\code\tool_path_with_orientation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {output_path}")
