"""Fibonacci球面采样可视化"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def fibonacci_sphere(n):
    """Fibonacci 球面均匀采样"""
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i+0.5)/n)      # 极角，均匀分布在 cos⁻¹
    theta = np.pi * (1+np.sqrt(5)) * i      # 黄金比例螺旋
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x,y,z])

def equiangular_sphere(n_th, n_ph):
    """等角间距采样（当前方法）"""
    th = np.linspace(0, np.pi, n_th)
    ph = np.linspace(0, 2*np.pi, n_ph, endpoint=False)
    Th,Ph = np.meshgrid(th,ph)
    x = np.sin(Th)*np.cos(Ph)
    y = np.sin(Th)*np.sin(Ph)
    z = np.cos(Th)
    pts = np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    return pts, Th.ravel(), Ph.ravel()

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Fibonacci: ~12800点（≈80×160）
n = 12800
pts_fib = fibonacci_sphere(n)
ax1.scatter(pts_fib[:,0], pts_fib[:,1], pts_fib[:,2], c='steelblue', s=0.5, alpha=0.6)
ax1.set_title(f'Fibonacci 球面采样 ({n}点)')
ax1.set_box_aspect([1,1,1])

# 等角间距: 80×160
pts_eq, _, _ = equiangular_sphere(80, 160)
ax2.scatter(pts_eq[:,0], pts_eq[:,1], pts_eq[:,2], c='orange', s=0.5, alpha=0.6)
ax2.set_title(f'等角间距采样 (80×160={80*160}点)')
ax2.set_box_aspect([1,1,1])

fig.tight_layout()
out = 'output/fibonacci_sphere.png'
import os
fig.savefig(os.path.join(os.path.dirname(__file__), out), dpi=150)
print(f'已保存 {out}')
plt.close(fig)
