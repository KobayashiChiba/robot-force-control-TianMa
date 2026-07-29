"""
plot_kc_calibration.py — K_C 标定验证图

两张图：球心曲线上接触面积随位置变化 + 力随位置变化
直接使用 sphere_contact.K_C，不做额外缩放。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, K_C

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
N = len(ball_ref)

s = np.zeros(N)
for i in range(1, N):
    s[i] = s[i-1] + np.linalg.norm(ball_ref[i] - ball_ref[i-1])

areas = np.zeros(N)
forces = np.zeros(N)

for i in range(N):
    F, a = sphere_contact_force(ball_ref[i], cz, cy)
    areas[i] = a
    forces[i] = np.linalg.norm(F)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 图1: 接触面积
ax1.plot(s, areas, 'b-', lw=0.8)
ax1.set_ylabel('Contact area (mm²)')
ax1.set_title('Contact area along ball center curve')
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.95, f'K_C = {K_C:.4f} (Fibonacci sampling)', transform=ax1.transAxes,
         fontsize=9, va='top')

# 图2: 力
ax2.plot(s, forces, 'r-', lw=1.2, label=f'|F| (K_C={K_C:.4f})')
ax2.axhline(8.0, color='gray', ls='--', lw=0.8, label='target 8N')
ax2.set_xlabel('Arc length (mm)')
ax2.set_ylabel('|F| (N)')
ax2.set_title(f'Force along ball center curve  (mean={forces.mean():.2f} ± {forces.std():.2f}N)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle(f'K_C = {K_C:.4f}  (Fibonacci sampling)', fontsize=13)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'output', 'kc_calibration.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f'✓ {out}')
print(f'  K_C = {K_C:.4f}')
print(f'  mean |F| = {forces.mean():.2f} ± {forces.std():.2f} N')
