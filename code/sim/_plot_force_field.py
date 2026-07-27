"""力场热力图 — 用写死的二次函数"""
import sys, os, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_field_fixed import predict

R, N = 3, 80
dn = np.linspace(-R, R, N)
db = np.linspace(-R, R, N)
DN, DB = np.meshgrid(dn, db)

FN = np.zeros_like(DN)
FO = np.zeros_like(DN)
for i in range(N):
    for j in range(N):
        fn, fo = predict(DN[i,j], DB[i,j])
        FN[i,j] = fn
        FO[i,j] = fo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Fn
c1 = ax1.contourf(DN, DB, FN, levels=30, cmap='RdBu_r')
plt.colorbar(c1, ax=ax1, label='Fn (N)')
ax1.axhline(0, color='gray', lw=0.5); ax1.axvline(0, color='gray', lw=0.5)
ax1.plot(0, 0, 'ko', markersize=6)
ax1.set_xlabel('dn (mm)'); ax1.set_ylabel('db (mm)')
ax1.set_title(f'Fn(dn,db)  Fn(0,0)={predict(0,0)[0]:.2f}N')

# Fo
c2 = ax2.contourf(DN, DB, FO, levels=30, cmap='RdBu_r')
plt.colorbar(c2, ax=ax2, label='Fo (N)')
ax2.axhline(0, color='gray', lw=0.5); ax2.axvline(0, color='gray', lw=0.5)
ax2.plot(0, 0, 'ko', markersize=6)
ax2.set_xlabel('dn (mm)'); ax2.set_ylabel('db (mm)')
ax2.set_title(f'Fo(dn,db)  Fo(0,0)={predict(0,0)[1]:.2f}N')

fig.suptitle('力场二次模型 (写死系数)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'force_field_heatmap.png')
fig.savefig(out, dpi=150)
print(f'已保存 {out}')
plt.close(fig)
