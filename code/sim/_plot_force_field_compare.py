"""力场对比 — 写死二次函数 vs 5个位置的实际模拟力"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
from force_field_fixed import predict
from sphere_contact import sphere_contact_force
from force_mechanics_v2 import compute_point_basis_ortho
from cylinder_geometry_v2 import sample_intersection

with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    d = pickle.load(f)
cy, cz, ball = d['cyl_contact_y'], d['cyl_contact_z'], d['ball_center_500']
geom = sample_intersection(cy, cz, n_samples=2000)

R, N = 3, 50
dn = np.linspace(-R, R, N)
db = np.linspace(-R, R, N)
DN, DB = np.meshgrid(dn, db)

# 解析力场
FN_ANA = np.zeros_like(DN)
FO_ANA = np.zeros_like(DN)
for i in range(N):
    for j in range(N):
        fn, fo = predict(DN[i,j], DB[i,j])
        FN_ANA[i,j] = fn; FO_ANA[i,j] = fo

# 5个位置
positions = [0, 0.25, 0.50, 0.75, 0.99]
pos_data = []
for p in positions:
    i = int(p * 499)
    P = ball[i]
    idx = np.argmin(np.linalg.norm(geom.sample_pts - P, axis=1))
    Pc = geom.sample_pts[idx]
    basis = compute_point_basis_ortho(Pc, geom)
    n, o = basis.normal, basis.ortho
    FN_REAL = np.zeros_like(DN)
    FO_REAL = np.zeros_like(DN)
    for ii in range(N):
        for jj in range(N):
            pos = P + DN[ii,jj]*n + DB[ii,jj]*o
            F, _ = sphere_contact_force(pos, cz, cy)
            FN_REAL[ii,jj] = np.dot(F, n)
            FO_REAL[ii,jj] = np.dot(F, o)
    pos_data.append((p, P, FN_REAL, FO_REAL))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(6, 2, figsize=(16, 30))

# 行0: 解析力场
for col, (data, title, cmap) in enumerate([
    (FN_ANA, f'Fn 解析  Fn(0,0)={predict(0,0)[0]:.1f}N', 'RdBu_r'),
    (FO_ANA, f'Fo 解析  Fo(0,0)={predict(0,0)[1]:.2f}N', 'RdBu_r'),
]):
    ax = axes[0, col]
    c = ax.contourf(DN, DB, data, levels=30, cmap=cmap)
    plt.colorbar(c, ax=ax)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.plot(0, 0, 'ko', markersize=6)
    ax.set_xlabel('dn (mm)'); ax.set_ylabel('db (mm)')
    ax.set_title(title, fontsize=10)

# 行1-5: 实际力场
for row_idx, (p, P, FN_R, FO_R) in enumerate(pos_data):
    for col, (data, cmap) in enumerate([(FN_R, 'RdBu_r'), (FO_R, 'RdBu_r')]):
        ax = axes[row_idx+1, col]
        c = ax.contourf(DN, DB, data, levels=30, cmap=cmap)
        plt.colorbar(c, ax=ax)
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.plot(0, 0, 'ko', markersize=6)
        ax.set_xlabel('dn (mm)'); ax.set_ylabel('db (mm)')
        lbl = 'Fn' if col==0 else 'Fo'
        ax.set_title(f'{lbl} p={p:.2f}  零偏移 {lbl}={data[N//2,N//2]:+.1f}N', fontsize=10)

fig.suptitle('力场对比：解析(写死系数) vs 5个位置实际模拟力', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'force_field_compare.png')
fig.savefig(out, dpi=120)
print(f'已保存 {out}')
plt.close(fig)
