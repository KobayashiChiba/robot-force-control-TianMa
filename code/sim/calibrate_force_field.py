"""
calibrate_force_field.py — 力场标定：统一二次多项式拟合

从 force_model.pkl 取 20 个进度位置，各采力场数据，统一拟合：
  ΔFn = fn(dn, db)   ΔFo = fo(dn, db)

拟合结果 (2026-07-27, 4778 个接触区采样点):
  ΔFn = -14.232*dn + 0.484*db + 1.345*dn² + 0.511*dn*db + 1.465*db²
  ΔFo = +1.381*dn + 1.076*db + 1.002*dn² + 2.819*dn*db - 0.290*db²

逆推验证: 随机采样误差 0.04~0.22mm

输出: data/force_field_calib.npz (系数 + 基准表)
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../lib_v2')

import pickle
import numpy as np
from force_profile import sphere_contact_force
from contact_frame_v2 import compute_frame

# ── 加载数据 ──
with open('../data/force_model.pkl', 'rb') as f:
    d = pickle.load(f)

ball = d['ball_center_500']
cy = d['cyl_contact_y']
cz = d['cyl_contact_z']
cg = d['contact_geom'].sample_pts
Npts = len(ball)

R = 2        # 偏移范围 (mm)
Ng = 21      # 网格分辨率
N_positions = 20  # 采样位置数

dn = np.linspace(-R, R, Ng)
db = np.linspace(-R, R, Ng)
DN, DB = np.meshgrid(dn, db)

# ── 采力场数据 ──
base_table = {}  # {p_idx: (Fn0, Fo0)}
all_dFn = []
all_dFo = []
AA = []  # [dn, db, dn², dn*db, db²]

for p in np.linspace(0, 0.99, N_positions):
    i = min(int(p * Npts), Npts - 1)
    bc0 = ball[i]
    idx = np.argmin(np.linalg.norm(cg - bc0, axis=1))
    Pc = cg[idx]
    frame = compute_frame(Pc, cy, cz)
    n = frame.normal
    b = np.cross(frame.tangent, n)
    b /= np.linalg.norm(b)

    f0, _ = sphere_contact_force(bc0, np.array([0, -1, 0]), cz, cy)
    Fn0 = np.dot(f0, n)
    Fo0 = np.dot(f0, b)
    base_table[i] = (Fn0, Fo0)

    for ii, dni in enumerate(dn):
        for jj, dbj in enumerate(db):
            f, _ = sphere_contact_force(
                bc0 + dni*n + dbj*b, np.array([0, -1, 0]), cz, cy)
            if np.linalg.norm(f) > 0.5:  # 仅接触区
                all_dFn.append(np.dot(f, n) - Fn0)
                all_dFo.append(np.dot(f, b) - Fo0)
                AA.append([dni, dbj, dni**2, dni*dbj, dbj**2])

all_dFn = np.array(all_dFn)
all_dFo = np.array(all_dFo)
AA = np.array(AA)

# ── 最小二乘拟合 ──
c_dfn, _, _, _ = np.linalg.lstsq(AA, all_dFn, rcond=None)
c_dfo, _, _, _ = np.linalg.lstsq(AA, all_dFo, rcond=None)

print(f'基于 {len(all_dFn)} 个接触区采样点，统一二次拟合:')
print(f'  ΔFn = {c_dfn[0]:+.3f}*dn {c_dfn[1]:+.3f}*db {c_dfn[2]:+.3f}*dn² {c_dfn[3]:+.3f}*dn*db {c_dfn[4]:+.3f}*db²')
print(f'  ΔFo = {c_dfo[0]:+.3f}*dn {c_dfo[1]:+.3f}*db {c_dfo[2]:+.3f}*dn² {c_dfo[3]:+.3f}*dn*db {c_dfo[4]:+.3f}*db²')

# ── 保存 ──
np.savez('../data/force_field_calib.npz',
         c_dfn=c_dfn, c_dfo=c_dfo,
         base_indices=list(base_table.keys()),
         base_Fn0=np.array([base_table[k][0] for k in sorted(base_table)]),
         base_Fo0=np.array([base_table[k][1] for k in sorted(base_table)]))
print('\n已保存 data/force_field_calib.npz')
