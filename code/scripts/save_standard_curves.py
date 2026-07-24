"""
精简版：只保存两个 Geom + ball_radius
"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

ROOT = r'C:\Users\KCserver\projects\formal\机器人末端力控\code'

df = pd.read_excel(os.path.join(ROOT, 'data', '球刀中心点及轮廓轨迹点.xlsx'))
pts_ball = np.column_stack([df['X'].values, df['Y'].values, df['Z'].values])
pts_contact = np.column_stack([df['x'].values, df['y'].values, df['z'].values])

# Z修正
_, geom_c = fit_cylinders_from_points(pts_contact, 'Y', 'Z')
curve_c = resample_curve(geom_c, n_samples=10000)
shift_z = curve_c.sample_pts[:, 2].mean() - pts_ball[:, 2].mean()
pts_ball[:, 2] += shift_z

# 拟合 + 500点采样
_, geom_ball = fit_cylinders_from_points(pts_ball, 'Y', 'Z')
geom_ball = resample_curve(geom_ball, n_samples=500)
geom_contact = resample_curve(geom_c, n_samples=500)

BALL_RADIUS = 4.0

data = {
    'ball_center': geom_ball,
    'contact': geom_contact,
    'ball_radius': BALL_RADIUS,
    'shift_z': shift_z,
}

with open(os.path.join(ROOT, 'data', 'standard_curves.pkl'), 'wb') as f:
    pickle.dump(data, f)

print('球刀中心: r1=%.1f r2=%.1f  采样=%d' % (geom_ball.r1, geom_ball.r2, geom_ball.n_samples))
print('标准接触: r1=%.1f r2=%.1f  采样=%d' % (geom_contact.r1, geom_contact.r2, geom_contact.n_samples))
print('球刀半径: %.1f mm' % BALL_RADIUS)
print('Z偏移:    %+.3f mm' % shift_z)
print('已保存: standard_curves.pkl')
