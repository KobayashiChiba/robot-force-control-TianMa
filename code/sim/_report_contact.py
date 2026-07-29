"""
接触力可视化 — 报告用3D图
球刀切入工件，接触区域高亮，XYZ总跨度10mm，三轴等比例
"""
import sys, os, pickle, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))

from force_mechanics_v2 import compute_point_basis_ortho
from cylinder_geometry_v2 import sample_intersection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

with open(os.path.join(_sdir, '..', 'data', 'force_model.pkl'), 'rb') as f:
    d = pickle.load(f)
cy, cz = d['cyl_contact_y'], d['cyl_contact_z']
ball_ref = d['ball_center_500']
contact_geom = sample_intersection(cy, cz, n_samples=2000)
contact_pts = contact_geom.sample_pts

# 球刀中心（沿法向压入0.5mm）
P_ball = ball_ref[0].copy()
dists = np.linalg.norm(contact_pts - P_ball, axis=1)
i_ct = np.argmin(dists)
P_ct = contact_pts[i_ct]
basis = compute_point_basis_ortho(P_ct, contact_geom)
P_ball = P_ball + 0.5 * basis.normal

R_BALL = 4.2
HALF = 5.0

# 最近接触点
dists = np.linalg.norm(contact_pts - P_ball, axis=1)
P_contact = contact_pts[np.argmin(dists)]

def inside_cyl_z(pts):  # Z圆柱内部
    X0,Y0 = cz.axis_point[0],cz.axis_point[1]
    return np.sqrt((pts[:,0]-X0)**2+(pts[:,1]-Y0)**2) < cz.radius-1e-6

def inside_cyl_y(pts):  # Y圆柱内部
    X0,Z0 = cy.axis_point[0],cy.axis_point[2]
    return np.sqrt((pts[:,0]-X0)**2+(pts[:,2]-Z0)**2) < cy.radius-1e-6

# === 绘图 ===
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Z圆柱面
phi = np.linspace(0,2*np.pi,72)
zs = np.linspace(P_ball[2]-HALF,P_ball[2]+HALF,20)
Phi,Zz = np.meshgrid(phi,zs)
ax.plot_surface(cz.axis_point[0]+cz.radius*np.cos(Phi),
                cz.axis_point[1]+cz.radius*np.sin(Phi),
                Zz, alpha=0.15, color='#4477AA', edgecolor='none')

# Y圆柱面
th = np.linspace(0,2*np.pi,72)
ys = np.linspace(P_ball[1]-HALF,P_ball[1]+HALF,20)
Th,Yy = np.meshgrid(th,ys)
ax.plot_surface(cy.axis_point[0]+cy.radius*np.cos(Th), Yy,
                cy.axis_point[2]+cy.radius*np.sin(Th),
                alpha=0.15, color='#44AA44', edgecolor='none')

# 球刀线框
u=np.linspace(0,2*np.pi,24); v=np.linspace(0,np.pi,12)
xs=P_ball[0]+R_BALL*np.outer(np.cos(u),np.sin(v))
ys=P_ball[1]+R_BALL*np.outer(np.sin(u),np.sin(v))
zs=P_ball[2]+R_BALL*np.outer(np.ones_like(u),np.cos(v))
ax.plot_wireframe(xs,ys,zs,color='gray',alpha=0.3,lw=0.3)

# 球面接触区域
n_th,n_ph=40,80
th_=np.linspace(0,np.pi,n_th); ph_=np.linspace(0,2*np.pi,n_ph,endpoint=False)
Th_,Ph_=np.meshgrid(th_,ph_)
xd=P_ball[0]+R_BALL*np.sin(Th_)*np.cos(Ph_)
yd=P_ball[1]+R_BALL*np.sin(Th_)*np.sin(Ph_)
zd=P_ball[2]+R_BALL*np.cos(Th_)
pts_d=np.column_stack([xd.ravel(),yd.ravel(),zd.ravel()])
contact=~inside_cyl_z(pts_d)&~inside_cyl_y(pts_d)
if contact.any():
    ax.scatter(xd.ravel()[contact],yd.ravel()[contact],zd.ravel()[contact],
               c='#FF4500',s=10,alpha=0.8,label='接触区域')

# 接触曲线
ax.plot(contact_pts[:,0],contact_pts[:,1],contact_pts[:,2],
        'k-',lw=1.2,alpha=0.6,label='接触曲线')

# 关键点
ax.scatter(*P_contact,c='black',s=80,zorder=10,label='接触点')
ax.scatter(*P_ball,c='red',s=100,zorder=10,label='球刀中心')
ax.plot([P_ball[0],P_contact[0]],[P_ball[1],P_contact[1]],[P_ball[2],P_contact[2]],
        'k--',lw=1,alpha=0.5)
ax.quiver(P_contact[0],P_contact[1],P_contact[2],
          basis.normal[0]*3,basis.normal[1]*3,basis.normal[2]*3,
          color='blue',lw=2,label='法向量')

# 范围与等比例
for lim,center in [(ax.set_xlim,P_ball[0]),(ax.set_ylim,P_ball[1]),(ax.set_zlim,P_ball[2])]:
    lim(center-HALF,center+HALF)
ax.set_aspect('equal')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
ax.set_title(r'球刀-工件接触力计算  $F = k_c \sqrt{S}$',fontsize=14,pad=20)
ax.legend(fontsize=9,loc='upper left')
ax.view_init(elev=18,azim=-55)

fig.tight_layout()
out=os.path.join(_sdir,'output','report_contact_force.png')
fig.savefig(out,dpi=200,bbox_inches='tight')
print(f'已保存 {out}')
plt.close(fig)
