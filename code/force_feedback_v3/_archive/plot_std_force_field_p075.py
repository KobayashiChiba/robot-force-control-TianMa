"""
plot_std_force_field_p075.py — 标准工件力场，p=0.75

原点=标准球心bc0，力场=标准圆柱，截面=标准圆柱
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from force_feedback_v3.lib import load_cylinders, load_ball_ref, generate_error_cylinders
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.cylinder_geometry import sample_intersection

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()

# seed5 误差圆柱
np.random.seed(5)
rng = np.random.RandomState(5)
cz_err, cy_err = generate_error_cylinders(cy, cz, rng)

# 标准接触曲线
ct_std = sample_intersection(cy, cz, n_samples=2000).sample_pts

# p=0.75
i = 375
bc0 = ball_ref[i]
idx = np.argmin(np.linalg.norm(ct_std - bc0, axis=1))
P_ct = ct_std[idx]
basis = compute_point_basis_ortho(P_ct, sample_intersection(cy, cz, n_samples=2000))
n, o, t = basis.normal, basis.ortho, basis.tangent

# ── 截面 ──
def _sec_z(phi, cyl, t_vec, P0):
    x=cyl.p1[0]+cyl.radius*np.cos(phi); y=cyl.p1[1]+cyl.radius*np.sin(phi)
    return np.array([x,y,(np.dot(t_vec,P0)-t_vec[0]*x-t_vec[1]*y)/t_vec[2]])
def _sec_y(th, cyl, t_vec, P0):
    x=cyl.p1[0]+cyl.radius*np.cos(th); z=cyl.p1[2]+cyl.radius*np.sin(th)
    return np.array([x,(np.dot(t_vec,P0)-t_vec[0]*x-t_vec[2]*z)/t_vec[1],z])
def _plot_split(ax,u,v,color):
    tr=np.where(np.diff(u>=0))[0]
    if len(tr)>0:
        u2,v2=[],[]
        for i in range(len(u)):
            u2.append(u[i]);v2.append(v[i])
            if i in tr:
                r=-u[i]/(u[i+1]-u[i]) if abs(u[i+1]-u[i])>1e-12 else 0
                u2.append(0.0);v2.append(v[i]+r*(v[i+1]-v[i]))
        u,v=np.array(u2),np.array(v2)
    mp=u>=0
    for mask,ls in [(u<0,(0,(4,3))),(mp,'-')]:
        segs,start,in_seg=[],0,mask[0]
        for i in range(1,len(mask)):
            if mask[i]!=in_seg:
                if in_seg:segs.append((start,i)); start=i
                in_seg=mask[i]
        if in_seg:segs.append((start,len(mask)))
        for i0,i1 in segs:
            su=np.array(u[i0:i1]);sv=np.array(v[i0:i1])
            if ls!='-' and i1<len(mask) and mask[i1]!=mask[i1-1]:
                r0=-u[i1-1]/(u[i1]-u[i1-1]) if abs(u[i1]-u[i1-1])>1e-12 else 0
                su=np.append(su,0.0);sv=np.append(sv,v[i1-1]+r0*(v[i1]-v[i1-1]))
            if ls!='-' and i0>0 and mask[i0-1]!=mask[i0]:
                r0=u[i0-1]/(u[i0-1]-u[i0]) if abs(u[i0-1]-u[i0])>1e-12 else 0
                su=np.insert(su,0,0.0);sv=np.insert(sv,0,v[i0]+r0*(v[i0-1]-v[i0]))
            ax.plot(su,sv,color=color,lw=1.2,linestyle=ls)

def add_sections(ax, P_ct, t, n, o):
    N2=2000; uv_clip=5.0
    to_uv=lambda P:(np.dot(P-P_ct,n),np.dot(P-P_ct,o))
    clip=lambda u,v:(abs(u)<uv_clip)&(abs(v)<uv_clip)
    if abs(t[2])>1e-6:
        a=[]; [a.append(to_uv(_sec_z(phi,cz,t,P_ct))) for phi in np.linspace(0,2*np.pi,N2) if clip(*to_uv(_sec_z(phi,cz,t,P_ct)))]
        if a: _plot_split(ax,np.array(a)[:,0],np.array(a)[:,1],'darkcyan')
    if abs(t[1])>1e-6:
        a=[]; [a.append(to_uv(_sec_y(phi,cy,t,P_ct))) for phi in np.linspace(0,2*np.pi,N2) if clip(*to_uv(_sec_y(phi,cy,t,P_ct)))]
        if a: _plot_split(ax,np.array(a)[:,0],np.array(a)[:,1],'green')

# ── 力场 ──
R_OFF=2.0; NG=51; dv=np.linspace(-R_OFF,R_OFF,NG)
Fm=np.zeros((NG,NG)); Fn=np.zeros_like(Fm); Fo=np.zeros_like(Fm)
for ii,dni in enumerate(dv):
    for jj,dbj in enumerate(dv):
        pos=bc0+dni*n+dbj*o; F,_=sphere_contact_force(pos,cz_err,cy_err)
        Fm[jj,ii]=np.linalg.norm(F); Fn[jj,ii]=np.dot(F,n); Fo[jj,ii]=np.dot(F,o)

# ── 误差接触曲线交点：双扫phi+theta找最小3D距离 ──
best_3d=1e9; bp=None; bt=None
for phi in np.linspace(0, 2*np.pi, 2000):
    xz = cz_err.p1[0]+cz_err.radius*np.cos(phi)
    yz = cz_err.p1[1]+cz_err.radius*np.sin(phi)
    zz = (np.dot(t,P_ct)-t[0]*xz-t[1]*yz)/t[2]
    for theta in np.linspace(0, 2*np.pi, 2000):
        xc = cy_err.p1[0]+cy_err.radius*np.cos(theta)
        zc = cy_err.p1[2]+cy_err.radius*np.sin(theta)
        if abs(t[1])<1e-6: continue
        yc = (np.dot(t,P_ct)-t[0]*xc-t[2]*zc)/t[1]
        d3 = np.sqrt((xz-xc)**2+(yz-yc)**2+(zz-zc)**2)
        if d3 < best_3d: best_3d=d3; bp=phi; bt=theta
xz=cz_err.p1[0]+cz_err.radius*np.cos(bp); yz=cz_err.p1[1]+cz_err.radius*np.sin(bp)
zz=(np.dot(t,P_ct)-t[0]*xz-t[1]*yz)/t[2]
xc=cy_err.p1[0]+cy_err.radius*np.cos(bt); zc=cy_err.p1[2]+cy_err.radius*np.sin(bt)
yc=(np.dot(t,P_ct)-t[0]*xc-t[2]*zc)/t[1]
Pc_err = np.array([(xz+xc)/2, (yz+yc)/2, (zz+zc)/2])
err_dn = np.dot(Pc_err - P_ct, n)
err_db = np.dot(Pc_err - P_ct, o)

# ── 画图 ──
fig,axes=plt.subplots(1,3,figsize=(18,6))
for ax,data,title,cmap in [(axes[0],Fm,'|F| (N)','Reds'),(axes[1],Fn,'Fn (N)','RdBu_r'),(axes[2],Fo,'Fo (N)','RdBu_r')]:
    vmax=max(abs(data).max(),1e-6); vmin=0 if cmap=='Reds' else -vmax
    cs=ax.contourf(dv,dv,data,levels=20,cmap=cmap,vmin=vmin,vmax=vmax)
    plt.colorbar(cs,ax=ax,shrink=0.85)
    add_sections(ax,P_ct,t,n,o)
    # 误差圆柱截面叠加（洋红/橙色）——使用相同变换，圆柱换为误差版本
    err_sec_z = []; err_sec_y = []
    uv_clip=5.0
    to_uv=lambda P:(np.dot(P-P_ct,n),np.dot(P-P_ct,o))
    clip=lambda u,v:(abs(u)<uv_clip)&(abs(v)<uv_clip)
    if abs(t[2])>1e-6:
        for phi in np.linspace(0,2*np.pi,2000):
            P=_sec_z(phi,cz_err,t,P_ct); uu,vv=to_uv(P)
            if clip(uu,vv): err_sec_z.append([uu,vv])
        if err_sec_z: _plot_split(ax,np.array(err_sec_z)[:,0],np.array(err_sec_z)[:,1],'magenta')
    if abs(t[1])>1e-6:
        for phi in np.linspace(0,2*np.pi,2000):
            P=_sec_y(phi,cy_err,t,P_ct); uu,vv=to_uv(P)
            if clip(uu,vv): err_sec_y.append([uu,vv])
        if err_sec_y: _plot_split(ax,np.array(err_sec_y)[:,0],np.array(err_sec_y)[:,1],'orange')
    ax.add_patch(Circle((0,0),R_BALL,fill=False,color='white',lw=1))
    ax.plot(0,0,'k+',ms=10,mew=2,zorder=10)
    # 误差接触曲线最近点（红色方块）
    ax.plot(err_dn, err_db, 'rs', ms=8, mew=2, zorder=10, label=f'Pc_err')
    ax.legend(fontsize=7)
    ax.axhline(0,color='gray',lw=0.3); ax.axvline(0,color='gray',lw=0.3)
    ax.set_xlim(-R_OFF,R_OFF); ax.set_ylim(-R_OFF,R_OFF); ax.set_aspect('equal')
    ax.set_xlabel('dn (mm)'); ax.set_ylabel('db (mm)'); ax.set_title(title)

fig.suptitle('Standard force field — p=0.75',fontsize=14)
fig.tight_layout()
out=os.path.join(os.path.dirname(__file__),'..','output','std_force_field_p075.png')
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches='tight')
print(f'✓ {out}')
