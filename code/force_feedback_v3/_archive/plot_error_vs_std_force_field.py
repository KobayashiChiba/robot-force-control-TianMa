"""
plot_error_vs_std_force_field.py — 同一位置(p=0.25,seed5) 标准工件 vs 误差工件力场对比

左边=标准工件力场，右边=误差工件力场，Fo行直接对比db敏感度差异
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

np.random.seed(5)
cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
rng = np.random.RandomState(5)
cz_err, cy_err = generate_error_cylinders(cy, cz, rng)

# 误差接触曲线 + 标架
contact_err_geom = sample_intersection(cy_err, cz_err, n_samples=2000)
ct_err = contact_err_geom.sample_pts
# 标准接触曲线 + 标架
contact_std_geom = sample_intersection(cy, cz, n_samples=2000)
ct_std = contact_std_geom.sample_pts

p = 0.25; i = int(p*(len(ball_ref)-1)); bc0 = ball_ref[i]

idx_e = np.argmin(np.linalg.norm(ct_err-bc0, axis=1))
Pc_e = ct_err[idx_e]; basis_e = compute_point_basis_ortho(Pc_e, contact_err_geom)
n_e, o_e, t_e = basis_e.normal, basis_e.ortho, basis_e.tangent

idx_s = np.argmin(np.linalg.norm(ct_std-bc0, axis=1))
Pc_s = ct_std[idx_s]; basis_s = compute_point_basis_ortho(Pc_s, contact_std_geom)
n_s, o_s, t_s = basis_s.normal, basis_s.ortho, basis_s.tangent

# ── 截面叠加 ──
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

def add_sections(ax,cy,cz,Pc,t,n,o):
    N2=2000; uv_clip=5.0
    to_uv=lambda P:(np.dot(P-Pc,n),np.dot(P-Pc,o))
    clip=lambda u,v:(abs(u)<uv_clip)&(abs(v)<uv_clip)
    if abs(t[2])>1e-6:
        a=[]; [a.append(to_uv(_sec_z(phi,cz,t,Pc))) for phi in np.linspace(0,2*np.pi,N2) if clip(*to_uv(_sec_z(phi,cz,t,Pc)))]
        if a: _plot_split(ax,np.array(a)[:,0],np.array(a)[:,1],'darkcyan')
    if abs(t[1])>1e-6:
        a=[]; [a.append(to_uv(_sec_y(phi,cy,t,Pc))) for phi in np.linspace(0,2*np.pi,N2) if clip(*to_uv(_sec_y(phi,cy,t,Pc)))]
        if a: _plot_split(ax,np.array(a)[:,0],np.array(a)[:,1],'green')

# ── 力场采样 ──
R_OFF=2.0; NG=51; dv=np.linspace(-R_OFF,R_OFF,NG)
grids = {}

# 标准工件: 以 bc0 为原点, 用标准标架
for (cyl_y,cyl_z,n,o),label in [((cy,cz,n_s,o_s),'std'),((cy_err,cz_err,n_e,o_e),'err')]:
    Fm=np.zeros((NG,NG)); Fn=np.zeros_like(Fm); Fo=np.zeros_like(Fm)
    for ii,dni in enumerate(dv):
        for jj,dbj in enumerate(dv):
            pos=bc0+dni*n+dbj*o; F,_=sphere_contact_force(pos,cyl_z,cyl_y)
            Fm[jj,ii]=np.linalg.norm(F); Fn[jj,ii]=np.dot(F,n); Fo[jj,ii]=np.dot(F,o)
    grids[label]=(Fm,Fn,Fo)

# ── 画图: 3行×2列, 左=std 右=err ──
fig,axes=plt.subplots(3,2,figsize=(12,16))

for col,(data,cy_cyl,cz_cyl,Pc,n,o,t,title) in enumerate([
    (grids['std'], cy,cz, Pc_s,n_s,o_s,t_s, 'Standard'),
    (grids['err'], cy_err,cz_err, Pc_e,n_e,o_e,t_e, 'Error (seed5)'),
]):
    for row,(d,ttl,cmap) in enumerate(zip(data,['|F| (N)','Fn (N)','Fo (N)'],['Reds','RdBu_r','RdBu_r'])):
        ax=axes[row,col]
        vmax=max(abs(d).max(),1e-6); vmin=0 if cmap=='Reds' else -vmax
        cs=ax.contourf(dv,dv,d,levels=20,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.colorbar(cs,ax=ax,shrink=0.85)
        add_sections(ax,cy_cyl,cz_cyl,Pc,n,o,t)
        ax.add_patch(Circle((0,0),R_BALL,fill=False,color='white',lw=1))
        ax.plot(0,0,'k+',ms=10,mew=2,zorder=10)
        ax.axhline(0,color='gray',lw=0.3); ax.axvline(0,color='gray',lw=0.3)
        ax.set_xlim(-R_OFF,R_OFF); ax.set_ylim(-R_OFF,R_OFF); ax.set_aspect('equal')
        if row==0: ax.set_title(title)
        if col==0: ax.set_ylabel(ttl)
        if row==2: ax.set_xlabel('dn (mm)')
        if col==1: ax.set_ylabel('db (mm)'); ax.yaxis.set_label_position('right')

fig.suptitle('Local force field @ p=0.25 — Standard vs Error workpiece', fontsize=14)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__),'..','output','error_vs_std_force_field.png')
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches='tight')
print(f'✓ {out}')
