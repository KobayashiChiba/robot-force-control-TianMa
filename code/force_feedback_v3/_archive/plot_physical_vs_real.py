"""
plot_physical_vs_real.py — 物理分块模型力场 vs 真实力场

格式: 3行(|F|/Fn/Fo) × 2列(物理模型/真实), (dn,db)∈[-2,2]mm
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from force_feedback_v3.lib import load_cylinders, load_ball_ref
from force_feedback_v3.lib.sphere_contact import sphere_contact_force, R_BALL
from force_feedback_v3.lib.force_mechanics import compute_point_basis_ortho
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts

# ── 物理分块模型正向 ──
K_FN, C_FN = 11.70, 6.48
K_RU, K_RD = 5.65, 1.60

def predict_physical(dn, db):
    Fn = min(0.0, -K_FN * dn - C_FN)
    if dn < 0:       Fo = 0.0
    elif db > 0:     Fo = K_RU * dn * db
    elif db < 0:     Fo = K_RD * dn * db
    else:            Fo = 0.0
    return Fn, Fo

# ── 截面叠加（复用） ──
def _sec_z(phi, cz, t, P0):
    x=cz.p1[0]+cz.radius*np.cos(phi); y=cz.p1[1]+cz.radius*np.sin(phi)
    return np.array([x, y, (np.dot(t,P0)-t[0]*x-t[1]*y)/t[2]])
def _sec_y(th, cy, t, P0):
    x=cy.p1[0]+cy.radius*np.cos(th); z=cy.p1[2]+cy.radius*np.sin(th)
    return np.array([x, (np.dot(t,P0)-t[0]*x-t[2]*z)/t[1], z])
def _dz(p): return min(abs(p-0),abs(p-0.5),abs(p-1.0))<0.05
def _dy(p): return min(abs(p-0.25),abs(p-0.75))<0.05

def _plot_split(ax, u, v, color):
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
                if in_seg:segs.append((start,i))
                start=i;in_seg=mask[i]
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

def draw_sections(ax, P_ct, t, n, o, p, uv_clip=5.0):
    dz=_dz(p); dy=_dy(p); N2=2000
    to_uv=lambda P:(np.dot(P-P_ct,n),np.dot(P-P_ct,o))
    clip=lambda u,v:(abs(u)<uv_clip)&(abs(v)<uv_clip)
    if abs(t[2])>1e-6:
        if dz:
            du,dv=n[2],o[2];L=np.hypot(du,dv)
            if L>1e-6:
                du,dv=du/L,dv/L;sgn=1 if du>0 else -1;ext=uv_clip*1.5
                ax.plot([0,sgn*ext*du],[0,sgn*ext*dv],color='darkcyan',lw=1.5)
                ax.plot([0,-sgn*ext*du],[0,-sgn*ext*dv],color='darkcyan',lw=1.5,ls=(0,(4,3)))
        else:
            uv_pts=[]
            for phi in np.linspace(0,2*np.pi,N2):
                P=_sec_z(phi,cz,t,P_ct);uu,vv=to_uv(P)
                if clip(uu,vv): uv_pts.append([uu,vv])
            if uv_pts:
                a=np.array(uv_pts);_plot_split(ax,a[:,0],a[:,1],'darkcyan')
    if abs(t[1])>1e-6:
        if dy:
            du,dv=n[1],o[1];L=np.hypot(du,dv)
            if L>1e-6:
                du,dv=du/L,dv/L;sgn=1 if du>0 else -1;ext=uv_clip*1.5
                ax.plot([0,sgn*ext*du],[0,sgn*ext*dv],color='green',lw=1.5)
                ax.plot([0,-sgn*ext*du],[0,-sgn*ext*dv],color='green',lw=1.5,ls=(0,(4,3)))
        else:
            uv_pts=[]
            for phi in np.linspace(0,2*np.pi,N2):
                P=_sec_y(phi,cy,t,P_ct);uu,vv=to_uv(P)
                if clip(uu,vv): uv_pts.append([uu,vv])
            if uv_pts:
                a=np.array(uv_pts);_plot_split(ax,a[:,0],a[:,1],'green')

# ── 网格 ──
R_OFF=2.0; NG=81
dv=np.linspace(-R_OFF,R_OFF,NG)

# p=0 位置
p=0.0; i0=int(p*(len(ball_ref)-1))
P_ball=ball_ref[i0]; idx0=np.argmin(np.linalg.norm(ct-P_ball,axis=1))
P_ct=ct[idx0]
basis=compute_point_basis_ortho(P_ct,sim.contact_geom)
n,o,t=basis.normal,basis.ortho,basis.tangent

# ── 物理模型力场 ──
Fm_phy=np.zeros((NG,NG)); Fn_phy=np.zeros_like(Fm_phy); Fo_phy=np.zeros_like(Fm_phy)
for i,dni in enumerate(dv):
    for j,dbj in enumerate(dv):
        fn_v,fo_v=predict_physical(dni,dbj)
        Fn_phy[j,i]=fn_v; Fo_phy[j,i]=fo_v; Fm_phy[j,i]=np.hypot(fn_v,fo_v)

# ── 真实力场（sphere_contact_force）──
Fm_real=np.zeros((NG,NG)); Fn_real=np.zeros_like(Fm_real); Fo_real=np.zeros_like(Fm_real)
for i,dni in enumerate(dv):
    for j,dbj in enumerate(dv):
        pos=P_ball+dni*n+dbj*o
        F,_=sphere_contact_force(pos,cz,cy)
        Fm_real[j,i]=np.linalg.norm(F)
        Fn_real[j,i]=np.dot(F,n)
        Fo_real[j,i]=np.dot(F,o)

# ── 画图 ──
fig,axes=plt.subplots(3,2,figsize=(11,16))

for col,(model_data,title) in enumerate([
    (Fm_phy, '物理分块'), (Fm_real, '真实力场')
]):
    for row,(data,ttl,cmap) in enumerate([
        ([Fm_phy,Fm_real][col], '|F| (N)', 'Reds'),
        ([Fn_phy,Fn_real][col], 'Fn (N)', 'RdBu_r'),
        ([Fo_phy,Fo_real][col], 'Fo (N)', 'RdBu_r'),
    ]):
        ax=axes[row,col]
        vmax=max(abs(data).max(),1e-6)
        vmin=0 if cmap=='Reds' else -vmax
        cs=ax.contourf(dv,dv,data,levels=20,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.colorbar(cs,ax=ax,shrink=0.85)
        draw_sections(ax,P_ct,t,n,o,p)
        ax.axhline(0,color='gray',lw=0.3); ax.axvline(0,color='gray',lw=0.3)
        ax.add_patch(Circle((0,0),R_BALL,fill=False,color='white',lw=1))
        ax.plot(0,0,'k+',ms=8,mew=2)
        ax.set_xlim(-R_OFF,R_OFF); ax.set_ylim(-R_OFF,R_OFF)
        ax.set_aspect('equal')
        if row==0: ax.set_title(title)
        if col==0: ax.set_ylabel(ttl)
        if row==2: ax.set_xlabel('dn (mm)')
        if col==1: ax.yaxis.set_label_position('right'); ax.set_ylabel('db (mm)')

fig.suptitle('物理分块力场 vs 真实力场  (p=0, dn∥法向 db∥o方向)',fontsize=14,y=0.995)
fig.tight_layout()

out=os.path.join(os.path.dirname(__file__),'..','output','physical_vs_real.png')
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches='tight')
print(f'✓ {out}')
