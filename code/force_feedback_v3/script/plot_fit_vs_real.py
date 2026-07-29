"""
plot_fit_vs_real.py — 左1列=拟合力场(p=0)，右6列=真实力场+截面叠加
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
from force_feedback_v3.lib.force_field_quadratic import inverse as inverse2
from force_feedback_v3.lib.simulator import Simulator

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

cy, cz = load_cylinders()
ball_ref, _ = load_ball_ref()
sim = Simulator(cy, cz)
ct = sim.contact_pts

# ── 系数 ──
C0_DN, C1_DN, C2_DN, C3_DN, C4_DN, C5_DN = -0.477186, -0.054439, -0.052543, 0.000651, -0.002317, 0.004243
C0_DB, C1_DB, C2_DB, C3_DB, C4_DB, C5_DB = -0.355445, -0.050930, 0.360596, -0.000812, 0.010073, 0.004712

# ── 建查表 ──
TN = 300; DIST = 0.1; FN_LO,FN_HI = -70,5; FO_LO,FO_HI = -70,70
ft = np.linspace(FN_LO,FN_HI,TN); fot = np.linspace(FO_LO,FO_HI,TN)
DNT = np.zeros((TN,TN)); DBT = np.zeros((TN,TN))
for i,fi in enumerate(ft):
    for j,fj in enumerate(fot):
        Fn_c = max(FN_LO,min(0.0,fi)) if fi>0 else max(FN_LO,fi)
        Fo_c = max(FO_LO,min(FO_HI,fj))
        x = np.array([1.0,Fn_c,Fo_c,Fn_c**2,Fn_c*Fo_c,Fo_c**2])
        DNT[j,i] = float(x@[C0_DN,C1_DN,C2_DN,C3_DN,C4_DN,C5_DN])
        DBT[j,i] = float(x@[C0_DB,C1_DB,C2_DB,C3_DB,C4_DB,C5_DB])
df = DNT.ravel(); dbf = DBT.ravel()
ffn = np.tile(ft,TN); ffo = np.repeat(fot,TN)

def lookup(dn,db):
    k = np.argmin((df-dn)**2+(dbf-db)**2)
    if np.hypot(df[k]-dn,dbf[k]-db) > DIST:
        return 0.0,0.0
    return float(ffn[k]), float(ffo[k])

# ── 截面叠加 ──
def _sec_z(phi, cz, t, P0):
    x=cz.p1[0]+cz.radius*np.cos(phi); y=cz.p1[1]+cz.radius*np.sin(phi)
    return np.array([x,y,(np.dot(t,P0)-t[0]*x-t[1]*y)/t[2]])
def _sec_y(th, cy, t, P0):
    x=cy.p1[0]+cy.radius*np.cos(th); z=cy.p1[2]+cy.radius*np.sin(th)
    return np.array([x,(np.dot(t,P0)-t[0]*x-t[2]*z)/t[1],z])

def _should_degenerate_z(p):
    return min(abs(p-0),abs(p-0.5),abs(p-1.0))<0.05
def _should_degenerate_y(p):
    return min(abs(p-0.25),abs(p-0.75))<0.05

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
    dz=_should_degenerate_z(p); dy=_should_degenerate_y(p)
    N2=2000; to_uv=lambda P:(np.dot(P-P_ct,n),np.dot(P-P_ct,o))
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

# ── 画图 ──
PROGS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
R_OFF=2.0;NG=41;dv=np.linspace(-R_OFF,R_OFF,NG)
N_COLS = 1 + len(PROGS)  # 1拟合 + 6真实

fig,axes=plt.subplots(3,N_COLS,figsize=(5.5*N_COLS,16))

# ── 列0: p=0 拟合力场 ──
p0=0.0; i0=int(p0*(len(ball_ref)-1))
P_ball0=ball_ref[i0]; idx0=np.argmin(np.linalg.norm(ct-P_ball0,axis=1))
basis0=compute_point_basis_ortho(ct[idx0],sim.contact_geom)
n0,o0=basis0.normal,basis0.ortho
fm=np.zeros((NG,NG)); fn_arr=np.zeros_like(fm); fo_arr=np.zeros_like(fm)
for ii,dni in enumerate(dv):
    for jj,dbj in enumerate(dv):
        ffi,ffj=lookup(dni,dbj)
        fn_arr[jj,ii]=ffi; fo_arr[jj,ii]=ffj; fm[jj,ii]=np.hypot(ffi,ffj)

for row,(data,ttl,cmap) in enumerate([(fm,'|F| (N)','Reds'),(fn_arr,'Fn (N)','RdBu_r'),(fo_arr,'Fo (N)','RdBu_r')]):
    ax=axes[row,0]
    vmax=max(abs(data).max(),1e-6); vmin=0 if cmap=='Reds' else -vmax
    cs=ax.contourf(dv,dv,data,levels=15,cmap=cmap,vmin=vmin,vmax=vmax)
    plt.colorbar(cs,ax=ax,shrink=0.8)
    ax.plot(0,0,'k+',ms=6,mew=2)
    ax.axhline(0,color='gray',lw=0.3); ax.axvline(0,color='gray',lw=0.3)
    ax.set_xlim(-R_OFF,R_OFF); ax.set_ylim(-R_OFF,R_OFF); ax.set_aspect('equal')
    if row==0: ax.set_title('拟合\np=0.00')
    if row==2: ax.set_xlabel('dn'); ax.set_ylabel('db')

# ── 列1-6: 真实力场+截面 ──
for col,p in enumerate(PROGS):
    i=int(p*(len(ball_ref)-1));P_ball=ball_ref[i]
    idx=np.argmin(np.linalg.norm(ct-P_ball,axis=1));P_ct=ct[idx]
    basis=compute_point_basis_ortho(P_ct,sim.contact_geom)
    n,o,t=basis.normal,basis.ortho,basis.tangent

    rm=np.zeros((NG,NG)); rn=np.zeros_like(rm); ro=np.zeros_like(rm)
    for ii,dni in enumerate(dv):
        for jj,dbj in enumerate(dv):
            pos=P_ball+dni*n+dbj*o;F,_=sphere_contact_force(pos,cz,cy)
            rm[jj,ii]=np.linalg.norm(F)
            rn[jj,ii]=np.dot(F,n); ro[jj,ii]=np.dot(F,o)

    for row,(data,ttl,cmap) in enumerate([(rm,'|F| (N)','Reds'),(rn,'Fn (N)','RdBu_r'),(ro,'Fo (N)','RdBu_r')]):
        ax=axes[row,1+col]
        vmax=max(abs(data).max(),1e-6); vmin=0 if cmap=='Reds' else -vmax
        cs=ax.contourf(dv,dv,data,levels=15,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.colorbar(cs,ax=ax,shrink=0.8)
        draw_sections(ax,P_ct,t,n,o,p)
        ax.axhline(0,color='gray',lw=0.3); ax.axvline(0,color='gray',lw=0.3)
        ax.add_patch(Circle((0,0),R_BALL,fill=False,color='white',lw=1))
        ax.plot(0,0,'k+',ms=8,mew=2)
        ax.set_xlim(-R_OFF,R_OFF); ax.set_ylim(-R_OFF,R_OFF); ax.set_aspect('equal')
        if row==0: ax.set_title(f'真实\np={p:.2f}')

fig.suptitle('Left: fitted force field (p=0)  |  Right: real force field + section curves',fontsize=14)
fig.tight_layout()
out=os.path.join(os.path.dirname(__file__),'..','output','fit_vs_real.png')
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150)
print(f'✓ {out}')
