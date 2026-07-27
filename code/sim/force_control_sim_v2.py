"""
force_control_sim_v2.py — 力控仿真（融合V2力场模型 + 导纳控制）
追踪方式：指令偏移 (dn_cmd, db_cmd)，每步直接设位置
"""
import sys, os
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
sys.path.insert(0, _sdir)

import numpy as np
import matplotlib
if os.environ.get('DISPLAY'): matplotlib.use('TkAgg')
else: matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings("ignore")

from cylinder_def import CylinderDef
from cylinder_geometry_v2 import sample_intersection
from contact_frame_v2 import compute_frame
from force_field_quadratic import calibrate as calib_quad

matplotlib.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class Admittance:
    def __init__(self,M=0.6,D=20.0,dt=0.005):
        self.M,self.D,self.dt=M,D,dt; self.vel=0.0
    def step(self,fe):
        a=(fe-self.D*self.vel)/self.M; self.vel+=a*self.dt
        return np.clip(self.vel*self.dt,-0.8,0.8)

class ForceModel:
    def __init__(self):
        cal=np.load(os.path.join(_sdir,'..','data','force_field_quadratic.npz'))
        self.c_fn=cal['c_dfn']; self.c_fo=cal['c_dfo']
        idxs=cal['base_indices']; fns=cal['base_Fn0']; fos=cal['base_Fo0']
        self._base=[(int(idxs[i]),float(fns[i]),float(fos[i])) for i in range(len(idxs))]
    def base(self,pi):
        return min(self._base,key=lambda x:abs(x[0]-pi))
    def predict(self,dn,db,pi=0):
        p=self.base(pi); c_fn,c_fo=self.c_fn,self.c_fo
        dFn=c_fn[0]*dn+c_fn[1]*db+c_fn[2]*dn**2+c_fn[3]*dn*db+c_fn[4]*db**2
        dFo=c_fo[0]*dn+c_fo[1]*db+c_fo[2]*dn**2+c_fo[3]*dn*db+c_fo[4]*db**2
        return p[1]+dFn, p[2]+dFo

def run(cyl_y,cyl_z,tr=4.0,ns=3000,dt=0.005,df=8.0,pre=0.08,Kf=0.15,label=""):
    ref=sample_intersection(cyl_y,cyl_z,n_samples=ns).sample_pts
    ref=np.vstack([ref,ref[0:1]])
    off=[]
    for P in ref:
        f=compute_frame(P,cyl_y,cyl_z); off.append(P+tr*(-f.normal))
    off=np.array(off)

    adm=Admittance(dt=dt); fm=ForceModel()
    # 指令偏移（在参考位置的局部帧中）
    dn_cmd=pre; db_cmd=0.0
    traj=[]; flog=[]; dnl=[]; dbl=[]

    for step in range(ns):
        i=step%len(off); i=min(i,len(off)-2)
        P_ref=off[i]; P_ideal=ref[i]
        frame=compute_frame(P_ideal,cyl_y,cyl_z)
        ni=-frame.normal; bi=frame.radial_z; bi/=np.linalg.norm(bi) if np.linalg.norm(bi)>1e-12 else 1.0

        # 用指令偏移算力
        Fn,Fo=fm.predict(dn_cmd,db_cmd,i)
        # 导纳更新指令
        ddn=adm.step(df-abs(Fn))
        ddb=-Kf*Fo
        dn_cmd+=ddn; db_cmd+=ddb

        # 位置 = 参考点 + 指令偏移在局部帧下的绝对位移
        pos=P_ref+dn_cmd*ni+db_cmd*bi
        traj.append(pos.copy())
        flog.append(abs(Fn)); dnl.append(dn_cmd); dbl.append(db_cmd)

    traj=np.array(traj); flog=np.array(flog); dnl=np.array(dnl); dbl=np.array(dbl)
    print(f'  [{label}] |Fn|均值={np.mean(flog):.2f}N std={np.std(flog):.2f}N dn_final={dn_cmd:.3f}mm db_final={db_cmd:.3f}mm')
    return traj,flog,dnl,dbl,ref,off

def plot(ref0,off0,traj0,ref1,off1,traj1,f0,f1,dn0,db0,dn1,db1,ta):
    fig=plt.figure(figsize=(18,14))
    ax=fig.add_subplot(221,projection='3d')
    ax.plot(ref0[:,0],ref0[:,1],ref0[:,2],'gray',ls='--',lw=1,alpha=0.5)
    ax.plot(ref1[:,0],ref1[:,1],ref1[:,2],'gray',ls=':',lw=1,alpha=0.5)
    ax.plot(traj0[:,0],traj0[:,1],traj0[:,2],'blue',lw=1.5,label='力控(无误差)')
    ax.plot(traj1[:,0],traj1[:,1],traj1[:,2],'red',lw=1.5,label=f'力控(倾斜{int(ta)}°)')
    ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
    ap=np.vstack([ref0,traj0,ref1,traj1])
    ax.set_box_aspect([ap[:,0].max()-ap[:,0].min(),ap[:,1].max()-ap[:,1].min(),ap[:,2].max()-ap[:,2].min()])
    ax.set_title('3D轨迹');ax.legend(fontsize=7)
    ax2=fig.add_subplot(222);ax2.plot(f0,'b-',lw=0.8);ax2.plot(f1,'r-',lw=0.8)
    ax2.axhline(8.0,color='gray',ls='--',lw=0.5)
    ax2.set_ylabel('|Fn|(N)');ax2.set_title('法向力(目标8N)');ax2.grid(alpha=0.3)
    ax3=fig.add_subplot(223);ax3.plot(dn0,'b-',lw=0.6,alpha=0.7);ax3.plot(dn1,'r-',lw=0.6,alpha=0.7)
    ax3.axhline(0,color='gray',lw=0.3);ax3.set_ylabel('dn(mm)');ax3.set_title('指令法向偏移');ax3.grid(alpha=0.3)
    ax4=fig.add_subplot(224);ax4.plot(db0,'b-',lw=0.6,alpha=0.7);ax4.plot(db1,'r-',lw=0.6,alpha=0.7)
    ax4.axhline(0,color='gray',lw=0.3);ax4.set_ylabel('db(mm)');ax4.set_title('指令复法向偏移');ax4.grid(alpha=0.3)
    fig.suptitle(f'V2力场模型+导纳控制 力控仿真(Z轴倾斜{int(ta)}°)',fontsize=14)
    fig.tight_layout();fig.savefig(os.path.join(_sdir,'output','force_control_sim.png'),dpi=150)
    print('已保存 output/force_control_sim.png');plt.close(fig)

def gen_cyl(tilt=0.0):
    with open(os.path.join(_sdir,'..','data','force_model.pkl'),'rb') as f:d=pickle.load(f)
    cy=d['cyl_contact_y'];cz=d['cyl_contact_z']
    if tilt!=0:
        a=np.radians(tilt);dz=np.array([np.sin(a),0,np.cos(a)])
        ctr=(cz.p1+cz.p2)/2;L=np.linalg.norm(cz.p2-cz.p1)
        cz=CylinderDef(p1=ctr-L/2*dz,p2=ctr+L/2*dz,radius=cz.radius)
    return cy,cz

def main():
    TA=5.0
    print(f"V2力场模型力控仿真(Z轴倾斜{int(TA)}°)");print("="*50)
    print("\n[1] 标定...");calib_quad()
    print("\n[2] 无误差仿真...");cy0,cz0=gen_cyl(0.0);t0,f0,dn0,db0,r0,o0=run(cy0,cz0,label='无误差')
    print(f"\n[3] 倾斜{int(TA)}°仿真...");cy1,cz1=gen_cyl(TA);t1,f1,dn1,db1,r1,o1=run(cy1,cz1,label=f'倾斜{int(TA)}°')
    print("\n[4] 绘图...");plot(r0,o0,t0,r1,o1,t1,f0,f1,dn0,db0,dn1,db1,TA)

if __name__=="__main__":main()
