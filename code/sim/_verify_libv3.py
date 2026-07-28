"""验证 lib_v3 端到端"""
import sys, os, time, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
sys.path.insert(0, os.path.join(_sdir, '..'))  # code/ for lib_v3 import
from lib_v3 import Simulator, ForceController, load_cylinders, load_ball_ref, perturb_cylinder

DT = 0.005; N = 3000
cy, cz = load_cylinders(os.path.join(_sdir, '..', 'data'))
ball_ref, L = load_ball_ref(os.path.join(_sdir, '..', 'data'))

# 创建仿真器和控制器
sim = Simulator(cy, cz, mu=0.2, sigma=0.5, seed=42)
ctrl = ForceController(ball_ref, L, sim.contact_geom)

pos = ball_ref[0]
flog = []
v_prev = np.zeros(3)
t0 = time.perf_counter()

for step in range(N):
    F_meas, F_raw, F_fric, F_noise, basis = sim.step(pos, v_prev)
    v_3d = ctrl.step(F_meas, pos, N, DT)
    pos += v_3d * DT
    v_prev = v_3d.copy()
    flog.append(np.linalg.norm(F_meas))

flog = np.array(flog)
elapsed = time.perf_counter() - t0
last500 = flog[-500:]
gap = np.linalg.norm(pos - ball_ref[0])
print(f"|F| = {np.mean(last500):.2f} +/- {np.std(last500):.2f} N  ({elapsed:.1f}s)")
print(f"首尾距离 = {gap:.3f} mm  (弧长 {L:.1f} mm)")
print("lib_v3 ✅")
