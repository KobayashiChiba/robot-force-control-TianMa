"""
对比: 无扰动 / 只摩擦 / 只噪声 / 全开 (lib_v3)
"""
import sys, os, numpy as np
_sdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_sdir, '..', 'lib_v2'))
sys.path.insert(0, os.path.join(_sdir, '..'))
from lib_v3 import Simulator, ForceController, load_cylinders, load_ball_ref

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

DT = 0.005; N_STEPS = 3000
cy, cz = load_cylinders(os.path.join(_sdir, '..', 'data'))
ball_ref, L = load_ball_ref(os.path.join(_sdir, '..', 'data'))


def run(sim, mu, sigma, seed):
    sim.set_friction(mu)
    sim.set_noise(sigma)
    sim.reset_rng(seed)
    ctrl = ForceController(ball_ref, L, sim.contact_geom)
    pos = ball_ref[0]
    flog = []
    v_prev = np.zeros(3)
    for step in range(N_STEPS):
        F_meas, _, _, _, _ = sim.step(pos, v_prev)
        v_3d = ctrl.step(F_meas, pos, N_STEPS, DT)
        pos += v_3d * DT
        v_prev = v_3d.copy()
        flog.append(np.linalg.norm(F_meas))
    flog = np.array(flog)
    last500 = flog[-500:]
    print(f"  [{mu=},{sigma=}] |F|={np.mean(last500):.2f}+/-{np.std(last500):.2f}N")
    return flog


print("V5 扰动分离对比 (lib_v3)")
print("=" * 50)
sim = Simulator(cy, cz)
f_clean = run(sim, 0.0, 0.0, 42)
f_fric  = run(sim, 0.2, 0.0, 42)
f_noise = run(sim, 0.0, 0.5, 42)
f_both  = run(sim, 0.2, 0.5, 42)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (flog, title, color) in zip(axes.flat, [
    (f_clean, "无扰动", "blue"),
    (f_fric,  "只摩擦 (μ=0.2)", "orange"),
    (f_noise, "只噪声 (σ=0.5)", "green"),
    (f_both,  "全开", "red"),
]):
    last500 = flog[-500:]
    ax.plot(flog, color, lw=0.5)
    ax.axhline(8.0, color='gray', ls='--', lw=0.5)
    ax.set_title(f'{title}\n{np.mean(last500):.2f}±{np.std(last500):.2f}N')
    ax.set_ylabel('|F| (N)'); ax.set_xlabel('Step'); ax.grid(alpha=0.3)

fig.suptitle('V5 扰动分离 (lib_v3)', fontsize=14)
fig.tight_layout()
out = os.path.join(_sdir, 'output', 'v5_disturbance_compare.png')
fig.savefig(out, dpi=150)
print(f'\n已保存 {out}')
plt.close(fig)
