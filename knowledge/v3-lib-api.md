# V3 力控库（force_feedback_v3/lib）技术文档

> 创建日期：2026-08-04
> 所属项目：机器人末端力控（机器人末端力控 / force_feedback_v3）
> 适用代码：`code/force_feedback_v3/lib/`（V5 力控仿真库）

## 1. 模块总览

`lib/` 是 V5 力控仿真库，共 9 个模块。分层依赖关系：

```
cylinder_def  (圆柱定义数据类)
   ├──→ cylinder_geometry  (双圆柱交线采样)
   │         └──→ contact_frame  (接触点局部标架 {t,n,rz})
   │                   └──→ force_mechanics  (正交力分解 {t,n,o})
   │                             ├──→ simulator  (仿真环境：接触力+摩擦+噪声)
   │                             │       ├──→ controller  (V5 力控控制器 ⭐)
   └──→ sphere_contact  (球刀接触力模型) ──→ simulator
                                              └──→ controller ⭐
                                       force_field_quadratic (二次逆推 ⭐) ──→ controller
                                                      __init__ (包入口+数据加载)
```

**一句话分层：**
- **数据层**：`cylinder_def` — 圆柱怎么定义
- **几何层**：`cylinder_geometry` / `contact_frame` — 交线在哪、接触点的坐标架
- **物理层**：`sphere_contact` — 球刀压进去产生多少力
- **仿真层**：`simulator` — 物理力 + 摩擦 + 噪声 = 仿真测量力
- **控制层**：`force_field_quadratic`（力→偏移逆推）+ `controller`（PID 闭环）⭐ 生产重点
- **入口**：`__init__.py` — 统一导出 + 数据加载

---

## 2. 基础模块

### 2.1 `cylinder_def.py` — 圆柱定义数据类

**功能**：定义圆柱的几何表示。V2 起用「轴线上两点 + 半径」替代旧的「轴向字符 + 轴心点 + 半径」，支持任意轴线方向（不依赖坐标轴对齐）。

**对外开放**：

| 函数/属性 | 输入 | 输出 |
|---|---|---|
| `CylinderDef(p1, p2, radius)` | p1/p2: (3,) 轴线上两点；radius: 半径(mm) | 圆柱对象 |
| `.direction` (属性) | — | (3,) 轴线单位方向 (p1→p2) |
| `.nearest_axis` (属性) | — | 'X'/'Y'/'Z' 最接近的坐标轴 |
| `.axis_point` (属性) | — | (3,) 轴线上一点（兼容旧接口） |
| `from_axis_aligned(axis, point, radius, pts_range=None)` | 轴对齐方式创建 | CylinderDef |
| `from_two_points(p1, p2, radius)` | 任两点+半径 | CylinderDef |

**用例**：由 `load_cylinders()` 间接创建；误差仿真的 `generate_error_cylinders()` 调用 `perturb_endpoints()` 返回新 CylinderDef。

---

### 2.2 `cylinder_geometry.py` — 双圆柱交线几何

**功能**：计算两个任意方向圆柱的相交曲线，并做均匀弧长采样。内部在 `{u1,u2,u3}` 正交基下求解 4 段分支曲线，拼接成闭合点列。

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `sample_intersection(cyl1, cyl2, n_samples=1000, N_curve=250)` | 两个 CylinderDef；n_samples=采样点数；N_curve=每分支离散点数 | `Geom`（含 sample_pts (N,3)、cyl1、cyl2） |
| `resample_curve(geom, n_samples)` | 已有 Geom + 新采样数 | 新 Geom |
| `Geom` (数据类) | — | n_samples / sample_pts (N,3) / cyl1 / cyl2 |

**用例**：`run_sim_error.py` 用它生成误差接触曲线（`sample_intersection(cy_err, cz_err, n_samples=2000)`）；`Simulator.__init__` 内部预计算标准接触曲线。

---

### 2.3 `contact_frame.py` — 接触点局部标架

**功能**：计算接触曲线上一点的局部标架 `{t, n, rz}`。切向量 `t = r_z × r_y`（精确正交两圆柱面），法向量 `n = w_y·r_y + w_z·r_z`（权重 w ∝ r^(2/3) 加权径向组合）。

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `compute_frame(contact_pt, cyl_y, cyl_z)` | (3,) 接触点；两个 CylinderDef | `ContactFrame`(tangent, normal, radial_z) |
| `compute_frames_batch(contact_pts, cyl_y, cyl_z)` | (N,3) 多点 + 圆柱 | dict: tangents/normals/radial_z (N,3) |
| `ContactFrame.as_matrix()` | — | (3,3) 矩阵 [t, n, rz] |

**用例**：`force_mechanics.compute_point_basis_ortho()` 内部调用（所有脚本间接用到）。

---

### 2.4 `force_mechanics.py` — 正交力分解

**功能**：在正交基 `{t, n, o=t×n}` 下分解力向量。因为 o 由 t×n 构造，三向量两两正交，分解退化为点积投影（无需解线性方程组）。

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `compute_point_basis_ortho(P, geom)` | (3,) 点 + Geom | `BasisOrtho`(tangent, normal, ortho) |
| `decompose_force_ortho(F, basis)` | (3,) 力 + BasisOrtho | `ForceDecompOrtho`(coeffs=[Ft,Fn,Fo], 各分量向量, error≈0) |
| `expected_force_ortho(coeffs, basis)` | 系数[a,b,c] + Basis | ForceDecompOrtho（逆构造） |

**用例**：几乎全部脚本——`calibrate_force.py` 算标架、`simulator.py` 算分解、`controller.py` 每步算 Fn/Fo。

---

### 2.5 `sphere_contact.py` — 球刀接触力模型 ⭐

**功能**：核心物理模型。球刀球面用 Fibonacci 均匀采样（N_SPHERE=12800 点），判断每个点是否在双圆柱内部，不在内部的点组成接触斑，`F = K_C·√S`（S=接触面积），方向指向球心。

**关键参数**：`K_C = 6.5225`（标定值）、`R_BALL = 4.2` mm、`N_SPHERE = 12800`。

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `sphere_contact_force(pos, cyl_z, cyl_y)` | (3,) 球心坐标；Z/Y 两个 CylinderDef | `(force_3d, area)`：力向量 (3,) N + 接触面积 mm² |

> ⚠️ 参数顺序注意：第二/第三参是 (cyl_z, cyl_y)，与其它模块的 (cy, cz) 顺序相反。内部 `_inside_cyl()` 用轴线投影法判断，对倾斜误差圆柱也正确（7/29 修复的 bug 点）。

**用例**：`simulator.step()` 每步调用；所有力场可视化脚本采样真实力场。

---

### 2.6 `simulator.py` — 仿真环境

**功能**：封装「接触力 + 库仑摩擦 + 高斯噪声」，模拟真实力传感器测量值。支持传入误差圆柱模拟工件几何误差。

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `Simulator(cy, cz, mu=0.2, sigma=0.5, seed=None)` | 标准圆柱；mu=摩擦系数；sigma=噪声σ | 仿真器 |
| `.step(pos, v_prev, cy_err=None, cz_err=None)` | 球心位置 (3,)；上帧速度 (3,)；误差圆柱可选 | `(F_meas, F_raw, F_fric, F_noise, basis)` 五元组 |
| `.set_friction(mu)` / `.set_noise(sigma)` | 新参数 | — |
| `.load_ball_ref(data_path)` | force_model.pkl 路径 | `(ball_ref (N,3), L 弧长)` |

**用例**：`run_sim_noerror.py`（无误差）、`run_sim_error.py`（传误差圆柱）、`gen_reference.py`（mu=0, sigma=0 参考线）。

---

## 3. 重点模块 ⭐（未来生产环境用）

### 3.1 `force_field_quadratic.py` — 锚定二次逆推 ⭐

**功能**：力→偏移的逆模型。给定测量力 `(Fn, Fo)`，反推球刀相对参考轨迹的偏移量 `(dn, db)`。dn/db 各为 Fn、Fo 的二次函数（6 系数），并做硬锚定约束 `(Fn=-8, Fo=0) → (dn=0, db=0)`——即目标力 8N 时不需要任何补偿。

**模型形式**：
```
dn = c0 + c1·Fn + c2·Fo + c3·Fn² + c4·Fn·Fo + c5·Fo²
db = c0 + c1·Fn + c2·Fo + c3·Fn² + c4·Fn·Fo + c5·Fo²
```
系数由 20 位置 × 21×21 网格（4779 有效点）最小二乘拟合得出。

**拟合精度**（锚定前全局）：
- dn: MAE=0.120mm, RMSE=0.152mm, PC95=0.298mm
- db: MAE=0.195mm, RMSE=0.306mm, PC95=0.680mm
- 强接触区 (|Fn|≥10N): dn 误差 ~0.03mm, db 误差 ~0.04mm
- 弱接触区 (|Fn|<5N): dn 偏小 ~0.2mm（保守退刀），db 偏移 ~0.3mm

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `inverse(Fn_meas, Fo_meas)` | 有符号法向力 Fn (N，负=压入)；有符号复法向力 Fo (N) | `(dn, db)` 偏移量 (mm)，锚定点 (-8,0)→(0,0) |
| `predict(dn, db)` | 偏移 (mm) | 近似 (Fn, Fo)（仅调试用，非逆推链路） |

**边界处理**：
- `Fn ≥ 0`（拉力/无力）→ 截为 0，不反推退刀
- `Fn` 截断到 [-30, 0]、`Fo` 截断到 [-10, 10]（防外推）

**用例**：`controller.py` 每步调用 `inverse(Fn_f, Fo_f)` 得到目标偏移；`_archive/plot_inverse_field.py`、`plot_inverse_residual.py`、`plot_fit_vs_real.py` 验证其精度。

**生产注意**：这是当前控制链路（controller → quadratic inverse）实际使用的逆推模型。若生产部署，建议重点验证弱接触区行为（db 偏移 ~0.3mm 是否可接受），并考虑用实测数据重新拟合系数。

---

### 3.2 `controller.py` — V5 力控控制器 ⭐

**功能**：力控核心闭环。每步：①找最近接触点算标架 → ②力分解得 (Fn, Fo) + 低通滤波 → ③二次逆推得目标偏移 (dn_target, db_target) → ④PID 追逆推归零（n/o 双方向）→ ⑤叠加切向推进速度，输出世界坐标速度。

**控制架构（三步）**：
```
力向量 F → 标架分解 (Fn,Fo) → quadratic.inverse() → (dn_t, db_t)
          → 软限位混合（近参考点用力反馈，偏离大时切位置弹簧）
          → PID n/o 双通道 → v = vn·n + vb·o + v_fwd·t
```

**关键参数（默认）**：
| 参数 | 值 | 含义 |
|---|---|---|
| F_TARGET | -8.0 N | 目标法向力 |
| Kp_n / Ki_n | 25.0 / 0.3 | n 方向 PID |
| Kp_o / Ki_o | 4.0 / 0.025 | o 方向 PID |
| k_pos | 8.0 | 软限位位置弹簧增益 |
| soft_lo / soft_hi | 2.0 / 3.0 mm | 软限位过渡区间（smoothstep 混合） |
| filt_a | 1.0 | 力低通滤波系数（1.0=无滤波） |
| DT | 0.005 s | 仿真步长 |

**对外开放**：

| 类/函数 | 输入 | 输出 |
|---|---|---|
| `ForceController(ball_ref, L, contact_geom, kp_n=25, ki_n=0.3, kp_o=4, ki_o=0.025, ...)` | 球刀参考轨迹 (N,3)；弧长 L；接触曲线 Geom；PID 增益 | 控制器 |
| `.step(F_vec, P_cur, total_steps, dt=0.005)` | 当前力向量 (3,)；球心位置 (3,)；总步数；步长 | `v_3d` (3,) 世界坐标速度 (mm/s) |
| `PID1D(Kp, Ki=0, Kd=0, dt=DT)` | PID 增益 | 一维 PID（输出限幅 ±160） |
| `.step(err)` | 误差 | 速度输出 (mm/s) |
| `LowPass(a=1.0)` | 滤波系数 | 一阶低通 |
| `.update(x)` | 新采样 | 滤波值 |

**核心逻辑细节**（`ForceController.step`）：
1. `_nearest_contact(P_cur)`：接触曲线上最近点 → 标架 `{t, n, o}`
2. 力分解 `Fn = F·n`, `Fo = F·o`，过低通滤波
3. `_nearest_ball_ref(P_cur)`：参考轨迹最近点 → 实际局部偏移 `dn_actual`, `do_actual`
4. `inverse(Fn_f, Fo_f)` → `(dn_target, db_target)`
5. `_blend()`：softmax 平滑混合——偏差 < soft_lo 用力反馈（`-dn_target`），> soft_hi 用位置弹簧（`-k_pos·dn_actual`），中间 smoothstep 过渡
6. PID → `vn`, `vb`；切向速度 `v_fwd = L / (total_steps·dt)` 恒定推进
7. 合成 `v_3d = vn·n + vb·o + v_fwd·t`

**用例**：`gen_reference.py`（参考轨迹）、`run_sim_noerror.py`（无误差仿真）、`run_sim_error.py`（误差仿真）。

**生产注意**：控制器本身不依赖误差圆柱，用标准圆柱几何算标架，靠力反馈自适应误差——这正是生产环境的用法。部署时关注：①n/o 方向增益需要按实际力传感器采样率重新整定；②软限位区间按执行器行程设定；③filt_a 用于力信号平滑，实际传感器有噪声时可降到 0.5~0.8。

---

## 4. 包入口 `__init__.py`

**功能**：统一导出 + 数据加载。

**对外开放**：

| 函数 | 输入 | 输出 |
|---|---|---|
| `load_cylinders(data_dir=None)` | 数据目录（默认 force_model.pkl 所在） | `(cy_contact, cz_contact)` 标准圆柱 |
| `load_ball_ref(data_dir=None)` | 同上 | `(ball_ref (N,3), L 弧长)` |
| `perturb_cylinder(cz, dx=0, dy=0, dz=0)` | 平移量 | 新 CylinderDef（兼容旧接口） |
| `perturb_endpoints(cyl, dp1, dp2)` | 两端点偏移 (3,) | 新 CylinderDef（轴线和位置都变） |
| `generate_error_cylinders(cy, cz, rng)` | 标准圆柱 + RandomState | `(cz_err, cy_err)` 12 参数 ±1mm 随机误差圆柱 |

**用例**：所有 script 开头 `from force_feedback_v3.lib import load_cylinders, load_ball_ref`。

---

## 5. 模块 → 用例映射速查

| 模块 | 正式用例 | 归档用例 |
|---|---|---|
| cylinder_def | 所有模块间接使用 | — |
| cylinder_geometry | `run_sim_error.py` | plot_error_* / plot_four_points / plot_std_force_field |
| contact_frame | 经 force_mechanics 间接使用 | — |
| force_mechanics | `calibrate_force.py` / `run_sim_*.py` | 绝大多数 plot_* |
| sphere_contact | `calibrate_force.py` / `run_sim_*.py` | 力场类 plot_* |
| simulator | `gen_reference.py` / `run_sim_*.py` | plot_fit_vs_real 等 |
| force_field_quadratic ⭐ | **controller.py** / `calibrate_force.py` | plot_inverse_* / plot_fit_vs_real |
| controller ⭐ | `gen_reference.py` / `run_sim_noerror.py` / `run_sim_error.py` | — |
| __init__ | 所有 script | — |
