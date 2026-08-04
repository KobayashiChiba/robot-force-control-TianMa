# 机器人末端力控项目 — 工作日报

**日期**：2026-07-03
**状态**：🆕 项目启动

---

## 项目摘要

针对工业机器人末端力控应用中的圆柱孔相交边缘打磨路径规划问题，建立两正交圆柱交线的通用几何模型，并计算沿交线各点的打磨法向量，为机器人末端执行器的姿态规划和力控策略提供几何基础。

---

## 今日完成

### 1. 文献调研任务创建
- 在飞书「科研任务」清单中创建"调研工业机器人末端力控相关文献"任务（截止 7月5日）

### 2. 正交圆柱交线几何建模
- 完成两个正交圆柱相交曲线的通用数学推导
- 支持任意轴向组合（X/Y/Z），自动识别公共径向坐标
- 四段半曲线独立求解，自然汇合形成闭合空间曲线
- ⚠️ 当前数字为测试数据（单位mm），后续需金属零件详细参数替换
- 理想情况：如有实际测量的曲线数据，可用于验证模型精度

### 3. 打磨法向量计算
- 基于两个圆柱面径向法向量的角平分线方向
- 物理场景：圆柱内部为空气（孔洞），外部为金属，法向量指向金属区域
- ✅ 方案已定，无需补充

### 4. 可视化代码实现
- 核心代码：`code/test.py`
- 依赖：numpy, matplotlib
- 可配置参数（半径、位置、轴线方向）集中于文件顶部
- 输出 3D 可视化图像（含圆柱面、交线、法向量箭头、轴线）

### 5. 实际测量数据可视化（7/3 下午）
- 收到实际测量数据（81个采样点）：球刀中心点坐标 + 四元数姿态 + 接触点坐标
- 脚本：`code/plot_tool_path.py`
- 绘制内容包括：
  - 球心轨迹（彩色渐变闭合曲线）
  - 工具姿态坐标架（每8个点一个RGB三轴箭头，四元数→旋转矩阵）
  - 接触点轨迹（灰色虚线）
  - 球心→接触点连线（橙色，示意工具轴方向）
- 输出图：`code/tool_path_with_orientation.png`
- 已插入飞书工作日报文档

---

## 技术方案

### 误差计算

需要先明确两个圆柱轴线的**允许误差范围**（位置偏差几mm？角度偏差多少？），然后：

1. 在误差范围内取一组误差值
2. 计算包含误差的"真实"交线曲线
3. 与标准（无误差）曲线在同一图中对比
4. 同时计算包含误差后的法向量

### 误差修正算法

- 基础算法：**阻抗控制**
- 核心思路：根据当前实际力测量值与预期值的差异，调整位置环的位置控制
- 控制器：先用 **PID 控制**，使末端贴合真实曲线

### 实验目标

在软件仿真中验证算法有效性：
- 在误差范围内生成随机误差曲线
- 运行一遍修正算法
- 若能对误差范围内的任意曲线都能较好贴合 → 算法有效

---

## ⚠️ 需要注意的问题

### 力传感器分辨率问题

| 参数 | 值 | 分析 |
|------|-----|------|
| 平均测量力 | ~8N | |
| 传感器分辨率 | 1N | 相对误差约 **12.5%** |
| 潜在风险 | PID震荡 | 力反馈步进变化（1N步长）可能导致位置环过度响应 |

建议：
1. **力信号预处理** — 加入低通滤波或滑动窗口平均，平滑阶梯状力信号
2. **PID参数整定** — 建议先仿真跑一遍，观察是否出现极限环振荡
3. **考虑变增益** — 误差大时用大增益快速接近，误差小时用小增益避免震荡

---

## 项目文件结构

### 本地
```
📁 projects/formal/机器人末端力控/
├── 📄 project.md                    ← 项目概述
├── 📄 progress.md                   ← 工作记录（本文件）
├── 📄 TECHNICAL_REFERENCE.md        ← 技术参考手册 v1.0
├── 📄 正交圆柱相交曲线分析.md        ← 完整数学建模文档
├── 📁 code/
│   ├── 📁 lib/                      ← 功能模块
│   │   ├── cylinder_geometry.py     ← Layer 1：几何计算
│   │   ├── force_mechanics.py       ← Layer 2：力学计算
│   │   └── cylinder_fitting.py      ← Layer 0：散点拟合
│   ├── 📁 scripts/                  ← 测试/绘图脚本
│   │   ├── test_fitting.py          ← 实测数据验证
│   │   ├── test_fitting_ball.py     ← 球刀中心 vs 接触点对比
│   │   ├── test.py                  ← 理论曲线可视化
│   │   ├── plot_tool_path.py        ← 实测轨迹可视化
│   │   └── ...
│   ├── 📁 output/                   ← 生成的图
│   │   └── *.png
│   ├── 📁 data/                     ← 原始数据
│   │   └── 球刀中心点及轮廓轨迹点.xlsx
│   └── 📁 _archive/                 ← 旧版备份
│       └── fit_cylinders.py / v2
└── make_*.py                        ← docx 生成辅助脚本
```
### 飞书云盘
```
📁 工作项目 > 去毛刺机器人末端力控/
├── 📋 项目概述                    ← 飞书原生文档
├── 📋 2026-07-03_工作日报（含图） ← 飞书原生文档
└── 📋 正交圆柱相交曲线分析 & ...  ← 飞书原生文档
```
```

---

## 下一步计划

（待补充具体排期）

---

## 2026-07-06 更新：第一周周报

### 创建周报
- 基于 7/3 日报改写成第一周周报（07/03~07/06）
- 去掉文献调研任务创建、飞书文档创建等描述
- 加入双视角工具路径图 + 正交圆柱相交曲线图
- 去掉项目文件结构部分
- 润色语言，统一章节格式

### 飞书文档操作
- ✅ 用 `lark-cli docs +create --doc-format markdown` 创建周报文档
- ✅ 用 `docs +media-insert` 插入两张图到文档末尾
- ✅ 用 `block_move_after` 将图片移动到对应章节（第4节、第5节末尾）

---

## 2026-07-07：圆柱拟合 + 误差分析

### 📊 圆柱拟合
基于81个实测接触点，用最小二乘法拟合两个正交圆柱的参数：

**Y方向圆柱（轴∥Y）：**
- 轴线 X₀ = 51.497 mm，Z₀ = -39.700 mm
- 半径 r₁ = 9.002 mm
- RMS残差 = 0.003 mm ✅

**Z方向圆柱（轴∥Z）：**
- 轴线 X₀ = 72.499 mm，Y₀ = 65.151 mm
- 半径 r₂ = 17.998 mm
- RMS残差 = 0.085 mm ✅

### 📈 可视化输出（4张图）
- `code/fig1_points_and_cylinders.png` — 测量点 + 两个拟合圆柱 3D
- `code/fig2_measured_vs_curve.png` — 测量点 vs 拟合交线对比
- `code/fig3_xz_projection.png` — Y投影 (XZ平面，Y圆柱截面)
- `code/fig4_xy_projection.png` — Z投影 (XY平面，Z圆柱截面)
- `code/fig_collage_2x2.png` — 4图拼合（用于日报）

### 🧹 文件清理
- 删除无用文件：`feishu_auth_qr.png`, `progress.docx`, 旧图等

### 📝 待办
- [ ] 误差修正算法仿真（阻抗控制 + PID）
- ✅ 用 `drive +move` 将周报移至云盘「工作项目 > 去毛刺机器人末端力控」

---

## 2026-07-09：代码模块化重构 — Layer 1 + Layer 2

### 📦 新增模块化代码（取代 test.py 中的内联实现）

**Layer 1 — `code/cylinder_geometry.py`**
- `sample_intersection()`：双圆柱相交曲线均匀弧长采样，对外单一接口
- `resample_curve()`：从已有 Geom 重采样
- 数据类 `Geom`：仅存采样结果 + 圆柱参数，无隐藏字段
- 内部 4 段分支求解 → 拼接闭合点列 → 折线均匀弧长重采样

**Layer 2 — `code/force_mechanics.py`**
- `compute_point_basis(P, geom)`：自动定位分支，计算基底 {t, n, v}
- `decompose_force(F, basis)`：非正交基下严格力分解（解 3×3 线性方程组）
- `expected_force(coeffs, basis)`：逆运算，从系数合成力向量
- `compute_normal_motion_trend()` / `compute_vertical_motion_trend()`：运动趋势向量
- 依赖 `cylinder_geometry` 的 `Geom` 和 `_get_branch_meta`

### 📘 技术文档
- `TECHNICAL_REFERENCE.md` v1.0 — 完整技术参考手册
  - 架构概览（Layer 1 → Layer 2 依赖关系）
  - 全部数据类定义（Geom / Basis / ForceDecomp）
  - 各函数数学推导、参数表、使用示例
  - 数学附录：弧长重参数化 + 非正交基可逆性证明
  - 术语表

### 🧪 验证
- [ ] 用 `code/test.py` 的旧数据跑通新模块
- [ ] 验证力分解重建误差 ≈ 0

---

## 2026-07-09（续）：新增 cylinder_fitting.py — Layer 0

### 📦 `code/cylinder_fitting.py`
- `fit_cylinders_from_points(pts, axis1, axis2)` — 主函数
  - 返回 `(list[CylinderParams], Geom)`
  - CylinderParams 存完整精度：axis / axis_point / radius / rms / max_err / residuals
  - Geom 存 1 位小数：n_samples=0，可直接传入 `resample_curve()`
- `_fit_projection(pts, axis)` — 内部：投影圆拟合（3 参数，线性最小二乘）
- **不保留 V2 的 3D 非线性拟合** — 圆柱无限延伸，空间距离无意义

### 🧪 验证
- ✅ 理论数据拟合：r1=10.0000, r2=20.0000, RMS≈0
- ✅ Geom → resample_curve 链路正常
- ✅ 实测数据验证（81 个接触点）：r1=9.002, r2=18.003, RMS<0.004mm
- ✅ 球刀中心 Z 轴偏移发现：原始 Z 偏低 4.815mm，修正后距离稳定在 3.81±0.04mm（球刀半径）

---

## 2026-07-09（续）：力分解方案对比 + 日报

### 🎨 力分解可视化
- 生成两种力分解方案对比图（Frenet正交 vs 当前非正交）
- 生成力分解示意图 + 运动趋势示意图
- 拼合输出：`output/fig_schemes_compare.png` / `output/fig_force_analysis.png`

### 📝 日报
- 文档化 `force_mechanics.py` 核心算法（`compute_point_basis` / `decompose_force` / 运动趋势）
- 分析非正交分解方案优势：
  - Ft（切向）→ 毛刺+震动，可忽略
  - Fn（法向）→ 切入深度表征，控制打磨质量
  - Fv（垂直/Z柱面法向）→ 圆柱壁接触检测，防止过切
- 日报已上传飞书：`2026-07-09 工作日报（力分解与运动趋势）`

---

## 🔍 重要发现：球刀中心 Z 轴系统偏移

### 现象
球刀中心点到接触曲线距离标准差大（~2mm），两端 7.8mm、中间 2.8mm。

### 根因
- 球刀中心 Z 坐标整体偏低 **4.815mm**（均值：-44.52 vs 接触曲线均值 -39.70）
- 修正后距离稳定在 **3.81 ± 0.04 mm**（即球刀半径 ≈3.8mm）

### 修正数据
- 修正后 Excel：`data/球刀中心点_修正后.xlsx`（新增 X_shifted/Y_shifted/Z_shifted 列）

---

## 2026-07-20：标准曲线定义 + 接触标架接口

### 📐 标准曲线（`data/standard_curves.pkl`）
- **球刀中心标准曲线**：Z修正+4.815mm 后拟合，Y圆柱 r=7.3, Z圆柱 r=15.8, 500点
- **标准接触曲线**：Y圆柱 r=9.0, Z圆柱 r=18.0, 500点
- **球刀半径**：4.0mm（实际球刀向里切进，非完全相切）

### 📊 法向量计算方法
- 两簇交点法：球刀中心 P → 接触曲线上距离≈R 的点分两簇 → 质心中点方向
- 发现：**法向量 ≈ P 到接触曲线最近点的方向**（夹角 <2°）
- 更优方法：**加权径向组合** `n = w_y·r_y + w_z·r_z`，权重 `w ∝ r^(2/3)`
  - 与真实法向量偏差仅 **1.9° ± 1.0°**（vs 角平分线 20.8°）

### 🧭 切向量
- `t = r_y × r_z`（精确，完全正交于两个圆柱面法向量）

### 💻 接口（`code/lib/contact_frame.py`）
- `compute_frame(C, cyl_y, cyl_z, r_y, r_z)` → `ContactFrame(tangent, normal, lateral)`
- `compute_frames_batch(pts, ...)` → 批量计算
- 三个向量完全正交、单位长度、右手标架
- 法向量用 r^(2/3) 加权，无需球刀中心曲线

### 📁 保存的数据文件
- `data/standard_curves.pkl` — 标准曲线（Geom × 2）
- `data/normals.pkl` / `data/normals.npz` — 500点法向量+两簇质心
- `data/ball_radius_results.txt` — 81点球刀半径计算结果

### ⏳ 待办
- [ ] 误差修正算法仿真（阻抗控制 + PID）

---

## 2026-07-20（续）：V2 标准库重构

### 🎯 目标
将圆柱表示从 `(axis_char, axis_point, radius)` 升级为 `CylinderDef(p1, p2, radius)`，支持任意轴线方向，为轴线误差分析做准备。

### 📦 新增 V2 模块（`code/lib_v2/`，共 5 个）

| 模块 | 层次 | 说明 |
|------|:--:|------|
| `cylinder_def.py` | 公共 | `CylinderDef(p1, p2, radius)` 数据类 |
| `cylinder_fitting_v2.py` | L0 | 投影圆拟合，输出 CylinderDef |
| `cylinder_geometry_v2.py` | L1 | 任意方向圆柱交线，`{u1,u2,u3}`基变换法 |
| `contact_frame_v2.py` | 标架 | `{t, n, rz}`，加权法向量 r^(2/3) |
| `force_mechanics_v2.py` | L2 | 力分解 `{t, n, rz}` 非正交基底 |

### 🔑 关键技术变更

| | V1 | V2 |
|---|---|---|
| 圆柱定义 | `(axis_char, axis_point, radius)` | `CylinderDef(p1, p2, radius)` |
| 轴线 | 固定坐标轴 X/Y/Z | 任意方向向量 |
| 交线算法 | 坐标索引投影 | `{u1,u2,u3}`基变换 |
| 切向量 | 解析（需分支定位） | `t = r_y x r_z`（通用） |
| 法向量 | `n1+n2` 角平分线 | `w_y*r_y + w_z*r_z` r^(2/3)加权 |

### 📐 V2 标准曲线（`data/standard_curves_v2.pkl`）

| | Y圆柱 | Z圆柱 | RMS |
|---|---|---|---|
| 球刀中心 | r=7.35mm | r=15.77mm | 0.07/0.02mm |
| 接触曲线 | r=9.00mm | r=18.00mm | 0.003/0.003mm |
| 球刀半径 | — | — | **4.0mm** |

- 接触曲线两圆柱轴线对称延长至 40mm
- 交线 500 点，闭合差 0.0000mm

### ✅ 验证结果
- L0+L1+L2 全链路通过
- V2 vs 旧版交线偏差：**0.0015mm**（浮点精度）
- 力分解重建误差：**3.97e-15**
- `|t*n|`、`|t*rz|`：**5.55e-17**（t 严格正交于 n 和 rz）

### 🧹 目录整理
```
code/
├── lib/          ← V1 旧库（保留）
├── lib_v2/       ← V2 标准库（5个模块，纯净）
├── scripts_v2/   ← 测试和流程脚本（6个）
└── data/
    └── standard_curves_v2.pkl  ← V2 唯一标准曲线
```
- 删除了旧数据文件 `normals.pkl`、`standard_curves.pkl`
- 删除了 `cylinder_fitting_v2.py` 中的过渡函数 `make_geom()`

### 📝 文档
- `V2-API.md` — 完整 API 文档（5个模块 + 标准曲线 + 使用示例）

### ⏳ 待办
- [ ] 阻抗力控算法设计

---

## 2026-07-24：力模型 + GitHub + 可视化

### GitHub 仓库
- ✅ 创建 GitHub 仓库 `robot-force-control-TianMa`，推送全部代码

### V2 代码库
- ✅ `force_mechanics_v2.py` 新增正交基底 `{t, n, t×n}`
- ✅ `cylinder_geometry_v2.py` 修复 `_sample_uniform`：argmin → np.interp

### 数据处理
- ✅ 接触曲线：V2圆柱参数 → 2000点均匀弧长，间距38μm
- ✅ 球刀中心：5阶傅里叶拟合替代圆柱拟合，500点，误差13μm
- ✅ `data/force_model.pkl` 保存完整模型参数

### 力模型
- ✅ 接触力 `F = k_c × √S`，R=4.2mm，k_c=7.37，均值≈8N
- ✅ 摩擦力 μ=0.2，切向<2N；噪声 σ=0.5N
- ✅ 正交分解 {t, n, t×n} → Ft/Fn/Fo

### 可视化脚本
- ✅ `draw_section.py` / `section_gallery.py` / `section_with_ball.py` / `force_profile.py` / `p0_force_field.py`

### 误差实验
- ✅ 不同方向轴线偏移产生可区分的力曲线指纹

---

## 2026-07-27 V5 力控仿真完成

### 架构
- PID 直接追逆推值归零 → 力级闭环 (`vn = pid(inverse(Fn))`)
- `_nearest_contact(pos)` 替代 `contact_at(s)` → 标架跟球刀走
- 一次物理模型: `Fn = kn·dn + F_TARGET`, `Fo = ko·dn·db` (kn=-24.5, ko=4.95)
- 逆推: `dn = (F_TARGET - Fn) / KN`（解析解）
- n/o 双方向硬限位 + anti-windup
- 无力时搜索模式：dn_target=0.5mm

### 结果
| 工况 | \|F\| | std | 限位触发 |
|:--|:--|:--|:--|
| 无误差 | 8.00N | 0.08N | 0 |
| X+1.5mm | 9.20N | 0.33N | 0 |
| 随机±0.5mm (10组) | 9.1~9.3N | ~0.1N | 0 |

### 与 V2 对比
- V2: 8.05±0.40N（模型力开环 + 参考轨迹锚定）
- V5: 8.00±0.08N（真实力闭环 + 速度积分）
- V5 无误差精度超越 V2

### 关键教训
- 逆推值直接驱动 PID（不追 dn_target-dn_actual）→ 力级闭环
- 标架在最近接触点计算（不按弧长取）→ Fn 方向正确
- 常数项 = F_target 嵌入逆推 → 打破自洽
- X-1.5mm 极端情况球刀穿透空腔碰对面壁 → 搁置

### 代码
- `code/sim/force_control_sim_v5.py` — 控制器
- `code/lib_v2/force_field_fixed.py` — 力场一次模型
- `code/sim/run_sim_v5.py` — 运行脚本
- 已推 GitHub: `9c15421`

---

## 2026-07-27（续）：V5 db 反馈（Fo→o方向）

### 改动
- `force_field_fixed.py` inverse 新增 o 方向逆推：`db = -Fo / KO`，KO=1.4
- 控制器 `force_control_sim_v5.py` o 方向 PID 改为 `pid_o.step(db_target - do_actual)`
- o-PID 增益降至 Kp=8.0, Ki=0.05（避免干扰 n 方向）

### 迭代过程
- 初版方案B乘积模型 `db = Fo/(ko·dn)`：dn≈0.05mm 处分母病态 + 符号反了（正反馈），导致 8.00→9.02N
- 改加性模型 `db = α·dn - Fo/KO`：α=-2.0 稍改善，但不明显
- 最终 `db = -Fo/KO`（纯负反馈）：最简单有效

### 结果
| 工况 | 改前 | 改后 |
|:--|:--|:--|
| 无误差 | 8.00±0.08 | **7.91±0.09** |
| X+1.5mm | 9.20±0.33 | **8.96±0.17** |
| 随机±0.5mm (10组) | 9.1~9.3N | **7.47~8.44N** |

### 脚本
- `code/sim/run_sim_v5_batch.py` — 批量误差测试（11组）
- `code/sim/run_sim_v5_rand.py` — 随机误差 3D 图（10组，含标准/误差接触曲线+球刀参考）

---

## 2026-08-04：physical_v2 逆推模型修复 + 生产对接准备

### 背景
- 从 `_archive/` 找回 `force_field_physical_v2.py`（物理分块逆推 V2，2026-07-29 拟合，K_C=6.5192 锚定）
- 放回 `lib/`，controller 切换测试（`from .force_field_physical_v2 import inverse`）
- 目的是为生产对接方案 B（仿真估计+实机微调）验证：方案 B 初始算式系数 = physical_v2 系数

### 🔴 发现截断 Bug（dn<=0 返回 0,0）
- 症状：有误差 seed42 Fn 掉到 **-4.15±3.22N**，力控失效
- 根因：`inverse()` 里 `dn = (|Fn|-8)/9.917`，力不足 8N 时 dn 为负，被 `if dn <= 0: return 0.0,0.0` 截断 → 控制器 err_n=0 → PID 不动 → 球刀停在弱力区
- 对比 quadratic 无此截断：力不足时返回负 dn → err_n 正 → 推入
- 与 7/29 文档"有误差均值 3-6N、弱力步>50%"吻合——根因就是截断

### ✅ 修复
- 去掉截断，力不足时保留负 dn 让控制器推入（浅接触不反推 db）

### 🔴 逆推不连续分析（抖动来源）
修复后 Fn=-8.02±0.56N，但 limit>1mm 70 次。三个跳变点：
1. 零接触边界 |Fn|=0.5N：dn 从 -0.756 突跳到 0
2. 8N 边界 dn=0：db 从 0 突跳到 ±0.7
3. Fo=0 系数切换：右上/右下段 k1/k3 不同 → db 跳变

### ✅ fix3 方案定型（小林确认保留）
三个修复方向对比（`script/run_sim_physical_v2_variants.py`）：
| 版本 | Fn | std |
|------|-----|-----|
| 基线（去截断） | -8.02 | ±0.57 |
| fix1 零接触去截断 | -8.02 | ±0.54 |
| fix2 +db 平滑 | -8.01 | ±0.54 |
| fix3 +Fo 连续 | -8.00 | ±0.52 |

fix3 = 零接触连续 + dn∈[0,DN_SAT] db smoothstep + Fo∈[-FO_SAT,+FO_SAT] 系数平滑混合（DN_SAT=0.3, FO_SAT=0.3）
- 完整有误差仿真：Fn=-8.01±0.54N，limit>1mm 8 次
- 已写入 `lib/force_field_physical_v2.py`（2026-08-04 fix3 注释标注）
- 自测往返：dn median=0.215mm，db median=0.414mm

### ⚠️ Simulator 随机性
- `Simulator` 默认 `seed=None` → 每次跑噪声不同 → limit>1mm 统计不可复现（同代码跑出 8/63/132）
- Fn 均值/方差稳定（-8.01±0.53~0.56）→ 力控收敛真实可靠
- 小林决定：不管，看图判断效果

### 📊 力场对比图（output_V2/）
- `script/plot_fit_v2_vs_real.py` → `output_V2/fit_v2_vs_real.png`
- 左1列 = physical_v2 predict 简化公式拟合热力图（|F|/Fn/Fo）
- 右6列 = 真实 sphere_contact_force 实测热力图 + 截面叠加（p=0~0.25）

### ⚠️ 当前状态提醒
- `lib/controller.py` import 已切到 `force_field_physical_v2`（原 quadratic 保留在 lib 可切回）
- 新脚本 `run_sim_physical_v2_variants.py`、`plot_fit_v2_vs_real.py` 待小林确认后再正式归档
- `output_V2/` 新目录（与 output/ 平级）
- git 有未提交改动：lib/controller.py、lib/force_field_physical_v2.py、新增 script 2个、output_V2/
- 工具脚本 feishu_send_tool.py 从 projects.zip 恢复到 hermes/scripts/（之前清理项目时误删）
