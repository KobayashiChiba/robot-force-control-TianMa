# 会话交接文档 — 机器人末端力控 V2

> 日期：2026-07-27（周一上午）  
> 会话主题：力场建模 + 力控仿真闭环

---

## 本次会话完成的工作

### 1. 代码归档
- 从 7/24 session 历史捞回 `error_experiment.py` 和 `error_fingerprints.py`（误差实验脚本）
- 归档后均跑通验证，推 GitHub

### 2. 力场热力图系列
- 创建 `p_multi_force_field.py`：多点力场热力图 + 截面叠加
  - 5 个进度位置 (p=0.05~0.25)，每个位置 3 面板 (|F|/Fn/Fo)
  - 设计要义写在文档注释中

### 3. 力场模型库（lib_v2/）
- 创建 `force_field_quadratic.py`：二次多项式方法
  - 20 位置 4778 采样点统一拟合
  - 接口：`calibrate()` / `predict(dn,db)` / `inverse(Fn,Fo,p_idx)`
  - 逆推误差中位 0.11mm
- 创建 `force_field_physical.py`：物理直觉模型
  - 3 参数：Fn = min(0, -k·dn - c), Fo ∝ dn·db（右上/右下独立系数）
  - 接口：`calibrate()` / `predict(dn,db)` / `inverse(Fn,Fo)`
  - 浅接触区稳定性较差，建议用二次模型

### 4. 力控仿真闭环（核心产出）
- 创建 `force_control_sim_v2.py`
  - **信号链路**：球心位置 → 力场模型 → F_vec(3D) → 标架分解 → (Fn,Fo) → 导纳控制 → 指令偏移修正 → 新位置
  - **追踪方式**：指令偏移 (dn_cmd, db_cmd) 标量状态，每步设位置 = P_ref + dn*ni + db*bi
  - 使用真实圆柱数据（从 force_model.pkl 加载）
  - **仿真参数**：导纳 D=20 M=0.6，目标力 8N，预压 0.08mm
  - **结果**：无误差和倾斜 5° 均收敛到 8.05N ± 0.40N
  - 3D 图三轴等比例

---

## 项目当前状态

### 代码结构
```
code/lib_v2/（功能模块 — 7 个）
├── cylinder_def.py              # L0: 数据类
├── cylinder_fitting_v2.py       # L0: 圆柱拟合
├── cylinder_geometry_v2.py      # L1: 交线几何
├── contact_frame_v2.py          # 接触标架 {t, n, radial_z}
├── force_mechanics_v2.py        # L2: 力分解
├── force_field_quadratic.py     # 🆕 力场二次模型
└── force_field_physical.py      # 🆕 力场物理模型

code/sim/（脚本 — 10 个）
├── force_profile.py             # 力剖面（接触力+摩擦+噪声）
├── p0_force_field.py            # p=0 力场热力图（接触点原点）
├── p_multi_force_field.py       # 🆕 多点力场 + 截面叠加
├── error_experiment.py          # 🆕 轴线偏移对力影响
├── error_fingerprints.py        # 🆕 多方向力指纹
├── calibrate_force_field.py     # 🆕 力场标定
├── force_control_sim_v2.py      # 🆕 力控仿真闭环
├── section_with_ball.py         # 截面+球刀
├── section_gallery.py           # 截面画廊
└── draw_section.py              # 单截面

code/data/
├── force_model.pkl              # 核心数据（接触/球心曲线+圆柱参数）
├── force_field_quadratic.npz    # 二次模型标定
├── force_field_physical.npz     # 物理模型标定
└── force_field_calib.npz        # 标定中间结果
```

### 待办
- [x] 力场建模 ✅
- [x] 力控仿真闭环 ✅
- [ ] 和上午发的原始脚本(python -c 版本)对比验证
- [ ] 参数调优（导纳增益、目标力等）
- [ ] 增加实际传感器噪声模型
- [ ] open claw 文献调研

### 关键教训
1. **坐标追踪**：不要在世界坐标下累积 offset，要追踪标量指令 (dn_cmd, db_cmd)
2. **力场基线变化**：Fn0(p) 沿曲线变化（6.5~8.2N），控制目标应针对 dn 偏移量
3. **p0_force_field.py**：是接触点原点的旧版，标准球心原点版本在 p_multi_force_field.py 中
4. **两个力场模型**：二次精度高但需 Newton 迭代，物理模型简单但浅接触区不稳定

### GitHub
https://github.com/KobayashiChiba/robot-force-control-TianMa
