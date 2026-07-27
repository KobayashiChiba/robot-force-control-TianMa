- ✅ 正交分解 {t, n, t×n} → Ft/Fn/Fo

---

## 2026-07-27 V5 力控仿真完成

### 架构
- PID 直接追逆推值归零 → 力级闭环 (`vn = pid(inverse(Fn))`)
- `_nearest_contact(pos)` 替代 `contact_at(s)` → 标架跟球刀走
- 一次物理模型: `Fn = kn·dn + F_TARGET`, `Fo = ko·dn·db` (kn=-24.5, ko=4.95)
- 逆推: `dn = (F_TARGET - Fn) / KN`（解析解，永不炸）
- n/o 双方向硬限位 + anti-windup
- 正交标架 {t, n, o复法向}

### 结果
| 工况 | \|F\| | dn | do | 限位触发 |
|:--|:--|:--|:--|:--|
| 无误差 | **8.00±0.08N** | 0.01mm | 0mm | 0 |
| X+1.5mm | 9.20±0.33N | -0.72mm | 0.01mm | 0 |

### 与 V2 对比
- V2: 8.05±0.40N（模型力开环 + 参考轨迹锚定）
- V5: 8.00±0.08N（真实力闭环 + 速度积分）
- V5 无误差精度超越 V2，带误差能自动补偿几何偏移

### 关键教训
- 逆推值直接驱动 PID（不追 `dn_target-dn_actual`）→ 力级闭环
- 标架必须在最近接触点计算（不按弧长取）→ 力分解方向正确
- 常数项 = F_target 嵌入逆推 → 打破自洽

### 代码
- `code/sim/force_control_sim_v5.py` — 控制器
- `code/lib_v2/force_field_fixed.py` — 力场一次模型
- `code/sim/run_sim_v5.py` — 运行脚本
- 已推 GitHub: `0fb54a7`

### 可视化脚本
- ✅ `draw_section.py` / `section_gallery.py` / `section_with_ball.py` / `force_profile.py` / `p0_force_field.py`

### 误差实验
- ✅ 不同方向轴线偏移产生可区分的力曲线指纹
