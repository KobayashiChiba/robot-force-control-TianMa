# sphere_contact_force 倾斜圆柱 Bug 修复方案

> 2026-07-29 发现

---

## 问题描述

`sphere_contact.py` 中的 `_inside_cyl_z` 和 `_inside_cyl_y` 函数在判断球面采样点是否在圆柱内部时，使用了**假设圆柱轴线平行于坐标轴**的简化公式：

```python
# 当前错误实现
def _inside_cyl_z(pts, cyl_z):
    X0, Y0 = cyl_z.p1[0], cyl_z.p1[1]  # ← 只取 p1 的 XY 坐标
    return sqrt((pts.x - X0)² + (pts.y - Y0)²) < radius  # ← 隐含假设轴线在 XY 平面上位置不变

def _inside_cyl_y(pts, cyl_y):
    X0, Z0 = cyl_y.p1[0], cyl_y.p1[2]  # ← 只取 p1 的 XZ 坐标
    return sqrt((pts.x - X0)² + (pts.z - Z0)²) < radius  # ← 同样假设
```

### 为什么之前没发现

标准圆柱由圆柱拟合模块生成（`cylinder_fitting.py`），拟合时强制轴线平行于坐标轴，`p1` 和 `p2` 的 XY（Z圆柱）或 XZ（Y圆柱）坐标相同。所以简化公式对标准圆柱**碰巧正确**。

但误差圆柱由 `generate_error_cylinders` 生成，对每个端点做 ±1mm 独立随机偏移：

```python
def generate_error_cylinders(cy, cz, rng):
    dp1_z = rng.uniform(-1, 1, 3)  # Z圆柱端点1 的 XYZ 各 ±1mm
    dp2_z = rng.uniform(-1, 1, 3)  # Z圆柱端点2
    dp1_y = rng.uniform(-1, 1, 3)  # Y圆柱端点1
    dp2_y = rng.uniform(-1, 1, 3)  # Y圆柱端点2
```

这导致误差圆柱的轴线不再平行于坐标轴。以 seed=5 为例：

```
Z 误差圆柱: p1=[71.95, 65.74, -60.29] → p2=[73.34, 64.98, -19.48]
            p1 和 p2 的 XY 差: ΔX=1.39mm, ΔY=-0.76mm
```

用 p1 的 XY 坐标作为整条轴线的 XY 坐标，实际上在 p2 处偏差了约 1.6mm。

### 错误影响量化

| 受影响点 | 正确径向距离 | 错误径向距离 | 偏差 |
|----------|-------------|-------------|------|
| 误差接触曲线点 [60.04, 78.23, -39.79] | Z: 18.0028mm | — | 碰巧对 |
| 标准接触曲线点 [60.49, 78.41, -39.89] | Z: 17.82mm(?) | — | — |
| 力场采样点（随机）| 取决于位置 | 可能偏差 0~2mm | ~0.5mm 量级 |

### 影响范围

影响**所有涉及误差工件的力计算**：

| 模块 | 影响 |
|------|------|
| `sphere_contact.py` | `_inside_cyl_z`, `_inside_cyl_y` — 接触点筛选错误 |
| `sphere_contact.py` | `sphere_contact_force` — 力方向和大小的计算偏差 |
| `simulator.py` | 仿真中每步的力反馈有偏差 |
| `controller.py` | 力控闭环信号有偏差 |
| `lookup_inverse.py` | 查表数据基于错误力场 |
| 力场图相关脚本 | 热力图数据有偏差 |
| `run_sim_error.py` | True dn/db 计算中的标架（不影响直接影响，但力场偏差间接影响仿真路径） |

---

## 修复方案

### 修改文件

`force_feedback_v3/lib/sphere_contact.py`

### 核心改动

将 `_inside_cyl_z` 和 `_inside_cyl_y` 替换为一个统一的 `_inside_cyl` 函数，使用**轴线投影法**计算径向距离：

```python
def _inside_cyl(pts, cyl):
    """
    判断点是否在圆柱内部（沿轴线投影后计算径向距离）。

    Args:
        pts: (N, 3) 待检测点
        cyl: CylinderDef，任意方向的圆柱

    Returns:
        (N,) bool array，True=在圆柱内部
    """
    axis = cyl.p2 - cyl.p1
    L = np.linalg.norm(axis)
    d = axis / L                           # 轴线方向单位向量

    v = pts - cyl.p1                       # (N,3) 每点到 p1 的向量
    proj = np.dot(v, d)                    # (N,)  沿轴向的投影长度
    proj = np.clip(proj, 0, L)            # 限制在圆柱有限长度内

    ax_pts = cyl.p1 + proj[:, None] * d   # (N,3) 轴线上投影点
    r = np.linalg.norm(pts - ax_pts, axis=1)  # (N,) 径向距离

    return r < cyl.radius - 1e-6
```

### 调用处修改

原代码：
```python
in_z = _inside_cyl_z(pts, cyl_z)
in_y = _inside_cyl_y(pts, cyl_y)
```

改为：
```python
in_z = _inside_cyl(pts, cyl_z)
in_y = _inside_cyl(pts, cyl_y)
```

### 性能评估

当前方案对 12800 个球面采样点做逐点循环（for 循环），改为向量化 `_inside_cyl` 后批量计算——**性能应有所提升**。

### 兼容性

对标准圆柱（轴线平行坐标轴），新算法和旧算法结果一致（轴线投影后 `ax_pts` 的无关坐标分量自动抵消）。

### 受影响需重跑的内容

| 项目 | 重跑方式 |
|------|---------|
| seed0-9 仿真 | 重新跑 `run_sim_error.py` |
| 力场对比图 | 重新跑 `plot_std_force_field_p075.py` |
| 查表数据 | 重新 `build_table()` |
| True dn/db 无误差验证 | 重新跑验证脚本 |

---

## 修复步骤（建议顺序）

1. **修改 `sphere_contact.py`**：替换 `_inside_cyl_z`/`_inside_cyl_y` 为 `_inside_cyl`
2. **自测验证**：取标准圆柱上一个已知点，验证新老算法结果一致
3. **误差圆柱验证**：取误差接触曲线上的点，验证其在两个圆柱面上（径向距离 ≈ 各自半径）
4. **跑无误差仿真**：验证和之前结果一致（标准圆柱不受影响）
5. **跑 seed5 仿真**：和之前结果对比差异量
6. **更新力场图**：重画 p=0.75 对比图
7. **更新查表**：如有需要重建
8. **文档记录**：更新 `progress.md` 和 `log.md`
