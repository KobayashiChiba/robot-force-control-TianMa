# 机器人末端力控 — V2 标准库文档

> 最后更新：2026-07-20

---

## 1. 文件结构

```
code/
├── lib_v2/                          ← V2 标准库（5 个模块）
│   ├── cylinder_def.py              ← CylinderDef 数据类
│   ├── cylinder_fitting_v2.py       ← L0：散点拟合圆柱
│   ├── cylinder_geometry_v2.py      ← L1：正交圆柱交线几何
│   ├── contact_frame_v2.py          ← 接触标架计算
│   └── force_mechanics_v2.py        ← L2：力分解与运动趋势
│
├── scripts_v2/                      ← 测试和流程脚本
│   ├── gen_standard_curves.py       ← 生成标准曲线 pkl
│   ├── pipeline_full.py             ← L0→L1→L2 全链路验证+可视化
│   └── test_*.py                    ← 各层验证脚本
│
├── lib/                             ← V1 旧库（保留参考，不删）
│
└── data/
    ├── standard_curves_v2.pkl       ← V2 标准曲线（唯一）
    ├── 球刀中心点及轮廓轨迹点.xlsx   ← 原始实测数据
    └── 球刀中心点_修正后.xlsx        ← Z+4.815mm 修正后数据
```

**依赖链：** `cylinder_def` → `fitting_v2` / `geometry_v2` → `contact_frame_v2` → `force_mechanics_v2`

---

## 2. 模块详解

### 2.0 `cylinder_def.py` — 圆柱定义

所有 V2 模块的公共数据类。

```python
from cylinder_def import CylinderDef
```

#### `CylinderDef(p1, p2, radius)`

| 字段 | 类型 | 说明 |
|------|------|------|
| `p1` | `ndarray (3,)` | 轴线上第一点 |
| `p2` | `ndarray (3,)` | 轴线上第二点（p1≠p2） |
| `radius` | `float` | 圆柱半径 (mm) |

**派生属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `direction` | `ndarray (3,)` | 轴线单位方向向量 `normalize(p2-p1)` |
| `nearest_axis` | `str` | 最接近的坐标轴 `'X'`/`'Y'`/`'Z'` |
| `axis_point` | `ndarray (3,)` | p1（任一点，用于径向向量计算） |
| `axis_dir_vector` | `ndarray (3,)` | 最近坐标轴的单位方向 |

**工厂方法：**

```python
# 从轴对齐参数创建（L0 拟合输出用）
cyl = CylinderDef.from_axis_aligned(
    axis='Y',
    axis_point_like=np.array([51.5, 65.2, -39.7]),
    radius=9.0,
    pts_range=(51.6, 78.4),  # 数据沿轴 min/max
)

# 从任意两点创建（误差仿真等场景）
cyl = CylinderDef.from_two_points(
    p1=np.array([...]),
    p2=np.array([...]),
    radius=10.0,
)
```

---

### 2.1 `cylinder_fitting_v2.py` — L0 散点拟合

```python
from cylinder_fitting_v2 import fit_cylinders_from_points
```

#### `fit_cylinders_from_points(pts, axis1, axis2) → (list[CylinderDef], list[dict])`

投影圆拟合，线性最小二乘。圆柱轴线方向由外部指定（坐标轴对齐假设）。

**参数：**
- `pts` — `(N, 3)` 实测散点
- `axis1`, `axis2` — `'X'`/`'Y'`/`'Z'`，两圆柱轴线方向，必须不同

**返回：**
- `cyls[0], cyls[1]` — `CylinderDef`，p1/p2 自动取数据沿轴 min/max
- `details[0], details[1]` — `{'rms', 'max_err', 'residuals'}`

**示例：**
```python
cyls, details = fit_cylinders_from_points(contact_pts, 'Y', 'Z')
cyl_y, cyl_z = cyls
print(f"Y: r={cyl_y.radius:.3f} RMS={details[0]['rms']:.4f}")
print(f"Z: r={cyl_z.radius:.3f} RMS={details[1]['rms']:.4f}")
```

---

### 2.2 `cylinder_geometry_v2.py` — L1 交线几何

```python
from cylinder_geometry_v2 import sample_intersection, resample_curve, GeomV2
```

#### `GeomV2`

| 字段 | 类型 | 说明 |
|------|------|------|
| `n_samples` | `int` | 采样点数 |
| `sample_pts` | `ndarray (N, 3)` | 均匀弧长采样点（世界坐标） |
| `cyl1`, `cyl2` | `CylinderDef` | 圆柱参数 |

#### `sample_intersection(cyl1, cyl2, n_samples=1000, N_curve=250) → GeomV2`

计算两圆柱相交曲线，均匀弧长采样。内部在 `{u1,u2,u3}` 基下求解，支持任意轴线方向。

**示例：**
```python
geom = sample_intersection(cyl_y, cyl_z, n_samples=500)
curve = geom.sample_pts  # (500, 3)
```

#### `resample_curve(geom, n_samples) → GeomV2`

对已有 GeomV2 按新采样点数重采样。

---

### 2.3 `contact_frame_v2.py` — 接触标架

```python
from contact_frame_v2 import compute_frame, compute_frames_batch, ContactFrame
```

#### `ContactFrame`

| 字段 | 类型 | 说明 |
|------|------|------|
| `tangent` | `ndarray (3,)` | 切向量 `t = r_y × r_z` |
| `normal` | `ndarray (3,)` | 法向量 `n = w_y·r_y + w_z·r_z`（w∝r^(2/3)） |
| `radial_z` | `ndarray (3,)` | Z圆柱径向（指向Z轴心） |

> 基底为非正交：`t⟂n` 且 `t⟂rz`，但 `n` 与 `rz` 不垂直。

#### `compute_frame(contact_pt, cyl_y, cyl_z) → ContactFrame`

计算接触曲线上一点处的局部标架。径向向量用真实轴线方向投影。

**参数：**
- `contact_pt` — `(3,)` 接触点坐标
- `cyl_y`, `cyl_z` — `CylinderDef`

**示例：**
```python
frame = compute_frame(P, cyl_y, cyl_z)
print(frame.tangent, frame.normal, frame.radial_z)
```

#### `compute_frames_batch(contact_pts, cyl_y, cyl_z) → dict`

批量计算。返回 `{'tangents': (N,3), 'normals': (N,3), 'radial_z': (N,3)}`。

---

### 2.4 `force_mechanics_v2.py` — L2 力分解

```python
from force_mechanics_v2 import (
    compute_point_basis, decompose_force,
    expected_force, compute_normal_motion_trend, compute_vertical_motion_trend,
    Basis, ForceDecomp,
)
```

#### `Basis`

| 字段 | 类型 | 对应 |
|------|------|------|
| `tangent` | `ndarray (3,)` | `t = r_y × r_z` |
| `normal` | `ndarray (3,)` | `n = w_y·r_y + w_z·r_z` |
| `vertical` | `ndarray (3,)` | `rz`（Z圆柱径向） |

#### `ForceDecomp`

| 字段 | 类型 | 说明 |
|------|------|------|
| `coeffs` | `(3,)` | 分解系数 `[a, b, c]` |
| `Ft_vec` | `(3,)` | a·t（切向力） |
| `Fn_vec` | `(3,)` | b·n（法向力） |
| `Fv_vec` | `(3,)` | c·v（Z径向力） |
| `error` | `float` | 重建误差（应≈0） |

#### `compute_point_basis(P, geom) → Basis`

计算交线上点 P 的力分解基底 `{t, n, rz}`。内部调用 `contact_frame.compute_frame`。

#### `decompose_force(F, basis) → ForceDecomp`

在非正交基 `{t, n, v}` 上严格分解力向量，解 3×3 线性方程组。

**示例：**
```python
basis = compute_point_basis(P, geom)
decomp = decompose_force(F, basis)
print(f"Ft={decomp.coeffs[0]:.2f}, Fn={decomp.coeffs[1]:.2f}, Fv={decomp.coeffs[2]:.2f}")
```

#### `expected_force(coeffs, basis) → ForceDecomp`

从系数反构力向量：`F = a·t + b·n + c·v`。

#### `compute_normal_motion_trend(decomp, basis, offset=8.0) → (3,)`

法向运动趋势：`(Fn_coeff + offset) × normal`。

#### `compute_vertical_motion_trend(decomp, basis) → (3,)`

垂向运动趋势：`|Fv_coeff| × unit(-t × v)`。

---

## 3. 标准曲线

### `standard_curves_v2.pkl`

pickle 文件，包含以下内容：

| 键 | 值 | 说明 |
|---|---|---|
| `ball_center_geom` | `GeomV2` | 球刀中心标准交线（500点） |
| `contact_geom` | `GeomV2` | 标准接触交线（500点） |
| `ball_radius` | `4.0` | 球刀半径 (mm) |
| `cyl_ball_y` | `CylinderDef` | 球刀中心 Y 圆柱 |
| `cyl_ball_z` | `CylinderDef` | 球刀中心 Z 圆柱 |
| `cyl_contact_y` | `CylinderDef` | 接触 Y 圆柱 |
| `cyl_contact_z` | `CylinderDef` | 接触 Z 圆柱 |

**圆柱参数一览：**

| | Y圆柱 | Z圆柱 | 轴线长 | RMS |
|---|---|---|---|---|
| 球刀中心 | r=7.35mm | r=15.77mm | — / — | 0.07 / 0.02mm |
| 接触曲线 | r=9.00mm | r=18.00mm | 40mm / 40mm | 0.003 / 0.003mm |

**加载示例：**
```python
import pickle
with open('code/data/standard_curves_v2.pkl', 'rb') as f:
    d = pickle.load(f)
curve = d['contact_geom'].sample_pts  # (500, 3)
```

**重新生成：**
```bash
python code/scripts_v2/gen_standard_curves.py
```

---

## 4. 完整使用示例

```python
import sys
sys.path.insert(0, 'code/lib')
sys.path.insert(0, 'code/lib_v2')

import numpy as np
from cylinder_fitting_v2 import fit_cylinders_from_points
from cylinder_geometry_v2 import sample_intersection
from force_mechanics_v2 import compute_point_basis, decompose_force

# L0: 拟合
cyls, details = fit_cylinders_from_points(measured_pts, 'Y', 'Z')
cyl_y, cyl_z = cyls

# L1: 交线
geom = sample_intersection(cyl_y, cyl_z, n_samples=500)

# L2: 力分解（逐个采样点）
F = np.array([5.0, -1.0, -8.0])  # 实测力
for P in geom.sample_pts:
    basis = compute_point_basis(P, geom)
    decomp = decompose_force(F, basis)
    # Ft = decomp.Ft_vec, Fn = decomp.Fn_vec, Fv = decomp.Fv_vec
```

---

## 5. V1 → V2 对照

| | V1 (`code/lib/`) | V2 (`code/lib_v2/`) |
|---|---|---|
| 圆柱表示 | `(axis_char, axis_point, radius)` | `CylinderDef(p1, p2, radius)` |
| 轴线方向 | 固定坐标轴 X/Y/Z | 任意方向向量 |
| 交线算法 | 坐标索引投影 | `{u1,u2,u3}` 基变换 |
| 标架法向量 | `n₁+n₂` 角平分线 | `w_y·r_y+w_z·r_z` 加权 (r^(2/3)) |
| 标架切向量 | 解析（需分支定位） | `r_y × r_z`（通用，无需分支） |
| 力分解基底 | `{t_a, n_平分, n₂}` | `{t_×, n_加权, rz}` |
| 兼容旧代码 | — | 不兼容（V2 独立） |
