# ForceFeedback 技术参考手册

> **版本** 1.1 &nbsp;|&nbsp; **日期** 2026-07-09 &nbsp;|&nbsp; **语言** Python 3.10+ / NumPy

---

## 目录

- [1. 架构概览](#1-架构概览)
- [2. 数据类定义](#2-数据类定义)
  - [2.1 Geom](#21-geom)
  - [2.2 Basis](#22-basis)
  - [2.3 ForceDecomp](#23-forcedecomp)
  - [2.4 CylinderParams](#24-cylinderparams)
- [3. cylinder_fitting — 参数估计层](#3-cylinder_fitting--参数估计层)
  - [3.1 fit_cylinders_from_points](#31-fit_cylinders_from_points)
- [4. cylinder_geometry — 几何计算层](#4-cylinder_geometry--几何计算层)
  - [4.1 sample_intersection](#41-sample_intersection)
  - [4.2 resample_curve](#42-resample_curve)
- [5. force_mechanics — 力学计算层](#5-force_mechanics--力学计算层)
  - [5.1 compute_point_basis](#51-compute_point_basis)
  - [5.2 decompose_force](#52-decompose_force)
  - [5.3 expected_force](#53-expected_force)
  - [5.4 compute_normal_motion_trend](#54-compute_normal_motion_trend)
  - [5.5 compute_vertical_motion_trend](#55-compute_vertical_motion_trend)
- [6. 数学附录](#6-数学附录)
  - [6.1 弧长重参数化补充](#61-弧长重参数化补充)
  - [6.2 非正交基可逆性证明](#62-非正交基可逆性证明)
  - [6.3 投影圆拟合线性化](#63-投影圆拟合线性化)
- [7. 术语表](#7-术语表)

---

## 1. 架构概览

```
┌─────────────────────────────────────┐
│       cylinder_fitting.py            │  Layer 0 — 参数估计
│  fit_cylinders_from_points()         │
└──────────────┬──────────────────────┘
               │  CylinderParams 列表 + Geom
               ▼
┌─────────────────────────────────────┐
│        cylinder_geometry.py          │  Layer 1 — 几何计算
│  sample_intersection()               │
│  resample_curve()                    │
└──────────────┬──────────────────────┘
               │  Geom 实例
               ▼
┌─────────────────────────────────────┐
│         force_mechanics.py           │  Layer 2 — 力学计算
│  compute_point_basis()               │
│  decompose_force()                   │
│  expected_force()                    │
│  compute_normal_motion_trend()       │
│  compute_vertical_motion_trend()     │
└─────────────────────────────────────┘
```

**依赖关系**：`force_mechanics` 依赖 `cylinder_geometry` 导出的 `Geom` 数据类及内部函数。`cylinder_fitting` 依赖 `cylinder_geometry` 的 `Geom` 数据类（作为输出载体）及 `resample_curve` 供下游调用。

**前置条件**：两圆柱轴线必须分别平行于 X / Y / Z 轴之一，且方向不同。拟合仅使用投影法，不涉及 3D 非线性拟合。

---

## 2. 数据类定义

### 2.1 Geom

> 来源：`cylinder_geometry.py`  
> 用途：存储两圆柱相交曲线的均匀弧长采样结果及生成参数。

| 属性 | 类型 | 说明 |
|---|---|---|
| `n_samples` | `int` | 采样点数量，即 `sample_pts.shape[0]` |
| `sample_pts` | `np.ndarray` 形状 `(N, 3)` | 沿相交曲线均匀弧长分布的 N 个三维坐标 |
| `axis1` | `str` | 圆柱 1 轴线方向，取值 `'X'` / `'Y'` / `'Z'` |
| `c1` | `tuple` 长度 3 | 圆柱 1 轴线上一点坐标 `(cx, cy, cz)` |
| `r1` | `float` | 圆柱 1 半径，> 0 |
| `axis2` | `str` | 圆柱 2 轴线方向，必须与 `axis1` 不同 |
| `c2` | `tuple` 长度 3 | 圆柱 2 轴线上一点坐标 `(cx, cy, cz)` |
| `r2` | `float` | 圆柱 2 半径，> 0 |

**构造方式**：通过 `sample_intersection()`、`resample_curve()` 或 `fit_cylinders_from_points()` 创建。拟合产生的 `Geom` 中 `n_samples=0`、`sample_pts` 为空数组，参数保留 1 位小数，供 `resample_curve` 直接使用。

**序列化**：标准 `dataclass`，可使用 `dataclasses.asdict()` 转换（`np.ndarray` 字段需额外处理，建议使用 `.tolist()` 后序列化为 JSON）。

---

### 2.2 Basis

> 来源：`force_mechanics.py`  
> 用途：曲线上一点处的非正交基底，由三个单位向量组成。

| 属性 | 类型 | 说明 |
|---|---|---|
| `tangent` | `np.ndarray` 形状 `(3,)` | 曲线单位切向量 $\mathbf{t}$，$\|\mathbf{t}\|=1$ |
| `normal` | `np.ndarray` 形状 `(3,)` | 打磨法向量 $\mathbf{n}$，两圆柱面径向法向的角平分线方向，$\|\mathbf{n}\|=1$ |
| `vertical` | `np.ndarray` 形状 `(3,)` | 垂直向量 $\mathbf{v}$，即圆柱 2 表面径向向外法向量，$\|\mathbf{v}\|=1$ |

**几何关系**：$\mathbf{t}, \mathbf{n}, \mathbf{v}$ 一般不正交。$\mathbf{n}$ 是 $\mathbf{n}_1$ 与 $\mathbf{n}_2$ 的和方向（归一化），位于两圆柱面法向的角平分线上。

**构造方式**：仅通过 `compute_point_basis()` 创建。

---

### 2.3 ForceDecomp

> 来源：`force_mechanics.py`  
> 用途：力向量 $\mathbf{F} \in \mathbb{R}^3$ 在非正交基 $\{\mathbf{t}, \mathbf{n}, \mathbf{v}\}$ 上的分解结果。

| 属性 | 类型 | 说明 |
|---|---|---|
| `coeffs` | `np.ndarray` 形状 `(3,)` | 分解系数 $[a, b, c]$，满足 $a\mathbf{t} + b\mathbf{n} + c\mathbf{v} = \mathbf{F}$ |
| `Ft_vec` | `np.ndarray` 形状 `(3,)` | 切向分量 $a \cdot \mathbf{t}$ |
| `Fn_vec` | `np.ndarray` 形状 `(3,)` | 法向分量 $b \cdot \mathbf{n}$ |
| `Fv_vec` | `np.ndarray` 形状 `(3,)` | 垂直分量 $c \cdot \mathbf{v}$ |
| `error` | `float` | 重建误差 $\|\mathbf{F} - (a\mathbf{t} + b\mathbf{n} + c\mathbf{v})\|$，理论上 $\approx 0$ |

**构造方式**：通过 `decompose_force()`（error $\approx 0$）或 `expected_force()`（error $= 0$）创建。

---

### 2.4 CylinderParams

> 来源：`cylinder_fitting.py`  
> 用途：单个圆柱的投影圆拟合结果，包含完整精度参数和残差统计。

| 属性 | 类型 | 说明 |
|---|---|---|
| `axis` | `str` | 圆柱轴线方向，取值 `'X'` / `'Y'` / `'Z'` |
| `axis_point` | `np.ndarray` 形状 `(3,)` | 轴线上一点的三维坐标 |
| `radius` | `float` | 圆柱半径 (mm)，> 0 |
| `rms` | `float` | RMS 残差 (mm)，$\sqrt{\frac{1}{N}\sum \text{res}_i^2}$ |
| `max_err` | `float` | 最大绝对残差 (mm) |
| `residuals` | `np.ndarray` 形状 `(N,)` | 各点到拟合圆的有向距离 |

**构造方式**：仅通过 `fit_cylinders_from_points()` 内部创建的列表获取。一次拟合返回两个 `CylinderParams` 实例（对应两个圆柱），供展示拟合结果使用。

**精度**：投影圆拟合使用线性最小二乘（3 参数），`axis_point` 中沿轴线方向的坐标用散点均值填充（该坐标无法从投影中确定）。

---

## 3. cylinder_fitting — 参数估计层

### 3.1 fit_cylinders_from_points

```python
def fit_cylinders_from_points(pts, axis1, axis2) -> (list[CylinderParams], Geom)
```

**功能**：从散点数据（如实测接触点）拟合两个正交圆柱的参数，返回完整拟合结果和可复用的几何对象。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `pts` | `np.ndarray` 形状 `(N, 3)` | — | 散点三维坐标 |
| `axis1` | `str` | — | 圆柱 1 轴线方向 (`'X'`/`'Y'`/`'Z'`) |
| `axis2` | `str` | — | 圆柱 2 轴线方向（必须 $\neq$ `axis1`） |

**返回值**：`(list[CylinderParams], Geom)` — 两个圆柱的完整拟合参数列表，以及一个 `n_samples=0` 的 `Geom` 对象（参数保留 1 位小数）。

**异常**：`ValueError` — 当 `axis1 == axis2` 时抛出。

**使用示例**：

```python
from cylinder_fitting import fit_cylinders_from_points
from cylinder_geometry import resample_curve

# 从实测点拟合
params_list, geom = fit_cylinders_from_points(pts_measured, 'Y', 'Z')

# 展示拟合结果
p1, p2 = params_list
print(f"Y圆柱: r={p1.radius:.3f}mm, RMS={p1.rms:.4f}mm")
print(f"Z圆柱: r={p2.radius:.3f}mm, RMS={p2.rms:.4f}mm")

# 用 geom 重采样得到曲线
geom_curve = resample_curve(geom, n_samples=500)
```

**内部流程**：

```
验证 axis1 ≠ axis2
  │
  ├─ ① 分别对两个圆柱做投影圆拟合 (→ CylinderParams)
  │
  └─ ② 组装 Geom (n_samples=0, 参数 1 位小数)
```

---

#### ① 投影圆拟合（内部：`_fit_projection`）

对单个圆柱，将散点投影到与轴线垂直的平面，拟合圆 $(u - U_0)^2 + (v - V_0)^2 = R^2$。

**坐标映射**：

| 轴线 | 投影平面 | 待求圆心 | 均值填充坐标 |
|---|---|---|---|
| `'X'` | YZ 平面 | $(Y_0, Z_0)$ | X |
| `'Y'` | XZ 平面 | $(X_0, Z_0)$ | Y |
| `'Z'` | XY 平面 | $(X_0, Y_0)$ | Z |

**线性化**：圆方程为 $(u - U_0)^2 + (v - V_0)^2 = R^2$，展开得：

$$u^2 + v^2 = 2U_0 \cdot u + 2V_0 \cdot v + (R^2 - U_0^2 - V_0^2)$$

令 $\mathbf{Y} = u^2 + v^2$，$\mathbf{A} = [u, v, \mathbf{1}]$，解线性方程组 $\mathbf{A} \cdot \mathbf{\beta} = \mathbf{Y}$，其中：

$$\beta_1 = 2U_0,\quad \beta_2 = 2V_0,\quad \beta_3 = R^2 - U_0^2 - V_0^2$$

反解：

$$U_0 = \frac{\beta_1}{2},\quad V_0 = \frac{\beta_2}{2},\quad R = \sqrt{\beta_3 + U_0^2 + V_0^2}$$

**复杂度**：$\mathcal{O}(N)$（线性最小二乘 `np.linalg.lstsq`）。

**局限性**：圆柱在轴线方向无限延伸，投影法无法确定轴线上的坐标（即被投影掉的坐标）。该坐标用散点均值填充，不影响 RMS 残差计算和后续曲线重建。

---

## 4. cylinder_geometry — 几何计算层

### 4.1 sample_intersection

```python
def sample_intersection(axis1, c1, r1, axis2, c2, r2, n_samples=1000, N_curve=250) -> Geom
```

**功能**：计算两圆柱相交曲线，并对其进行均匀弧长采样。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `axis1` | `str` | — | 圆柱 1 轴线方向 (`'X'`/`'Y'`/`'Z'`) |
| `c1` | `tuple` | — | 圆柱 1 轴心坐标 |
| `r1` | `float` | — | 圆柱 1 半径 |
| `axis2` | `str` | — | 圆柱 2 轴线方向（必须 $\neq$ `axis1`） |
| `c2` | `tuple` | — | 圆柱 2 轴心坐标 |
| `r2` | `float` | — | 圆柱 2 半径 |
| `n_samples` | `int` | 1000 | 输出采样点数 |
| `N_curve` | `int` | 250 | 每段分支内部离散点数（影响采样精度） |

**返回值**：`Geom` 实例。

**异常**：`ValueError` — 当 `axis1 == axis2` 或两圆柱不相交时抛出。

**使用示例**：

```python
from cylinder_geometry import sample_intersection

# Y 轴圆柱 (半径 10) 与 Z 轴圆柱 (半径 20) 相交
geom = sample_intersection('Y', (0, 0, 0), 10, 'Z', (27, 0, 0), 20, n_samples=500)
print(geom.sample_pts.shape)  # (500, 3)
```

**内部流程总览**：

```
验证 axis1 ≠ axis2
  │
  ├─ ① 参数化求解 → 生成 4 段分支曲线 (每段 N_curve 点)
  │
  ├─ ② 拼接 4 段为闭合点列 (去重衔接点)
  │
  └─ ③ 折线均匀弧长重采样 → n_samples 个点 → 组装 Geom
```

---

#### ① 参数化求解 （内部：`_get_branch_meta` → `_intersect_raw`）

**坐标映射**

设圆柱 1 轴线平行于 `axis1`，圆柱 2 轴线平行于 `axis2`。三个坐标轴中，两个被轴线占用，剩余一个为公共参数 $t$。令：

- $k$ = 公共坐标索引（不被轴线占用的坐标）
- $i$ = 圆柱 1 的另一径向坐标索引 (= `axis2` 的索引)
- $j$ = 圆柱 2 的另一径向坐标索引 (= `axis1` 的索引)

**圆柱方程参数化**

圆柱 1（轴线 $\parallel \mathbf{d}_1$）的隐式方程消去轴线分量后：
$$(t - c_{1,k})^2 + (x_i - c_{1,i})^2 = R_1^2$$

解得：
$$x_i(t) = c_{1,i} \pm \sqrt{R_1^2 - (t - c_{1,k})^2}$$

同理圆柱 2：
$$x_j(t) = c_{2,j} \pm \sqrt{R_2^2 - (t - c_{2,k})^2}$$

**$t$ 的有效范围**（两圆柱根号内同时非负）：
$$t \in [t_\text{min},\ t_\text{max}] = [\max(c_{1,k}-R_1,\ c_{2,k}-R_2),\ \min(c_{1,k}+R_1,\ c_{2,k}+R_2)]$$

若 $t_\text{min} \ge t_\text{max}$，两圆柱不相交，抛出 `ValueError`。

**4 段分支生成**

正负号组合 $(s_1, s_2) \in \{+1, -1\}^2$ 共 4 种，每段配以正向或反向的 $t$ 参数化以保证闭合连续性：

| 分支 | $s_1$ | $s_2$ | t 遍历方向 | is_t_up |
|---|---|---|---|---|
| 0 | +1 | +1 | $t_\text{min} \to t_\text{max}$ | True |
| 1 | +1 | −1 | $t_\text{max} \to t_\text{min}$ | False |
| 2 | −1 | −1 | $t_\text{min} \to t_\text{max}$ | True |
| 3 | −1 | +1 | $t_\text{max} \to t_\text{min}$ | False |

每段分支的离散点：
$$t_\text{up} = \text{linspace}(t_\text{min}, t_\text{max}, N_\text{curve}),\quad t_\text{down} = \text{linspace}(t_\text{max}, t_\text{min}, N_\text{curve})$$

三维坐标：
$$\mathbf{P}(t) = \big(t,\ x_j(t),\ x_i(t)\big)$$

代码中预计算差值 $\Delta_{1,\text{up}} = t_\text{up} - c_{1,k}$ 等来避免重复平方/开方，4 次 `np.sqrt` 各作用于长度为 $N_\text{curve}$ 的数组。

**复杂度**：$\mathcal{O}(N_\text{curve})$。

---

#### ② 闭合点列拼接 （内部：`_build_closed_curve`）

4 段分支按 `[0, 3, 2, 1]` 顺序首尾相接：

```
分支 0 ──→ 分支 3 ──→ 分支 2 ──→ 分支 1 ──→ 回到分支 0
```

相邻段衔接点去重：首段从索引 0 开始，后续段跳过首点。总点数 $M = 4N_\text{curve} - 3$。

---

#### ③ 均匀弧长重采样 （内部：`_sample_uniform`）

直接按 $t$ 等距采样会导致点密度不均匀（$\|\mathbf{P}'(t)\|$ 在边界附近趋于无穷）。本实现采用折线近似弧长重采样：

**弦长计算**：
$$L_m = \|\mathbf{P}_{m+1} - \mathbf{P}_m\|,\quad m = 0,\dots,M-2$$
$$L_{M-1} = \|\mathbf{P}_0 - \mathbf{P}_{M-1}\|\quad\text{(闭合边)}$$

**累积弧长**：
$$S_m = \sum_{p=0}^{m} L_p,\quad S_0 = 0,\quad S_M = L_\text{total}$$

**均匀目标**：
$$\hat{S}_q = \frac{q \cdot L_\text{total}}{n_\text{samples}},\quad q = 0,\dots,n_\text{samples}-1$$

**最近邻采样**：
$$m^*_q = \arg\min_m |S_m - \hat{S}_q|$$

取出 $\mathbf{P}_{m^*_q}$ 作为第 $q$ 个采样点。

**精度**：取决于 $N_\text{curve}$（默认 250），越大折线越逼近真实曲线。总复杂度 $\mathcal{O}(N_\text{curve} + n_\text{samples} \cdot N_\text{curve})$。

---

### 4.2 resample_curve

```python
def resample_curve(geom, n_samples) -> Geom
```

**功能**：对已有 `Geom` 重新按不同采样点数做均匀弧长采样。本质是 `sample_intersection` 的便捷封装——从 `geom` 提取原始圆柱参数后重新走完整流程。

| 参数 | 类型 | 说明 |
|---|---|---|
| `geom` | `Geom` | 已有的采样结果 |
| `n_samples` | `int` | 新的采样点数 |

**返回值**：新的 `Geom` 实例（原始圆柱参数保持一致）。

---

## 5. force_mechanics — 力学计算层

### 5.1 compute_point_basis

```python
def compute_point_basis(P, geom) -> Basis
```

**功能**：计算相交曲线上任意一点 $\mathbf{P}$ 处的非正交基底 $\{\mathbf{t}, \mathbf{n}, \mathbf{v}\}$。内部自动定位 $\mathbf{P}$ 所属的分支和参数 $t$，无需调用者提供。

| 参数 | 类型 | 说明 |
|---|---|---|
| `P` | `np.ndarray` 形状 `(3,)` | 曲线上一点的空间坐标 |
| `geom` | `Geom` | 由 `sample_intersection()` 或 `resample_curve()` 生成 |

**返回值**：`Basis` 实例。

---

#### 步骤 1：获取分支元信息

调用 `_get_branch_meta(geom)` 获取坐标映射 $(k,i,j)$、t 参数数组 `curves_t`、4 段分支符号表 `branch_info`。此调用仅计算元信息，不生成完整曲线几何（详见 [4.1 节 ①](#参数化求解-内部_get_branch_meta--_intersect_raw)）。

---

#### 步骤 2：自动定位分支

从输入点提取公共坐标分量 $t_\text{val} = P_k$。

对 4 个分支 $(s_1, s_2)$，分别按参数方程重建期望点 $\mathbf{P}_\text{exp}$：
$$P_{\text{exp},k} = t_\text{val}$$
$$P_{\text{exp},i} = c_{1,i} + s_1 \cdot \sqrt{R_1^2 - (t_\text{val} - c_{1,k})^2}$$
$$P_{\text{exp},j} = c_{2,j} + s_2 \cdot \sqrt{R_2^2 - (t_\text{val} - c_{2,k})^2}$$

选择欧氏距离最小的分支：
$$\text{bid}^* = \arg\min_\text{bid} \|\mathbf{P} - \mathbf{P}_\text{exp}^{(\text{bid})}\|$$

再在选中分支的 t 数组中找最近邻，得到精确参数：
$$t^* = \text{curves\_t}[\text{bid}^*]\big[\arg\min_j |\text{curves\_t}[\text{bid}^*][j] - t_\text{val}|\big]$$

---

#### 步骤 3：圆柱面径向法向量

圆柱 $i$ 表面点 $\mathbf{P}$ 的径向向外法向量为 $\mathbf{P}$ 减去其在轴向上的投影（梯度法）：

$$\mathbf{n}_i = \frac{\mathbf{P} - \mathbf{c}_i - \big((\mathbf{P} - \mathbf{c}_i) \cdot \mathbf{d}_i\big) \mathbf{d}_i}
                     {\|\mathbf{P} - \mathbf{c}_i - \big((\mathbf{P} - \mathbf{c}_i) \cdot \mathbf{d}_i\big) \mathbf{d}_i\|},\quad i = 1, 2$$

其中 $\mathbf{d}_i \in \{(1,0,0), (0,1,0), (0,0,1)\}$ 为轴线方向单位向量。

---

#### 步骤 4：打磨法向量与垂直向量

打磨法向量取两圆柱面法向的角平分线方向（指向金属内部）：
$$\mathbf{n} = \frac{\mathbf{n}_1 + \mathbf{n}_2}{\|\mathbf{n}_1 + \mathbf{n}_2\|}$$

垂直向量即圆柱 2 表面法向：
$$\mathbf{v} = \mathbf{n}_2$$

---

#### 步骤 5：切向量

参数化曲线 $\mathbf{P}(t) = (t,\ x_j(t),\ x_i(t))$ 的导数：

$$\frac{dx_i}{dt} = -s_1 \cdot \frac{t - c_{1,k}}{\sqrt{R_1^2 - (t - c_{1,k})^2}},\qquad
  \frac{dx_j}{dt} = -s_2 \cdot \frac{t - c_{2,k}}{\sqrt{R_2^2 - (t - c_{2,k})^2}}$$

未归一化切向量：
$$\mathbf{T} = \left(1,\ \frac{dx_j}{dt},\ \frac{dx_i}{dt}\right)$$

若分支的 `is_t_up == False`（$t$ 递减遍历），取反 $\mathbf{T} \leftarrow -\mathbf{T}$。最后归一化：
$$\mathbf{t} = \frac{\mathbf{T}}{\|\mathbf{T}\|}$$

**复杂度**：$\mathcal{O}(N_\text{curve})$（来自 `_get_branch_meta`）。

---

### 5.2 decompose_force

```python
def decompose_force(F, basis) -> ForceDecomp
```

**功能**：将外力 $\mathbf{F} \in \mathbb{R}^3$ 在非正交基 $\{\mathbf{t}, \mathbf{n}, \mathbf{v}\}$ 上做严格（非投影）分解。

| 参数 | 类型 | 说明 |
|---|---|---|
| `F` | `np.ndarray` 形状 `(3,)` | 外力向量 |
| `basis` | `Basis` | `compute_point_basis()` 的返回值 |

**返回值**：`ForceDecomp` 实例。

**数学推导**：

非正交基下的分解等价于解线性方程组（而非点积投影）：

$$\mathbf{M} \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \mathbf{F},\quad
  \mathbf{M} = [\mathbf{t}\ |\ \mathbf{n}\ |\ \mathbf{v}] \in \mathbb{R}^{3 \times 3}$$

即 $a\mathbf{t} + b\mathbf{n} + c\mathbf{v} = \mathbf{F}$。由于 $\mathbf{t}, \mathbf{n}, \mathbf{v}$ 线性无关（证明见 [6.2 节](#62-非正交基可逆性证明)），$\mathbf{M}$ 可逆，使用 `np.linalg.solve` 直接求解。

各分量：$\mathbf{F}_t = a\mathbf{t},\ \mathbf{F}_n = b\mathbf{n},\ \mathbf{F}_v = c\mathbf{v}$

重建误差：$\varepsilon = \|\mathbf{F} - (\mathbf{F}_t + \mathbf{F}_n + \mathbf{F}_v)\| \approx 0$（仅受浮点精度影响）。

**复杂度**：$\mathcal{O}(1)$（3×3 矩阵求逆）。

---

### 5.3 expected_force

```python
def expected_force(coeffs, basis) -> ForceDecomp
```

**功能**：给定分解系数 $[a, b, c]$ 和基底，反向合成力向量。与 `decompose_force` 互逆。

| 参数 | 类型 | 说明 |
|---|---|---|
| `coeffs` | `ndarray(3,)` 或 `tuple` | 分解系数 $[a, b, c]$ |
| `basis` | `Basis` | 基底 |

**返回值**：`ForceDecomp` 实例（`error = 0.0`）。

**数学推导**：

$$\mathbf{F} = a\mathbf{t} + b\mathbf{n} + c\mathbf{v}$$
$$\mathbf{F}_t = a\mathbf{t},\quad \mathbf{F}_n = b\mathbf{n},\quad \mathbf{F}_v = c\mathbf{v}$$

---

### 5.4 compute_normal_motion_trend

```python
def compute_normal_motion_trend(decomp, basis, offset=8.0) -> np.ndarray
```

**功能**：计算法向力引起的运动趋势向量。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `decomp` | `ForceDecomp` | — | 力分解结果 |
| `basis` | `Basis` | — | 基底 |
| `offset` | `float` | 8.0 | 偏移量，确保力为零时仍有基准趋势 |

**返回值**：`np.ndarray` 形状 `(3,)`。

**公式**：

$$\mathbf{F}_\text{motion,n} = (b + \text{offset}) \cdot \mathbf{n},\quad b = \text{decomp.coeffs}[1]$$

---

### 5.5 compute_vertical_motion_trend

```python
def compute_vertical_motion_trend(decomp, basis) -> np.ndarray
```

**功能**：计算垂直力引起的运动趋势向量。

| 参数 | 类型 | 说明 |
|---|---|---|
| `decomp` | `ForceDecomp` | 力分解结果 |
| `basis` | `Basis` | 基底 |

**返回值**：`np.ndarray` 形状 `(3,)`。

**公式**：

方向取垂直于 $\mathbf{t}$ 和 $\mathbf{v}$ 张成平面的方向（符合打磨设备坐标系约定）：
$$\hat{\mathbf{d}} = \frac{-(\mathbf{t} \times \mathbf{v})}{\|\mathbf{t} \times \mathbf{v}\|}$$

大小取垂直系数绝对值：
$$\mathbf{F}_\text{motion,v} = |c| \cdot \hat{\mathbf{d}},\quad c = \text{decomp.coeffs}[2]$$

---

## 6. 数学附录

### 6.1 弧长重参数化补充

原始曲线 $\mathbf{P}(t)$ 不是弧长参数化。弧长微元：
$$ds = \|\mathbf{P}'(t)\|\, dt = \sqrt{1 + \left(\frac{dx_i}{dt}\right)^2 + \left(\frac{dx_j}{dt}\right)^2}\, dt$$

在 $t \to t_\text{min}$ 或 $t \to t_\text{max}$ 时 $\|\mathbf{P}'(t)\| \to \infty$（导数分母 $\sqrt{R^2 - (t-c)^2} \to 0$），因此直接按 $t$ 等距采样会导致曲线两端点密度远高于中间段。

本实现采用折线近似重采样（详见 [4.1 节 ③](#均匀弧长重采样-内部_sample_uniform)），以弦长 $\|\mathbf{P}_{m+1} - \mathbf{P}_m\|$ 近似微弧长 $ds$。当 $N_\text{curve} \to \infty$ 时，近似误差 $\to 0$。

---

### 6.2 非正交基可逆性证明

基底矩阵 $\mathbf{M} = [\mathbf{t}\ |\ \mathbf{n}\ |\ \mathbf{v}]$ 的可逆性等价于 $\mathbf{t}, \mathbf{n}, \mathbf{v}$ 线性无关。证明如下：

1. **$\mathbf{t}$ 沿交线方向**。交线为空间曲线，$\mathbf{t} \neq \mathbf{0}$。

2. **$\mathbf{n}$ 在角平分面内**。$\mathbf{n} = \frac{\mathbf{n}_1 + \mathbf{n}_2}{\|\mathbf{n}_1 + \mathbf{n}_2\|}$，其中 $\mathbf{n}_1 \perp$ 圆柱 1 轴线，$\mathbf{n}_2 \perp$ 圆柱 2 轴线。在非退化交线情形下 $\mathbf{n}_1 \neq \pm \mathbf{n}_2$（两圆柱面法向不一致），故 $\mathbf{n} \neq \mathbf{0}$。

3. **$\mathbf{v} = \mathbf{n}_2$**，垂直于圆柱 2 轴线方向。

4. 由于 $\mathbf{d}_1 \neq \mathbf{d}_2$（两轴线不平行），$\mathbf{n}_1$ 与 $\mathbf{n}_2$ 不共线，因此 $\mathbf{n}$（两者之和）不与 $\mathbf{v} = \mathbf{n}_2$ 共线。同时 $\mathbf{t}$ 沿交线，而 $\mathbf{n}$ 和 $\mathbf{v}$ 均为曲面法向，故 $\mathbf{t}$ 不在 $\mathbf{n}$–$\mathbf{v}$ 平面内。

因此 $\mathbf{t}, \mathbf{n}, \mathbf{v}$ 线性无关，$\det(\mathbf{M}) \neq 0$，方程组有唯一解。

---

### 6.3 投影圆拟合线性化

将空间散点投影到与圆柱轴线垂直的平面后，问题退化为平面圆拟合。圆的隐式方程：

$$(u - U_0)^2 + (v - V_0)^2 = R^2$$

展开：

$$u^2 + v^2 = 2U_0 u + 2V_0 v + (R^2 - U_0^2 - V_0^2)$$

令 $\mathbf{Y} = [u_i^2 + v_i^2]$，设计矩阵 $\mathbf{A} = [u_i, v_i, 1]$，则最小二乘解为：

$$\boldsymbol{\beta} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{Y}$$

反解：

$$U_0 = \frac{\beta_1}{2},\quad V_0 = \frac{\beta_2}{2},\quad R = \sqrt{\beta_3 + U_0^2 + V_0^2}$$

该方法为线性方法（$\mathcal{O}(N)$），无需迭代，数值稳定。投影面上圆心和半径的求解为完整 3 参数拟合，不同于常见的先固定投影均值再拟合 2 参数的简化做法。

**投影坐标的缺失**：圆柱在轴线方向无限延伸，投影法无法确定被投影坐标（如 Y 圆柱的 Y 坐标）。该坐标用散点在该方向上的均值填充，不影响曲线重建——因为 `cylinder_geometry` 在求解交线时仅使用截面圆心和半径。

---

## 7. 术语表

| 术语 | 英文 | 说明 |
|---|---|---|
| 轴线方向 | Axis direction | 圆柱中心轴线的方向，平行于 X / Y / Z 之一 |
| 公共坐标 | Common coordinate | 三条坐标轴中不被两圆柱轴线占用的那个坐标 |
| 径向坐标 | Radial coordinate | 圆柱轴线以外的两个坐标（用于表达圆柱方程） |
| 分支 | Branch | 交线因 $\pm\sqrt{\ }$ 符号组合分为 4 段，每段称为一个分支 |
| 分支符号 | Branch sign $(s_1, s_2)$ | 控制径向坐标正/负根号取值的符号对，$s_i \in \{-1, +1\}$ |
| t 参数 | t-parameter | 以公共坐标为自由参数的曲线参数化 |
| 弧长采样 | Arc-length sampling | 以弧长为参数的等距采样，保证点在曲线上均匀分布 |
| 非正交基 | Non-orthogonal basis | 基底向量之间不正交，力分解需解线性方程组 |
| 打磨法向量 | Grinding normal | 两圆柱面法向的角平分线方向，指向打磨/材料移除方向 |
| 垂直向量 | Vertical vector | 即圆柱 2 的表面径向法向量 |
| 力分量 | Force component | 力在切向/法向/垂直方向上的分解分量 |
| 运动趋势向量 | Motion trend vector | 描述力在某一方向上引起的运动趋势的向量 |
| 投影圆拟合 | Projection circle fitting | 将散点投影到圆柱截面平面，用线性最小二乘拟合圆 |
| 投影坐标缺失 | Projected coordinate gap | 圆柱在轴线方向无限延伸，被投影掉的那个坐标无法从拟合中确定 |
| 参数估计层 | Parameter estimation layer | Layer 0，从实测数据反推几何参数，独立于正向几何计算 |
