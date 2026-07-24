"""
cylinder_geometry.py — 双圆柱相交几何计算与均匀弧长采样 (Layer 1)

对外接口 (仅 2 个):
    sample_intersection()  — 输入圆柱参数 + 采样点数 → 返回 Geom 实例
    resample_curve()       — 输入 Geom + 新采样点数 → 返回新的 Geom 实例

数据类:
    Geom  — 仅包含采样结果和圆柱原始参数, 无内部隐藏字段

适用条件:
  - 两个圆柱轴线分别平行于 X / Y / Z 轴之一, 且方向不同
"""

import numpy as np
from dataclasses import dataclass

# 轴线字符 → 坐标索引
_AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}
_AXIS_DIR = {'X': np.array([1., 0., 0.]),
             'Y': np.array([0., 1., 0.]),
             'Z': np.array([0., 0., 1.])}


# ============================================================
# 数据类
# ============================================================

@dataclass
class Geom:
    """两圆柱相交曲线的均匀弧长采样结果.

    字段
    ----
    n_samples  : int               — 采样点数量
    sample_pts : ndarray (N, 3)    — 均匀弧长采样点坐标
    axis1, axis2 : str             — 圆柱轴线方向 'X'|'Y'|'Z'
    c1, c2       : tuple          — 圆柱轴心坐标 (cx, cy, cz)
    r1, r2       : float           — 圆柱半径
    """
    n_samples:  int
    sample_pts: np.ndarray
    axis1: str
    c1:    tuple
    r1:    float
    axis2: str
    c2:    tuple
    r2:    float


# ============================================================
# 对外接口 1
# ============================================================

def sample_intersection(axis1, c1, r1, axis2, c2, r2, n_samples=1000, N_curve=250):
    """
    输入两圆柱参数和采样点总数, 返回均匀弧长采样的 Geom 实例.

    参数
    ----
    axis1, axis2 : str
        两圆柱轴线方向, 必须不同 ('X'|'Y'|'Z').
    c1, c2 : tuple (cx, cy, cz)
        两圆柱轴线上一点坐标.
    r1, r2 : float
        两圆柱半径.
    n_samples : int
        沿相交曲线均匀采样的点数 (默认 1000).
    N_curve : int
        每段分支内部离散点数 (默认 250, 影响采样精度).
    """
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 must be different.")

    raw = _intersect_raw(axis1, c1, r1, axis2, c2, r2, N_curve)
    closed_pts, pt_src = _build_closed_curve(raw['curves'])
    sample_pts, _ = _sample_uniform(closed_pts, pt_src, n_samples)

    return Geom(
        n_samples  = n_samples,
        sample_pts = sample_pts,
        axis1 = axis1, c1 = c1, r1 = r1,
        axis2 = axis2, c2 = c2, r2 = r2,
    )


# ============================================================
# 对外接口 2
# ============================================================

def resample_curve(geom, n_samples):
    """
    对已有 Geom 用新采样点数重新做均匀弧长采样.

    参数
    ----
    geom : Geom
    n_samples : int

    返回
    ----
    Geom
    """
    return sample_intersection(
        geom.axis1, geom.c1, geom.r1,
        geom.axis2, geom.c2, geom.r2,
        n_samples,
    )


# ============================================================
# 内部函数
# ============================================================

def _get_branch_meta(axis1, axis2, c1, c2, r1, r2, N):
    """获取分支元信息: 坐标映射、t参数范围、分支符号表 (不计算完整曲线几何)."""
    ai1 = _AXIS_IDX[axis1]
    ai2 = _AXIS_IDX[axis2]
    common = ({0, 1, 2} - {ai1, ai2}).pop()
    other1 = ai2
    other2 = ai1

    t_min = max(c1[common] - r1, c2[common] - r2)
    t_max = min(c1[common] + r1, c2[common] + r2)
    if t_min >= t_max:
        raise ValueError(f"No intersection: common coord range empty [{t_min:.2f}, {t_max:.2f}].")

    t_up   = np.linspace(t_min, t_max, N)
    t_down = np.linspace(t_max, t_min, N)

    branch_info = [
        (+1, +1, True),
        (+1, -1, False),
        (-1, -1, True),
        (-1, +1, False),
    ]
    curves_t = [t_up, t_down, t_up, t_down]

    return {
        'common':      common,
        'other1':      other1,
        'other2':      other2,
        'c1':          c1,
        'c2':          c2,
        'curves_t':    curves_t,
        'branch_info': branch_info,
    }


def _intersect_raw(axis1, c1, r1, axis2, c2, r2, N):
    """求解原始4段分支曲线 (内部)."""
    meta = _get_branch_meta(axis1, axis2, c1, c2, r1, r2, N)
    common = meta['common']
    other1 = meta['other1']
    other2 = meta['other2']
    c1 = meta['c1']
    c2 = meta['c2']
    t_up   = meta['curves_t'][0]
    t_down = meta['curves_t'][1]

    # 预计算差值, 避免重复平方/开方
    d1_up   = t_up   - c1[common]
    d1_down = t_down - c1[common]
    d2_up   = t_up   - c2[common]
    d2_down = t_down - c2[common]

    s1_up   = np.sqrt(r1**2 - d1_up**2)
    s1_down = np.sqrt(r1**2 - d1_down**2)
    s2_up   = np.sqrt(r2**2 - d2_up**2)
    s2_down = np.sqrt(r2**2 - d2_down**2)

    p1_up   = c1[other1] + s1_up
    p1_down = c1[other1] + s1_down
    n1_up   = c1[other1] - s1_up
    n1_down = c1[other1] - s1_down

    p2_up   = c2[other2] + s2_up
    p2_down = c2[other2] + s2_down
    n2_up   = c2[other2] - s2_up
    n2_down = c2[other2] - s2_down

    branches = [
        (p1_up,   p2_up,   t_up),
        (p1_down, n2_down, t_down),
        (n1_up,   n2_up,   t_up),
        (n1_down, p2_down, t_down),
    ]

    curves = []
    curves_t = []
    for v1, v2, t_vals in branches:
        pt = [None, None, None]
        pt[common] = t_vals
        pt[other1] = v1
        pt[other2] = v2
        curves.append((pt[0], pt[1], pt[2]))
        curves_t.append(t_vals)

    return {**meta, 'curves': curves}


def _build_closed_curve(curves):
    """将4段分支按连接顺序 [0,3,2,1] 拼接成闭合点列 (内部)."""
    branch_order = [0, 3, 2, 1]
    closed_pts = []
    pt_src = []

    for i, bid in enumerate(branch_order):
        xs, ys, zs = curves[bid]
        start = 0 if i == 0 else 1
        for j in range(start, len(xs)):
            closed_pts.append([xs[j], ys[j], zs[j]])
            pt_src.append((bid, j))

    return np.array(closed_pts), pt_src


def _sample_uniform(closed_pts, pt_src, n_samples):
    """对闭合点列做均匀弧长采样 (内部)."""
    M = len(closed_pts)

    diffs = np.diff(closed_pts, axis=0)
    chord_lens = np.linalg.norm(diffs, axis=1)
    chord_lens = np.append(chord_lens, np.linalg.norm(closed_pts[0] - closed_pts[-1]))
    cum_len = np.cumsum(np.insert(chord_lens, 0, 0.0))
    total_len = cum_len[-1]

    target_lens = np.linspace(0, total_len, n_samples, endpoint=False)
    indices = [np.argmin(np.abs(cum_len[:M] - t)) for t in target_lens]

    sample_pts = closed_pts[indices]
    sample_src = [pt_src[i] for i in indices]
    return sample_pts, sample_src
