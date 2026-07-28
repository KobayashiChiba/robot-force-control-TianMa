"""
force_field_quadratic.py — 锚定二次逆推: (Fn, Fo) → (dn, db)

模型: dn/db 各为 Fn,Fo 的二次函数（6参数），消去常数项使 (-8,0)→(0,0) 精确成立。
基于 20 位置 × 21×21 网格采样（4779 有效点）最小二乘拟合。

用法:
    from force_field_quadratic import inverse
    dn, db = inverse(Fn_meas, Fo_meas)
"""
import numpy as np

# ── 锚定二次拟合系数 ──
# 约束: (Fn=-8, Fo=0) → (dn=0, db=0) 硬锚定
# dn = c0 + c1*Fn + c2*Fo + c3*Fn² + c4*Fn*Fo + c5*Fo²
_COEF_DN = np.array([-0.496811, -0.055361,  0.081592,  0.000843,  0.003985,  0.003034])
_COEF_DB = np.array([ 0.540874,  0.075355,  0.447720,  0.000968,  0.010277, -0.003157])

# 全局拟合精度（锚定前）
#   dn: MAE=0.120mm  RMSE=0.152mm  PC95=0.298mm
#   db: MAE=0.195mm  RMSE=0.306mm  PC95=0.680mm
# 锚定代价: dn_PC95 ↑0.10mm, db 几乎不变
#   强接触区 (|Fn|≥10N): dn误差~0.03mm, db误差~0.04mm
#   弱接触区 (|Fn|<5N): dn偏小~0.2mm（保守退刀），db偏移~0.3mm


def inverse(Fn_meas, Fo_meas):
    """从测量力反推偏移量（锚定二次模型）。

    Args:
        Fn_meas: 有符号法向力 (N)，负值=压入
        Fo_meas: 有符号复法向力 (N)

    Returns:
        (dn, db): 偏移量 (mm)，相对参考轨迹
          dn>0 → 压入更深
          db>0 → o 正方向偏移

    边界处理:
        - Fn ≥ 0 → 自动截为 0（拉力/无力时不反推退刀）
        - |Fn| > 30N, |Fo| > 10N → 截断（防外推）
    """
    # Fn≥0 → 截零（无接触时仍需向曲线移动）
    if Fn_meas >= 0:
        Fn_meas = 0.0

    # 防外推截断
    Fn_c = max(-30.0, min(0.0, Fn_meas))
    Fo_c = max(-10.0, min(10.0, Fo_meas))

    x = np.array([1.0, Fn_c, Fo_c, Fn_c**2, Fn_c*Fo_c, Fo_c**2])
    dn = float(x @ _COEF_DN)
    db = float(x @ _COEF_DB)
    return dn, db


def predict(dn, db):
    """正向预测（近似）: (dn, db) → (Fn, Fo)。仅供调试，非逆推链路使用。"""
    # 这是简化线性近似，精确预测需 sphere_contact_force
    # dn=0 时基准力 ≈ -8N
    Fn_approx = -8.0 + _COEF_DN[1] * dn + _COEF_DN[2] * db  # 忽略二次项做粗略估计
    Fo_approx = _COEF_DB[1] * dn + _COEF_DB[2] * db
    return Fn_approx, Fo_approx


# ── 自测 ──
if __name__ == '__main__':
    tests = [
        (-8.0,  0.0,  "(-8,0) 目标"),
        ( 0.0,  0.0,  "(0,0) 无力→截0"),
        ( 3.0,  0.0,  "(+3,0) 拉力→截0"),
        (-5.0,  0.0,  "(-5,0)"),
        (-10.0, 0.0,  "(-10,0)"),
        (-8.0,  2.0,  "(-8,+2)"),
        (-8.0, -2.0,  "(-8,-2)"),
    ]
    print(f"{'工况':>12}  {'Fn':>7} {'Fo':>7}  →  {'dn':>8} {'db':>8}")
    print(f"{'':>12}  {'──':>7} {'──':>7}     {'──':>8} {'──':>8}")
    for fn, fo, label in tests:
        dn, db = inverse(fn, fo)
        print(f"{label:>12}  {fn:7.1f} {fo:7.1f}  →  {dn:8.4f} {db:8.4f}")

    # 验证锚定点
    dn, db = inverse(-8.0, 0.0)
    assert abs(dn) < 1e-4 and abs(db) < 1e-4, f"锚定点不精确: ({dn},{db})"
    print("\n✓ 锚定点 (-8,0)→(0,0) 精确成立")
