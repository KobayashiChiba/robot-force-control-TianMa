"""
force_field_fixed.py — 力场一次模型（常数项 = F_target）

inverse: dn = (F_TARGET - Fn_meas) / kn
  力过大(Fn更负)→F_TARGET-Fn>0→dn负→退出 ✓
  力不够(Fn偏正)→F_TARGET-Fn<0→dn正→压入 ✓
"""
import numpy as np

F_TARGET = -8.0
KN = -24.533
KO = 4.954

def inverse(Fn_meas, Fo_meas):
    if Fn_meas >= 0:
        # 无接触：推力方向的目标搜索偏移，让PID往n方向推
        return 0.5, 0.0
    dn = (F_TARGET - Fn_meas) / KN
    db = 0.0
    return dn, db
