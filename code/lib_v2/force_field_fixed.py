"""
force_field_fixed.py — 力场逆推

inverse(Fn_meas, Fo_meas):
  dn = (F_TARGET - Fn_meas) / KN     ← 法向
  db = -Fo_meas / KO                 ← 复法向
"""
import numpy as np

F_TARGET = -8.0
KN = -24.533
KO = 1.4


def inverse(Fn_meas, Fo_meas, dn_actual=None):
    if Fn_meas >= 0:
        return 0.5, 0.0
    dn = (F_TARGET - Fn_meas) / KN
    db = -Fo_meas / KO
    return dn, db
