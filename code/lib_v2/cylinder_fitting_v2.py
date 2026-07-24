"""cylinder_fitting_v2.py - Layer 0 v2: point-to-cylinder fitting

Uses CylinderDef(p1, p2, radius) instead of old (axis_char, axis_point, radius).
Same fitting algorithm (projection circle fit), new output format.
p1/p2 span the data range along the axis direction.

Public API:
    fit_cylinders_from_points(pts, axis1, axis2) -> (list[CylinderDef], list[dict])
"""

import numpy as np
from typing import List, Tuple

from cylinder_def import CylinderDef

_AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}


def _fit_projection(pts: np.ndarray, axis: str) -> Tuple[CylinderDef, dict]:
    """Fit a single cylinder via projection circle (linear least squares, 3 params).

    Same algorithm as v1, output as CylinderDef.
    Returns (CylinderDef, details_dict).
    """
    ax_idx = _AXIS_IDX[axis]
    plane_idx = [i for i in range(3) if i != ax_idx]
    u = pts[:, plane_idx[0]]
    v = pts[:, plane_idx[1]]

    # Circle linearization: (u-U0)^2 + (v-V0)^2 = r^2
    Y = u**2 + v**2
    A = np.column_stack([u, v, np.ones_like(u)])
    coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]

    U0 = coeffs[0] / 2
    V0 = coeffs[1] / 2
    R = np.sqrt(coeffs[2] + U0**2 + V0**2)

    # Reconstruct 3D axis point
    axis_point = np.zeros(3)
    axis_point[plane_idx[0]] = U0
    axis_point[plane_idx[1]] = V0
    axis_point[ax_idx] = pts[:, ax_idx].mean()

    residuals = np.sqrt((u - U0)**2 + (v - V0)**2) - R

    # Data range along axis
    t_min = float(pts[:, ax_idx].min())
    t_max = float(pts[:, ax_idx].max())

    cyl = CylinderDef.from_axis_aligned(
        axis=axis,
        axis_point_like=axis_point,
        radius=float(R),
        pts_range=(t_min, t_max),
    )

    details = {
        'rms': float(np.sqrt(np.mean(residuals**2))),
        'max_err': float(np.max(np.abs(residuals))),
        'residuals': residuals,
    }

    return cyl, details


def fit_cylinders_from_points(
    pts: np.ndarray,
    axis1: str,
    axis2: str,
) -> Tuple[List[CylinderDef], List[dict]]:
    """Fit two cylinders from scattered points.

    Parameters
    ----------
    pts : ndarray (N, 3)
    axis1, axis2 : str
        Cylinder axis directions, must differ ('X'|'Y'|'Z').

    Returns
    -------
    (cyls, details)
        cyls[0], cyls[1] — CylinderDef
        details[0], details[1] — {'rms', 'max_err', 'residuals'}
    """
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 must be different.")

    cyl1, d1 = _fit_projection(pts, axis1)
    cyl2, d2 = _fit_projection(pts, axis2)

    return [cyl1, cyl2], [d1, d2]
