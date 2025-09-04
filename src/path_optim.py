"""Path optimisation using nonlinear programming.

This module provides a convenience function to optimise the control points
of a lateral offset :math:`e(s)` so that the resulting path minimises the
integrated squared curvature and its derivative while respecting track
boundaries.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# ``path_optim`` can be imported either as part of the ``src`` package or as a
# stand-alone module.  The ``try``/``except`` block below supports both usages
# by attempting a relative import first and falling back to an absolute import
# if that fails.
try:  # pragma: no cover - import shim
    from .path_param import LateralOffsetSpline, path_curvature
except ImportError:  # pragma: no cover - direct execution support
    from path_param import LateralOffsetSpline, path_curvature


def optimise_lateral_offset(
    s: Iterable[float],
    centreline_curvature: Iterable[float],
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    s_control: Sequence[float],
    e_init: Sequence[float] | None = None,
    buffer: float = 0.0,
    method: str = "SLSQP",
):
    """Optimise lateral offset control points for a racing line.

    Parameters
    ----------
    s:
        Arc-length coordinates of the reference centreline where curvature and
        track boundaries are defined.
    centreline_curvature:
        Curvature of the centreline at each value of ``s``.
    left_edge, right_edge:
        ``(N, 2)`` arrays giving Cartesian coordinates of the track edges
        corresponding to ``s``.
    s_control:
        Arc-length positions of the optimisation control points. The lateral
        offset at these points is varied by the optimiser.
    e_init:
        Optional initial guess for the offset values at ``s_control``. If not
        supplied, zeros are used.
    buffer:
        Safety margin subtracted from the track half-width to keep the path
        away from the edges.
    method:
        Optimisation algorithm passed to :func:`scipy.optimize.minimize`.
        Either ``'SLSQP'`` or ``'trust-constr'``.

    Returns
    -------
    LateralOffsetSpline
        Spline representing the optimised lateral offset ``e(s)``.
    """
    s = np.asarray(s, dtype=float)
    kappa_c = np.asarray(centreline_curvature, dtype=float)
    left_edge = np.asarray(left_edge, dtype=float)
    right_edge = np.asarray(right_edge, dtype=float)
    s_control = np.asarray(s_control, dtype=float)

    if s.shape != kappa_c.shape:
        raise ValueError("s and centreline_curvature must have the same shape")
    if left_edge.shape != right_edge.shape or left_edge.shape[0] != s.size:
        raise ValueError("left_edge and right_edge must match shape of s")

    if e_init is None:
        e_init = np.zeros_like(s_control)
    else:
        e_init = np.asarray(e_init, dtype=float)
        if e_init.shape != s_control.shape:
            raise ValueError("e_init must have the same shape as s_control")

    # Track half-width along s.
    half_width = 0.5 * np.linalg.norm(left_edge - right_edge, axis=1)
    upper_bound = half_width - buffer
    lower_bound = -upper_bound

    def objective(e_ctrl: np.ndarray) -> float:
        spline = LateralOffsetSpline(s_control, e_ctrl)
        kappa = path_curvature(s, spline, kappa_c)
        dkappa_ds = np.gradient(kappa, s, edge_order=2)
        integrand = kappa**2 + dkappa_ds**2
        # np.trapz is used for backward compatibility.
        return float(np.trapz(integrand, s))

    if method == "trust-constr":
        def eval_e(e_ctrl: np.ndarray) -> np.ndarray:
            spline = LateralOffsetSpline(s_control, e_ctrl)
            return spline(s)

        constraints = NonlinearConstraint(eval_e, lower_bound, upper_bound)
    else:  # SLSQP style inequality constraints
        def inequality(e_ctrl: np.ndarray) -> np.ndarray:
            spline = LateralOffsetSpline(s_control, e_ctrl)
            e_vals = spline(s)
            return np.hstack((upper_bound - e_vals, e_vals - lower_bound))

        constraints = {"type": "ineq", "fun": inequality}

    result = minimize(objective, e_init, method=method, constraints=constraints)

    if not result.success:
        raise RuntimeError("Optimisation failed: " + result.message)

    return LateralOffsetSpline(s_control, result.x)
