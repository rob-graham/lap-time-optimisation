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
    from . import speed_solver
except ImportError:  # pragma: no cover - direct execution support
    from path_param import LateralOffsetSpline, path_curvature
    import speed_solver


def optimise_lateral_offset(
    s: Iterable[float],
    centreline_curvature: Iterable[float],
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    s_control: Sequence[float],
    e_init: Sequence[float] | None = None,
    buffer: float = 0.0,
    method: str = "SLSQP", # SLSQP (default) or trust-constr
    max_iterations: int | None = None,
    fd_step: float | None = None, # default | None = None,
    path_tol: float = 1e-3,
    cost: str = "curvature",
    mu: float = 1.0,
    a_wheelie_max: float = 9.81,
    a_brake: float = 9.81,
    v_start: float | None = None,
    v_end: float | None = None,
    closed_loop: bool = False,
    g: float = 9.81,
    speed_max_iterations: int = 50,
    speed_tol: float | None = None,
    lap_time_weight: float = 1.0,
) -> tuple[LateralOffsetSpline, int]:
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
        supplied and ``cost='lap_time'``, a preliminary optimisation with
        ``cost='curvature'`` is performed to provide a non-zero warm start;
        otherwise zeros are used.
    buffer:
        Safety margin subtracted from the track half-width to keep the path
        away from the edges.
    method:
        Optimisation algorithm passed to :func:`scipy.optimize.minimize`.
        Either ``'SLSQP'`` or ``'trust-constr'``.
    max_iterations:
        Maximum number of iterations for the optimiser. Passed as
        ``maxiter`` in the ``options`` argument to
        :func:`scipy.optimize.minimize`. If ``None``, SciPy's default is
        used.
    fd_step:
        Step size for the finite-difference gradient approximation passed as
        ``eps`` in the ``options`` argument to
        :func:`scipy.optimize.minimize`. If ``None``, SciPy's default is
        used.
    path_tol:
        Convergence tolerance passed as ``tol`` to
        :func:`scipy.optimize.minimize`.
    cost:
        Objective to minimise. ``"curvature"`` minimises the integral of the
        squared curvature and its derivative. ``"lap_time"`` minimises the
        lap time computed by :func:`speed_solver.solve_speed_profile`.
    lap_time_weight:
        Multiplicative factor applied to the lap time when ``cost='lap_time'``.
    mu, a_wheelie_max, a_brake, v_start, v_end, closed_loop, g:
        Parameters forwarded to :func:`speed_solver.solve_speed_profile` when
        ``cost='lap_time'``.
    speed_max_iterations:
        Maximum iterations for the speed profile solver. Reducing this value
        yields faster but potentially less accurate lap time estimates.
    speed_tol:
        Convergence tolerance for the speed profile solver. A looser tolerance
        speeds up evaluation at the expense of precision. If ``None`` the
        solver's default tolerance is used.

    Returns
    -------
    (LateralOffsetSpline, int)
        Spline representing the optimised lateral offset ``e(s)`` and the
        number of iterations performed by the optimiser.
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

    if cost not in {"curvature", "lap_time"}:
        raise ValueError("cost must be 'curvature' or 'lap_time'")

    if e_init is None:
        if cost == "lap_time":
            warm_start, _ = optimise_lateral_offset(
                s,
                kappa_c,
                left_edge,
                right_edge,
                s_control,
                buffer=buffer,
                method=method,
                max_iterations=max_iterations,
                path_tol=path_tol,
                cost="curvature",
            )
            e_init = warm_start.e_control
        else:
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
        if cost == "curvature":
            dkappa_ds = np.gradient(kappa, s, edge_order=2)
            integrand = kappa**2 + dkappa_ds**2
            # np.trapz is used for backward compatibility.
            return float(np.trapz(integrand, s))
        else:  # cost == "lap_time"
            solve_kwargs = {
                "v_start": v_start,
                "v_end": v_end,
                "closed_loop": closed_loop,
                "g": g,
                "max_iterations": speed_max_iterations,
            }
            if speed_tol is not None:
                solve_kwargs["tol"] = speed_tol
            _, _, _, _, lap_time, _, _ = speed_solver.solve_speed_profile(
                s,
                kappa,
                mu,
                a_wheelie_max,
                a_brake,
                **solve_kwargs,
            )
            return float(lap_time_weight * lap_time)

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

    options = {}
    if max_iterations is not None:
        options["maxiter"] = max_iterations
    if fd_step is not None:
        options["eps"] = fd_step
    if not options:
        options = None

    result = minimize(
        objective,
        e_init,
        method=method,
        constraints=constraints,
        options=options,
        tol=path_tol,
    )

    if not result.success:
        raise RuntimeError("Optimisation failed: " + result.message)

    return LateralOffsetSpline(s_control, result.x), int(result.nit)
