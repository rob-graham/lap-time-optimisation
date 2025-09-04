r"""Speed profile solver for a predefined path.

This module computes the maximum speed profile along a path described by an
``s``-grid and curvature :math:`\kappa(s)`.  The solver enforces lateral
acceleration limits via a friction ellipse and accounts for additional
longitudinal constraints such as wheelies and stoppies.  The algorithm performs
iterative forward and backward passes updating the speed until convergence.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def solve_speed_profile(
    s: Iterable[float],
    kappa: Iterable[float],
    mu: float,
    a_wheelie_max: float,
    a_brake: float,
    v_init: Iterable[float] | None = None,
    v_start: float | None = None,
    v_end: float | None = None,
    closed_loop: bool = False,
    g: float = 9.81,
    max_iterations: int = 50,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    r"""Solve for the feasible speed profile along a path.

    Parameters
    ----------
    s:
        Arc-length positions along the path.
    kappa:
        Curvature :math:`\kappa` at each ``s``.
    mu:
        Tyre-road friction coefficient.
    a_wheelie_max:
        Maximum longitudinal acceleration before a wheelie occurs.
    a_brake:
        Maximum braking deceleration before a stoppie occurs.  This also
        defines the longitudinal limit of the friction ellipse in braking.
    v_init:
        Optional initial guess for the speed profile.  If ``None`` a profile
        based on the lateral acceleration limit ``mu * g`` is used.
    v_start, v_end:
        Optional speed constraints at the first and last ``s`` positions.  A
        value of ``None`` leaves the respective boundary unconstrained.
        ``closed_loop`` takes precedence over these parameters.
    closed_loop:
        If ``True`` the path is treated as a closed loop and the solver iterates
        until the initial and final speeds converge to the same value.
    g:
        Gravitational acceleration in ``m/s^2``.
    max_iterations:
        Maximum number of forward/backward passes.
    tol:
        Convergence tolerance on the change in speed between iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        Arrays of speed ``v``, longitudinal acceleration ``ax`` and lateral
        acceleration ``ay`` sampled at ``s`` along with a string array
        ``limit`` describing the active constraint at each sample
        (``"corner"``, ``"accel"``, ``"braking"``, ``"wheelie"`` or
        ``"stoppie"``) and the total ``lap_time`` obtained by integrating
        ``ds / v``.
    """
    s = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if s.shape != kappa.shape:
        raise ValueError("s and kappa must have the same shape")
    if s.ndim != 1:
        raise ValueError("s and kappa must be one-dimensional")
    if np.any(np.diff(s) <= 0):
        raise ValueError("s must be strictly increasing")

    n = s.size
    ds = np.diff(s)
    if closed_loop and n < 2:
        raise ValueError("closed loop requires at least two points")

    mu_g = mu * g
    kappa_abs = np.abs(kappa)
    kappa_mask = kappa_abs > 1e-9
    v_lat = np.full_like(kappa, np.inf)
    v_lat[kappa_mask] = np.sqrt(mu_g / kappa_abs[kappa_mask])

    if v_init is None:
        # Initial guess limited only by lateral grip.
        v = np.full_like(s, 1e3)
        v[kappa_mask] = v_lat[kappa_mask]
    else:
        v = np.asarray(v_init, dtype=float)
        if v.shape != s.shape:
            raise ValueError("v_init must have the same shape as s")
    if not closed_loop:
        if v_start is not None:
            v[0] = float(v_start)
        if v_end is not None:
            v[-1] = float(v_end)

    # Arrays used to record limiting factors during the passes
    limit_forward = np.full(n, "", dtype=object)
    limit_backward = np.full(n, "", dtype=object)

    for _ in range(max_iterations):
        # reset limit trackers for this iteration
        limit_forward.fill("")
        limit_backward.fill("")

        if not closed_loop:
            if v_start is not None:
                v[0] = float(v_start)
            if v_end is not None:
                v[-1] = float(v_end)

        # Enforce lateral acceleration limits before the passes
        v_before = v[kappa_mask].copy()
        v[kappa_mask] = np.minimum(v[kappa_mask], v_lat[kappa_mask])
        limit_forward[kappa_mask] = np.where(
            v[kappa_mask] < v_before, "corner", limit_forward[kappa_mask]
        )

        v_prev = v.copy()
        ay = v**2 * kappa

        # Longitudinal acceleration limits from friction ellipse.
        ay_sq = ay**2
        ax_friction = np.sqrt(np.maximum(mu_g**2 - ay_sq, 0.0))
        ax_max = np.minimum(ax_friction, a_wheelie_max)
        ax_fric_brake = np.sqrt(np.maximum(a_brake**2 - ay_sq, 0.0))
        ax_min = -np.minimum(ax_fric_brake, a_brake)

        # Forward pass (acceleration)
        for i in range(n - 1):
            v_next = np.sqrt(max(v[i] ** 2 + 2.0 * ax_max[i] * ds[i], 0.0))
            limit_type: str
            if kappa_mask[i + 1]:
                v_lat_next = v_lat[i + 1]
                if v_next > v_lat_next:
                    v_next = v_lat_next
                    limit_type = "corner"
                else:
                    if np.isclose(ax_max[i], a_wheelie_max) and a_wheelie_max <= ax_friction[i] + 1e-6:
                        limit_type = "wheelie"
                    elif np.isclose(ax_max[i], ax_friction[i]):
                        limit_type = "corner"
                    else:
                        limit_type = "accel"
            else:
                if np.isclose(ax_max[i], a_wheelie_max) and a_wheelie_max <= ax_friction[i] + 1e-6:
                    limit_type = "wheelie"
                elif np.isclose(ax_max[i], ax_friction[i]):
                    limit_type = "corner"
                else:
                    limit_type = "accel"

            v[i + 1] = v_next
            limit_forward[i + 1] = limit_type

        if not closed_loop and v_end is not None:
            v[-1] = float(v_end)

        # Enforce lateral limits again after forward pass
        v_before = v[kappa_mask].copy()
        v[kappa_mask] = np.minimum(v[kappa_mask], v_lat[kappa_mask])
        limit_forward[kappa_mask] = np.where(
            v[kappa_mask] < v_before, "corner", limit_forward[kappa_mask]
        )

        # Backward pass (braking)
        for i in range(n - 1, 0, -1):
            v_prev_allowed = np.sqrt(
                max(v[i] ** 2 - 2.0 * ax_min[i - 1] * ds[i - 1], 0.0)
            )
            if v_prev_allowed < v[i - 1]:
                # Determine limiting type for this segment
                ax_fric_b = ax_fric_brake[i - 1]
                dec_lim = -ax_min[i - 1]
                if np.isclose(dec_lim, a_brake) and a_brake <= ax_fric_b + 1e-6:
                    limit_type = "stoppie"
                elif np.isclose(dec_lim, ax_fric_b):
                    limit_type = "corner"
                else:
                    limit_type = "braking"
                v[i - 1] = v_prev_allowed
                limit_backward[i - 1] = limit_type

        if closed_loop:
            v_edge = 0.5 * (v[0] + v[-1])
            v[0] = v[-1] = v_edge
            if max(abs(v[0] - v_prev[0]), abs(v[-1] - v_prev[-1])) < tol:
                break
        else:
            if np.max(np.abs(v - v_prev)) < tol:
                break

    ay = v**2 * kappa
    ax = 0.5 * np.gradient(v**2, s, edge_order=2)

    # Combine limiting factors from passes into a single array
    limit = np.empty(n, dtype=object)
    limit[:] = ""
    # Start with forward/backward limits based on sign of ax
    f_mask = ax >= 0
    limit[f_mask] = limit_forward[f_mask]
    b_mask = ~f_mask
    limit[b_mask] = limit_backward[b_mask]

    # Ensure cornering limits are flagged where speed hits lateral bound
    corner_mask = kappa_mask & np.isclose(
        v, v_lat, rtol=1e-3, atol=1e-2
    )
    limit[corner_mask] = "corner"

    # Fill any remaining unspecified entries based on acceleration sign
    unset = limit == ""
    limit[unset & (ax >= 0)] = "accel"
    limit[unset & (ax < 0)] = "braking"
    limit[limit == ""] = "accel"

    v_avg = 0.5 * (v[:-1] + v[1:])
    lap_time = float(np.sum(ds / np.maximum(v_avg, 1e-9)))

    return v, ax, ay, limit, lap_time
