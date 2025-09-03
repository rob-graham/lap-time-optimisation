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
    g: float = 9.81,
    max_iterations: int = 50,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    g:
        Gravitational acceleration in ``m/s^2``.
    max_iterations:
        Maximum number of forward/backward passes.
    tol:
        Convergence tolerance on the change in speed between iterations.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of speed ``v``, longitudinal acceleration ``ax`` and lateral
        acceleration ``ay`` sampled at ``s``.
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

    if v_init is None:
        # Initial guess limited only by lateral grip.
        v = np.empty_like(s)
        mask = np.abs(kappa) > 1e-9
        v[mask] = np.sqrt(mu * g / np.abs(kappa[mask]))
        v[~mask] = 1e3
    else:
        v = np.asarray(v_init, dtype=float)
        if v.shape != s.shape:
            raise ValueError("v_init must have the same shape as s")
    # Enforce boundary conditions of starting and ending at rest.
    v[0] = 0.0
    v[-1] = 0.0

    mu_g = mu * g
    for _ in range(max_iterations):
        v_prev = v.copy()
        ay = v**2 * kappa

        # Longitudinal acceleration limits from friction ellipse.
        ay_sq = ay**2
        ax_friction = np.sqrt(np.maximum(mu_g**2 - ay_sq, 0.0))
        ax_max = np.minimum(ax_friction, a_wheelie_max)
        ax_min = -np.minimum(np.sqrt(np.maximum(a_brake**2 - ay_sq, 0.0)), a_brake)

        # Forward pass (acceleration)
        for i in range(n - 1):
            v_next = np.sqrt(max(v[i] ** 2 + 2.0 * ax_max[i] * ds[i], 0.0))
            if v_next < v[i + 1]:
                v[i + 1] = v_next
        # Backward pass (braking)
        for i in range(n - 1, 0, -1):
            v_prev_allowed = np.sqrt(
                max(v[i] ** 2 - 2.0 * ax_min[i - 1] * ds[i - 1], 0.0)
            )
            if v_prev_allowed < v[i - 1]:
                v[i - 1] = v_prev_allowed

        if np.max(np.abs(v - v_prev)) < tol:
            break

    ay = v**2 * kappa
    ax = 0.5 * np.gradient(v**2, s, edge_order=2)
    return v, ax, ay
