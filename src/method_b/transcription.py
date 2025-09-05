from __future__ import annotations

"""Temporal grid generation and trapezoidal collocation utilities."""

from typing import Tuple

import numpy as np
import casadi as ca

from .ocp import OCP


def create_grid(s_start: float, s_end: float, n_points: int) -> np.ndarray:
    """Return an evenly spaced grid between ``s_start`` and ``s_end``."""

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    return np.linspace(s_start, s_end, n_points)


def trapezoidal_collocation(ocp: OCP, grid: np.ndarray) -> Tuple[ca.SX, ca.SX, ca.SX]:
    """Construct defect constraints for trapezoidal transcription.

    Parameters
    ----------
    ocp:
        Problem definition providing dynamics and sizes.
    grid:
        Array of monotonically increasing grid points.

    Returns
    -------
    (x, u, g_defect)
        Symbolic decision variables for the states ``x`` and controls ``u`` as
        well as the stacked defect constraint expressions ``g_defect`` enforcing
        the dynamics via the trapezoidal rule.
    """

    n_x, n_u = ocp.n_x, ocp.n_u
    N = grid.size - 1

    x = ca.SX.sym("x", n_x, N + 1)
    u = ca.SX.sym("u", n_u, N + 1)

    defects = []
    for k in range(N):
        h = grid[k + 1] - grid[k]
        xk = x[:, k]
        xk1 = x[:, k + 1]
        uk = u[:, k]
        uk1 = u[:, k + 1]
        fk = ocp.dynamics(xk, uk)
        fk1 = ocp.dynamics(xk1, uk1)
        defect = xk1 - xk - 0.5 * h * (fk + fk1)
        defects.append(defect)

    g_defect = ca.vertcat(*defects) if defects else ca.SX.zeros(0)
    return x, u, g_defect
