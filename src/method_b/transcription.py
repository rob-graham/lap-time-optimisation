from __future__ import annotations

"""Temporal grid generation and trapezoidal collocation utilities."""

from typing import Tuple

import numpy as np
import casadi as ca

from .ocp import OCP


def create_grid(
    s_start: float,
    s_end: float,
    *,
    n_points: int | None = None,
    ds: float | None = None,
) -> np.ndarray:
    """Return a monotonically increasing grid between ``s_start`` and ``s_end``.

    The grid can either be specified by the number of points ``n_points`` or a
    constant spacing ``ds``.  Exactly one of the two arguments must be given.
    The end point is always included which means that the actual spacing may be
    slightly smaller than ``ds`` when ``(s_end - s_start)`` is not an integer
    multiple of it.
    """

    if (n_points is None) == (ds is None):
        raise ValueError("Specify exactly one of n_points or ds")

    if ds is not None:
        if ds <= 0:
            raise ValueError("ds must be positive")
        n_points = int(np.floor((s_end - s_start) / ds)) + 1
        grid = s_start + ds * np.arange(n_points)
        if grid[-1] < s_end:
            grid = np.append(grid, s_end)
        else:
            grid[-1] = s_end
        return grid

    assert n_points is not None  # for mypy
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    return np.linspace(s_start, s_end, n_points)


def state_control_vectors(ocp: OCP, grid: np.ndarray) -> Tuple[ca.SX, ca.SX]:
    """Create symbolic state and control matrices for the given grid."""

    n_nodes = grid.size
    x = ca.SX.sym("x", ocp.n_x, n_nodes)
    u = ca.SX.sym("u", ocp.n_u, n_nodes)
    return x, u


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

    x, u = state_control_vectors(ocp, grid)
    ds = np.diff(grid)

    defects = []
    for k, h in enumerate(ds):
        xk = x[:, k]
        xk1 = x[:, k + 1]
        uk = u[:, k]
        uk1 = u[:, k + 1]
        fk = ocp.dynamics(xk, uk, k)
        fk1 = ocp.dynamics(xk1, uk1, k + 1)
        defect = xk1 - xk - 0.5 * h * (fk + fk1)
        defects.append(defect)

    g_defect = ca.vertcat(*defects) if defects else ca.SX.zeros(0)
    return x, u, g_defect


def decision_variables(x: ca.SX, u: ca.SX) -> ca.SX:
    """Stack the state and control matrices into a single decision vector."""

    return ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))


def extract_unscaled_solution(ocp: OCP, z: np.ndarray, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape and unscale IPOPT solution vector.

    Parameters
    ----------
    ocp:
        Problem definition providing scaling information.
    z:
        Optimiser decision vector consisting of stacked state and control
        variables in *scaled* form.
    n_nodes:
        Number of grid nodes used in the transcription.

    Returns
    -------
    (x, u):
        Tuple of unscaled state and control trajectories.
    """

    n_x, n_u = ocp.n_x, ocp.n_u
    n_dec_x = n_x * n_nodes
    x_scaled = z[:n_dec_x].reshape(n_x, n_nodes)
    u = z[n_dec_x : n_dec_x + n_u * n_nodes].reshape(n_u, n_nodes)
    x = ocp.unscale_x_array(x_scaled)
    return x, u


def assemble_constraints(*cons: ca.SX) -> ca.SX:
    """Vertically stack constraint vectors while handling empty inputs."""

    cons = [c for c in cons if c.size1() > 0]
    return ca.vertcat(*cons) if cons else ca.SX.zeros(0)


__all__ = [
    "create_grid",
    "state_control_vectors",
    "trapezoidal_collocation",
    "decision_variables",
    "assemble_constraints",
    "extract_unscaled_solution",
]
