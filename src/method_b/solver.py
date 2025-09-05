from __future__ import annotations

"""CasADi-based nonlinear programme builder and solver for method B."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import casadi as ca

from .ocp import OCP
from .transcription import create_grid, trapezoidal_collocation


@dataclass
class SolverResult:
    """Container for optimisation results."""

    x: np.ndarray
    """State trajectory of shape ``(n_x, N)``."""

    u: np.ndarray
    """Control trajectory of shape ``(n_u, N)``."""

    grid: np.ndarray
    """Grid associated with the solution."""

    stats: Dict[str, Any]
    """Solver statistics as reported by CasADi."""


# ---------------------------------------------------------------------------
# NLP construction utilities
# ---------------------------------------------------------------------------

def _build_nlp(
    ocp: OCP,
    grid: np.ndarray,
    slack: bool = False,
    ipopt_opts: Optional[Dict[str, Any]] = None,
) -> tuple[ca.Function, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Assemble the CasADi NLP and return solver and bounds."""

    n_x, n_u = ocp.n_x, ocp.n_u
    x, u, g_defect = trapezoidal_collocation(ocp, grid)
    n_nodes = grid.size
    N = n_nodes - 1

    # Path constraints at each node
    g_path = [ocp.path_constraints(x[:, k], u[:, k]) for k in range(n_nodes)]
    g_path = ca.vertcat(*g_path) if g_path else ca.SX.zeros(0)

    # Objective (trapezoidal integration of stage cost)
    J = 0
    for k in range(N):
        h = grid[k + 1] - grid[k]
        J += 0.5 * h * (
            ocp.stage_cost(x[:, k], u[:, k]) + ocp.stage_cost(x[:, k + 1], u[:, k + 1])
        )

    # Decision variables
    z = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))

    # Bounds
    x_min, x_max, u_min, u_max, g_min, g_max = ocp.bounds()
    lbx = np.concatenate([np.tile(x_min, n_nodes), np.tile(u_min, n_nodes)])
    ubx = np.concatenate([np.tile(x_max, n_nodes), np.tile(u_max, n_nodes)])
    lbg_defect = np.zeros(n_x * N)
    ubg_defect = np.zeros(n_x * N)
    g_dim = len(g_min)
    lbg_path = np.tile(g_min, n_nodes)
    ubg_path = np.tile(g_max, n_nodes)

    g = ca.vertcat(g_defect, g_path)
    lbg = np.concatenate([lbg_defect, lbg_path])
    ubg = np.concatenate([ubg_defect, ubg_path])

    if slack and g_path.size1() > 0:
        s = ca.SX.sym("s", g_path.size1())
        z = ca.vertcat(z, s)
        g = ca.vertcat(g_defect, g_path - s)
        lbx = np.concatenate([lbx, np.zeros(s.size1())])
        ubx = np.concatenate([ubx, np.full(s.size1(), np.inf)])
        lbg = np.concatenate([lbg_defect, lbg_path])
        ubg = np.concatenate([ubg_defect, ubg_path])
        J += 1e6 * ca.sumsqr(s)

    nlp = {"x": z, "f": J, "g": g}

    opts = {"ipopt.print_level": 0, "print_time": False}
    if ipopt_opts:
        opts.update(ipopt_opts)

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    return solver, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u


# ---------------------------------------------------------------------------
# Public solve function
# ---------------------------------------------------------------------------

def solve(
    ocp: OCP,
    s_start: float,
    s_end: float,
    n_points: int,
    warm_start: Optional[Dict[str, np.ndarray]] = None,
    slack_retry: bool = True,
    ipopt_opts: Optional[Dict[str, Any]] = None,
) -> SolverResult:
    """Solve the OCP using trapezoidal collocation and IPOPT.

    Parameters
    ----------
    ocp:
        Problem definition.
    s_start, s_end, n_points:
        Parameters passed to :func:`create_grid`.
    warm_start:
        Optional dictionary containing ``x0`` and ``lam_g0`` initial guesses for
        the IPOPT solver.
    slack_retry:
        If ``True`` the optimisation is automatically retried with slack
        variables added to the path constraints when the first attempt fails.
    ipopt_opts:
        Additional IPOPT options overriding the defaults.
    """

    grid = create_grid(s_start, s_end, n_points)
    solver, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u = _build_nlp(
        ocp, grid, slack=False, ipopt_opts=ipopt_opts
    )

    x0 = warm_start.get("x0") if warm_start else None
    lam_g0 = warm_start.get("lam_g0") if warm_start else None
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0, lam_g0=lam_g0)
    stats = solver.stats()

    if slack_retry and stats.get("return_status") not in {"Solve_Succeeded", "Solved_Succeeded"}:
        solver, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u = _build_nlp(
            ocp, grid, slack=True, ipopt_opts=ipopt_opts
        )
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()

    z = np.array(sol["x"]).reshape(-1)
    n_dec_x = n_x * n_nodes
    x_sol = z[:n_dec_x].reshape(n_x, n_nodes)
    u_sol = z[n_dec_x : n_dec_x + n_u * n_nodes].reshape(n_u, n_nodes)
    return SolverResult(x=x_sol, u=u_sol, grid=grid, stats=stats)
