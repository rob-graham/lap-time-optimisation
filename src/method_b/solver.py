from __future__ import annotations

"""CasADi-based nonlinear programme builder and solver for method B."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import casadi as ca

from .ocp import OCP
from .transcription import (
    create_grid,
    trapezoidal_collocation,
    decision_variables,
    assemble_constraints,
    extract_unscaled_solution,
)


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
    *,
    closed_loop: bool,
    slack: bool = False,
) -> tuple[Dict[str, ca.SX], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Assemble the CasADi NLP expression and accompanying bounds."""

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
    z = decision_variables(x, u)

    # Bounds
    x_min, x_max, u_min, u_max, g_min, g_max = ocp.bounds()
    lbx = np.concatenate([np.tile(x_min, n_nodes), np.tile(u_min, n_nodes)])
    ubx = np.concatenate([np.tile(x_max, n_nodes), np.tile(u_max, n_nodes)])
    lbg_defect = np.zeros(n_x * N)
    ubg_defect = np.zeros(n_x * N)
    lbg_path = np.tile(g_min, n_nodes)
    ubg_path = np.tile(g_max, n_nodes)

    g_list = [g_defect]
    lbg_list = [lbg_defect]
    ubg_list = [ubg_defect]

    if slack and g_path.size1() > 0:
        s = ca.SX.sym("s", g_path.size1())
        z = ca.vertcat(z, s)
        g_list.append(g_path - s)
        lbx = np.concatenate([lbx, np.zeros(s.size1())])
        ubx = np.concatenate([ubx, np.full(s.size1(), np.inf)])
        lbg_list.append(lbg_path)
        ubg_list.append(ubg_path)
        J += 1e6 * ca.sumsqr(s)
    else:
        g_list.append(g_path)
        lbg_list.append(lbg_path)
        ubg_list.append(ubg_path)

    if closed_loop:
        g_bc = x[:, 0] - x[:, -1]
        g_list.append(g_bc)
        zeros = np.zeros(n_x)
        lbg_list.append(zeros)
        ubg_list.append(zeros)
    else:
        lbx[:n_x] = 0.0
        ubx[:n_x] = 0.0

    g = assemble_constraints(*g_list)
    lbg = np.concatenate(lbg_list)
    ubg = np.concatenate(ubg_list)

    nlp = {"x": z, "f": J, "g": g}
    return nlp, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u


def _latest_method_a_results() -> Optional[Path]:
    """Return path to the latest Method A ``results.csv`` if available."""

    out_dir = Path("outputs")
    if not out_dir.is_dir():
        return None

    candidates = [d for d in out_dir.iterdir() if d.is_dir()]
    for directory in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        csv_path = directory / "method_a" / "results.csv"
        if csv_path.exists():
            return csv_path
    return None


def _centreline_initialisation(ocp: OCP, n_nodes: int) -> Dict[str, np.ndarray]:
    """Initial guess corresponding to the reference centreline."""

    x_guess = np.zeros((ocp.n_x, n_nodes))
    x_guess[3, :] = ocp.kappa_c
    x_guess = ocp.scale_x_array(x_guess)
    u_guess = np.zeros((ocp.n_u, n_nodes))
    x0 = np.concatenate([x_guess.reshape(-1), u_guess.reshape(-1)])
    return {"x0": x0}


def _load_method_a_results(ocp: OCP, grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Load warm-start data from the latest Method A CSV output."""

    csv_path = _latest_method_a_results()
    if csv_path is None:
        return {}

    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
    except OSError:
        return {}

    required = {"e", "kappa", "v", "psi"}
    if not required.issubset(data.dtype.names or {}):
        return {}

    e = np.asarray(data["e"], dtype=float)
    kappa = np.asarray(data["kappa"], dtype=float)
    v = np.asarray(data["v"], dtype=float)
    psi = np.asarray(data["psi"], dtype=float)

    if e.size != grid.size:
        return {}

    x_guess = np.vstack([e, psi, v, kappa])
    u_kappa = np.gradient(kappa, grid, edge_order=2)
    dv_ds = np.gradient(v, grid, edge_order=2)
    a_x = v * dv_ds
    u_guess = np.vstack([u_kappa, a_x])

    x_guess = ocp.scale_x_array(x_guess)
    x0 = np.concatenate([x_guess.reshape(-1), u_guess.reshape(-1)])
    return {"x0": x0}


# ---------------------------------------------------------------------------
# Public solve function
# ---------------------------------------------------------------------------

def solve(
    ocp: OCP,
    s_start: float,
    s_end: float,
    n_points: int,
    *,
    closed_loop: bool = True,
    warm_start_from_method_a: bool = True,
    use_slacks: bool = False,
    auto_slack_retry: bool = True,
    tol: float = 1e-8,
    print_level: int = 0,
    linear_solver: str = "mumps",
    ipopt_opts: Optional[Dict[str, Any]] = None,
) -> SolverResult:
    """Solve the OCP using trapezoidal collocation and IPOPT.

    Parameters
    ----------
    ocp:
        Problem definition.
    s_start, s_end, n_points:
        Parameters passed to :func:`create_grid`.
    warm_start_from_method_a:
        If ``True`` attempt to initialise the solver from the latest Method A
        outputs.  When ``False`` a simple centreline initialisation is used.
    use_slacks:
        If ``True`` slack variables are included from the start.
    auto_slack_retry:
        If ``True`` the optimisation is automatically retried with slack
        variables when the first attempt fails.
    tol, print_level, linear_solver:
        IPOPT options controlling tolerance, verbosity and the linear solver.
    ipopt_opts:
        Additional IPOPT options overriding the defaults.
    """

    grid = create_grid(s_start, s_end, n_points=n_points)
    nlp, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u = _build_nlp(
        ocp, grid, closed_loop=closed_loop, slack=use_slacks
    )

    opts: Dict[str, Any] = {
        "print_time": False,
        "ipopt.tol": tol,
        "ipopt.print_level": print_level,
        "ipopt.linear_solver": linear_solver,
    }
    if ipopt_opts:
        opts.update(ipopt_opts)

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    if warm_start_from_method_a:
        warm = _load_method_a_results(ocp, grid)
    else:
        warm = {}
    if not warm:
        warm = _centreline_initialisation(ocp, n_nodes)

    x0 = warm.get("x0")
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
    stats = solver.stats()

    if (use_slacks or auto_slack_retry) and stats.get("return_status") not in {"Solve_Succeeded", "Solved_Succeeded"}:
        nlp, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u = _build_nlp(
            ocp, grid, closed_loop=closed_loop, slack=True
        )
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
        stats = solver.stats()

    z = np.array(sol["x"]).reshape(-1)
    x_sol, u_sol = extract_unscaled_solution(ocp, z, n_nodes)
    return SolverResult(x=x_sol, u=u_sol, grid=grid, stats=stats)
