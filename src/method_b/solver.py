from __future__ import annotations

"""CasADi-based nonlinear programme builder and solver for method B."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import casadi as ca

from .ocp import OCP
from .transcription import (
    create_grid,
    trapezoidal_collocation,
    decision_variables,
    assemble_constraints,
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

    g = assemble_constraints(g_defect, g_path)
    lbg = np.concatenate([lbg_defect, lbg_path])
    ubg = np.concatenate([ubg_defect, ubg_path])

    if slack and g_path.size1() > 0:
        s = ca.SX.sym("s", g_path.size1())
        z = ca.vertcat(z, s)
        g = assemble_constraints(g_defect, g_path - s)
        lbx = np.concatenate([lbx, np.zeros(s.size1())])
        ubx = np.concatenate([ubx, np.full(s.size1(), np.inf)])
        lbg = np.concatenate([lbg_defect, lbg_path])
        ubg = np.concatenate([ubg_defect, ubg_path])
        J += 1e6 * ca.sumsqr(s)

    nlp = {"x": z, "f": J, "g": g}
    return nlp, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u


def _load_method_a_warm_start(
    source: Union[str, Path, Dict[str, Any]],
    n_x: int,
    n_u: int,
    n_nodes: int,
) -> Dict[str, np.ndarray]:
    """Load warm-start data produced by Method A.

    ``source`` may either be a mapping with the expected arrays or the path to
    an ``.npz`` file containing them.  Only the keys present are used.  Arrays
    ``x`` and ``u`` are reshaped and stacked to form an ``x0`` vector compatible
    with the NLP decision variables.
    """

    if isinstance(source, (str, bytes, Path)):
        data = np.load(source, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            data_dict = {k: data[k] for k in data.files}
        else:  # pragma: no cover - defensive branch
            data_dict = dict(data)
    else:
        data_dict = dict(source)

    warm: Dict[str, np.ndarray] = {}
    x_guess = data_dict.get("x")
    u_guess = data_dict.get("u")
    if x_guess is not None and u_guess is not None:
        x_guess = np.asarray(x_guess).reshape(n_x, n_nodes)
        u_guess = np.asarray(u_guess).reshape(n_u, n_nodes)
        warm["x0"] = np.concatenate([x_guess.reshape(-1), u_guess.reshape(-1)])

    lam_g = data_dict.get("lam_g") or data_dict.get("lam_g0")
    if lam_g is not None:
        warm["lam_g0"] = np.asarray(lam_g).reshape(-1)

    return warm


# ---------------------------------------------------------------------------
# Public solve function
# ---------------------------------------------------------------------------

def solve(
    ocp: OCP,
    s_start: float,
    s_end: float,
    n_points: int,
    *,
    warm_start_from_method_a: Union[str, Dict[str, Any], None] = None,
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
        Either a mapping or a path to data produced by Method A used to
        initialise the IPOPT solver.
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
    nlp, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u = _build_nlp(ocp, grid, slack=use_slacks)

    opts: Dict[str, Any] = {
        "print_time": False,
        "ipopt.tol": tol,
        "ipopt.print_level": print_level,
        "ipopt.linear_solver": linear_solver,
    }
    if ipopt_opts:
        opts.update(ipopt_opts)

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    warm: Dict[str, np.ndarray] = {}
    if warm_start_from_method_a is not None:
        warm = _load_method_a_warm_start(warm_start_from_method_a, n_x, n_u, n_nodes)

    x0 = warm.get("x0")
    lam_g0 = warm.get("lam_g0")

    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0, lam_g0=lam_g0)
    stats = solver.stats()

    if (use_slacks or auto_slack_retry) and stats.get("return_status") not in {"Solve_Succeeded", "Solved_Succeeded"}:
        nlp, lbx, ubx, lbg, ubg, n_nodes, n_x, n_u = _build_nlp(ocp, grid, slack=True)
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0, lam_g0=lam_g0)
        stats = solver.stats()

    z = np.array(sol["x"]).reshape(-1)
    n_dec_x = n_x * n_nodes
    x_sol = z[:n_dec_x].reshape(n_x, n_nodes)
    u_sol = z[n_dec_x : n_dec_x + n_u * n_nodes].reshape(n_u, n_nodes)
    return SolverResult(x=x_sol, u=u_sol, grid=grid, stats=stats)
