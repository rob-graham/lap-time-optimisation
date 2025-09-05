from __future__ import annotations

"""Post-processing helpers for method B results."""

from typing import Iterable

import numpy as np

from .solver import SolverResult
from ..io_utils import write_csv
from .. import plots


def reconstruct(z: Iterable[float], n_x: int, n_u: int, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Reshape flat decision vector ``z`` into state and control arrays."""

    z = np.asarray(z, dtype=float).reshape(-1)
    n_dec_x = n_x * n_nodes
    x = z[:n_dec_x].reshape(n_x, n_nodes)
    u = z[n_dec_x : n_dec_x + n_u * n_nodes].reshape(n_u, n_nodes)
    return x, u


def write_solution(result: SolverResult, file_path: str) -> None:
    """Write optimisation results to ``file_path`` as CSV."""

    data: dict[str, Iterable[float]] = {"s": result.grid}
    for i in range(result.x.shape[0]):
        data[f"x{i}"] = result.x[i]
    for j in range(result.u.shape[0]):
        data[f"u{j}"] = result.u[j]
    write_csv(data, file_path)


def plot_solution(result: SolverResult) -> None:
    """Create basic plots for the optimised trajectory."""

    plots.plot_speed_profile(result.grid, result.x[1])
    plots.plot_acceleration_profile(result.grid, result.u[0], result.u[1])
