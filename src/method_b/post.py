"""Post-processing helpers for method B results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Dict

import numpy as np
import matplotlib.pyplot as plt

from .solver import SolverResult
from ..io_utils import write_csv
from .. import plots

G = 9.81  # gravitational acceleration used for normalising forces


def reconstruct(z: Iterable[float], n_x: int, n_u: int, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Reshape flat decision vector ``z`` into state and control arrays."""

    z = np.asarray(z, dtype=float).reshape(-1)
    n_dec_x = n_x * n_nodes
    x = z[:n_dec_x].reshape(n_x, n_nodes)
    u = z[n_dec_x : n_dec_x + n_u * n_nodes].reshape(n_u, n_nodes)
    return x, u


def extract_arrays(result: SolverResult) -> Dict[str, np.ndarray]:
    """Return named arrays for states, controls and derived quantities."""

    s = result.grid
    e, psi, v, kappa = result.x
    u_kappa, a_x = result.u
    a_y = v**2 * kappa
    ellipse = np.hypot(a_x, a_y) / G

    return {
        "s": s,
        "e": e,
        "psi": psi,
        "v": v,
        "kappa": kappa,
        "u_kappa": u_kappa,
        "a_x": a_x,
        "a_y": a_y,
        "ellipse": ellipse,
    }


def write_solution(result: SolverResult, file_path: str) -> None:
    """Write optimisation results to ``file_path`` as CSV."""

    data = extract_arrays(result)
    write_csv(data, file_path)


def plot_solution(result: SolverResult) -> None:
    """Create basic plots for the optimised trajectory."""

    data = extract_arrays(result)
    plots.plot_speed_profile(data["s"], data["v"], label="Method B")
    plots.plot_acceleration_profile(data["s"], data["a_x"], data["a_y"], label="Method B")


def save_outputs(result: SolverResult, timestamp: str | None = None) -> Path:
    """Save CSV data and plots under ``outputs/<timestamp>/method_b/``.

    Parameters
    ----------
    result:
        Optimisation result returned by :mod:`solver`.
    timestamp:
        Optional timestamp string.  If ``None`` the current time is used.
    """

    data = extract_arrays(result)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path("outputs") / timestamp / "method_b"

    write_csv(data, out_dir / "solution.csv")

    ax = plots.plot_speed_profile(data["s"], data["v"], label="Method B")
    ax.figure.savefig(out_dir / "speed_profile.png")
    plt.close(ax.figure)

    ax = plots.plot_acceleration_profile(data["s"], data["a_x"], data["a_y"], label="Method B")
    ax.figure.savefig(out_dir / "acceleration_profile.png")
    plt.close(ax.figure)

    return out_dir
