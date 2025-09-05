import sys
from pathlib import Path

import numpy as np

# Ensure the repository root is on the path so ``src`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.method_b.ocp import OCP
from src.method_b import solver


def test_closed_loop_adds_periodic_constraints():
    ocp_def = OCP()
    grid = np.linspace(0.0, 1.0, 5)
    _, lbx, ubx, lbg, ubg, n_nodes, n_x, _ = solver._build_nlp(
        ocp_def, grid, closed_loop=True
    )
    # Last n_x constraints correspond to periodicity
    assert np.allclose(lbg[-n_x:], 0.0)
    assert np.allclose(ubg[-n_x:], 0.0)
    # Start states are not fixed when closed loop
    assert not np.allclose(lbx[:n_x], ubx[:n_x])


def test_open_loop_fixes_start_states():
    ocp_def = OCP()
    grid = np.linspace(0.0, 1.0, 5)
    _, lbx, ubx, lbg_open, _, _, n_x, _ = solver._build_nlp(
        ocp_def, grid, closed_loop=False
    )
    assert np.allclose(lbx[:n_x], 0.0)
    assert np.allclose(ubx[:n_x], 0.0)
    # Open loop omits periodicity constraints
    _, _, _, lbg_closed, _, _, _, _ = solver._build_nlp(
        ocp_def, grid, closed_loop=True
    )
    assert lbg_closed.size == lbg_open.size + n_x

