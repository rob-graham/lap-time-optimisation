import sys
from pathlib import Path

import casadi as ca
import numpy as np

# Ensure the repository root is on the path so ``src`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.method_b.ocp import OCP


def test_variable_shapes():
    kappa = np.zeros(5)
    width = np.full(5, 5.0)
    ocp_def = OCP(kappa_c=kappa, track_half_width=width)
    x, u = ocp_def.variables()
    assert x.size1() == ocp_def.n_x
    assert u.size1() == ocp_def.n_u
    # Smoke-test casadi by creating a scalar symbol
    assert isinstance(ca.SX.sym("z"), ca.SX)
