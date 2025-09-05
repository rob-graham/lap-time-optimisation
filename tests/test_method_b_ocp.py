import sys
from pathlib import Path

import casadi as ca

# Ensure the repository root is on the path so ``src`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.method_b.ocp import OCP


def test_variable_shapes():
    ocp = OCP()
    x, u = ocp.variables()
    assert x.size1() == ocp.n_x
    assert u.size1() == ocp.n_u
    # Smoke-test casadi by creating a scalar symbol
    assert isinstance(ca.SX.sym("z"), ca.SX)
