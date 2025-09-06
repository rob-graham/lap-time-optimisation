import sys
from pathlib import Path

import numpy as np
import casadi as ca

# Ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.method_b.ocp import OCP


def _eval_dynamics(ocp: OCP, x_vals, u_vals, k: int = 0):
    x = ocp.scale_x(ca.DM(x_vals))
    u = ca.DM(u_vals)
    dx_scaled = ocp.dynamics(x, u, k)
    dx = ocp.unscale_x(dx_scaled)
    return np.array(dx).squeeze()


def test_dynamics_straight_line():
    ocp = OCP(kappa_c=0.0, track_half_width=5.0)
    psi = 0.3
    kappa = 0.1
    x_vals = [0.0, psi, 10.0, kappa]
    u_vals = [0.0, 0.0]
    de_ds, dpsi_ds, *_ = _eval_dynamics(ocp, x_vals, u_vals)
    assert np.isclose(de_ds, np.sin(psi))
    assert np.isclose(dpsi_ds, kappa)


def test_dynamics_constant_radius_corner():
    kappa_c = 0.1
    ocp = OCP(kappa_c=kappa_c, track_half_width=5.0)
    psi = 0.0
    e = 0.0
    v = 10.0
    kappa = kappa_c
    x_vals = [e, psi, v, kappa]
    u_vals = [0.0, 0.0]
    de_ds, dpsi_ds, *_ = _eval_dynamics(ocp, x_vals, u_vals, k=0)
    assert np.isclose(de_ds, 0.0)
    assert np.isclose(dpsi_ds, 0.0)
