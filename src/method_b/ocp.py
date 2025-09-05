from __future__ import annotations

"""Optimal control problem definition used by method B.

This module formulates a simple lap-time minimisation problem in the spatial
domain.  The independent variable is the arc length ``s`` along a reference
centre line.  The model is intentionally lightweight but contains the typical
ingredients required for racing-line optimisation such as friction limits and a
power envelope.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import casadi as ca


@dataclass
class OCP:
    """Spatial domain optimal control problem for lap-time minimisation."""

    # ------------------------------------------------------------------
    # Problem parameters
    kappa_c: float = 0.0
    """Curvature of the reference centre line (assumed constant)."""

    track_half_width: float = 5.0
    """Half of the track width used for lateral bounds."""

    mu: float = 1.0
    """Tyre-road friction coefficient."""

    g: float = 9.81
    """Gravitational acceleration."""

    a_wheelie_max: float = 9.81
    """Maximum forward acceleration before a wheelie."""

    a_brake: float = 9.81
    """Maximum braking deceleration before a stoppie."""

    power: float = 100000.0
    """Available engine power in watts used for the power envelope."""

    mass: float = 200.0
    """Vehicle mass in kilograms."""

    rho: float = 1.225
    CdA: float = 0.5
    Crr: float = 0.015

    phi_max_deg: float | None = None
    """Optional maximum lean angle in degrees. ``None`` disables this limit."""

    kappa_bounds: Tuple[float, float] = (-0.2, 0.2)
    """Bounds on curvature state ``kappa``."""

    u_kappa_bounds: Tuple[float, float] = (-0.1, 0.1)
    """Bounds on curvature rate control ``u_kappa``."""

    w_u_kappa: float = 1e-4
    """Regularisation weight for curvature rate control."""

    w_a_x: float = 1e-4
    """Regularisation weight for longitudinal acceleration."""

    # State and control dimensions
    n_x: int = 4
    n_u: int = 2

    # ------------------------------------------------------------------
    # Symbolic variables
    def variables(self) -> Tuple[ca.SX, ca.SX]:
        """Return symbolic state and control vectors."""

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)
        return x, u

    # ------------------------------------------------------------------
    # Dynamics in the arc-length domain
    def dynamics(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """State derivatives with respect to arc length ``s``."""

        e, psi, v, kappa = x[0], x[1], x[2], x[3]
        u_kappa, a_x = u[0], u[1]

        # Small-angle Frenet relations
        de_ds = psi
        dpsi_ds = kappa - self.kappa_c
        dv_ds = a_x / ca.fmax(v, 1e-3)
        dkappa_ds = u_kappa
        return ca.vertcat(de_ds, dpsi_ds, dv_ds, dkappa_ds)

    # ------------------------------------------------------------------
    # Path constraints
    def _a_power_max(self, v: ca.SX) -> ca.SX:
        """Compute speed-dependent acceleration limit from power."""

        drag = 0.5 * self.rho * self.CdA * v**2
        rr = self.Crr * self.mass * self.g
        # P = F * v  =>  a_max = (P/v - drag - rr) / m
        return self.power / ca.fmax(self.mass * v, 1e-3) - drag / self.mass - rr / self.mass

    def path_constraints(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """Return stacked path-constraint expressions."""

        e, _, v, kappa = x[0], x[1], x[2], x[3]
        a_x = u[1]

        ay = v**2 * kappa
        friction = a_x**2 + ay**2
        power_con = a_x - self._a_power_max(v)

        cons = [e, friction, power_con]
        if self.phi_max_deg is not None:
            lean = v**2 * ca.fabs(kappa)
            cons.append(lean)

        return ca.vertcat(*cons)

    # ------------------------------------------------------------------
    # Bounds for states, controls and path constraints
    def bounds(
        self,
    ) -> Tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
        e_min = -self.track_half_width
        e_max = self.track_half_width

        x_min = [e_min, -ca.inf, 0.0, self.kappa_bounds[0]]
        x_max = [e_max, ca.inf, ca.inf, self.kappa_bounds[1]]

        u_min = [self.u_kappa_bounds[0], -self.a_brake]
        u_max = [self.u_kappa_bounds[1], self.a_wheelie_max]

        g_min = [e_min, 0.0, -ca.inf]
        g_max = [e_max, (self.mu * self.g) ** 2, 0.0]

        if self.phi_max_deg is not None:
            g_min.append(0.0)
            g_max.append(self.g * np.tan(np.deg2rad(self.phi_max_deg)))

        return x_min, x_max, u_min, u_max, g_min, g_max

    # ------------------------------------------------------------------
    # Stage cost
    def stage_cost(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """Lap-time objective with control regularisation."""

        v = x[2]
        u_kappa, a_x = u[0], u[1]
        inv_v = 1.0 / ca.fmax(v, 1e-3)
        return inv_v + self.w_u_kappa * u_kappa**2 + self.w_a_x * a_x**2


__all__ = ["OCP"]

