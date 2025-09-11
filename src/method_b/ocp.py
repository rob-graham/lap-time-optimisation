from __future__ import annotations

"""Optimal control problem definition used by method B.

This module formulates a simple lap-time minimisation problem in the spatial
domain.  The independent variable is the arc length ``s`` along a reference
centre line.  The model is intentionally lightweight but contains the typical
ingredients required for racing-line optimisation such as friction limits and a
power envelope.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import casadi as ca


@dataclass
class OCP:
    """Spatial domain optimal control problem for lap-time minimisation."""

    # ------------------------------------------------------------------
    # Problem parameters
    kappa_c: np.ndarray | float = 0.0
    """Curvature profile of the reference centre line."""

    track_half_width: np.ndarray | float = 5.0
    """Half of the track width (may vary along the track)."""

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

    # Numerical safeguard for speed denominators
    v_eps: float = 1e-3

    # State and control dimensions
    n_x: int = 4
    n_u: int = 2

    # Scaling factors for state variables (set in ``__post_init__``)
    e_scale: float = field(init=False)
    kappa_scale: float = field(init=False)

    def __post_init__(self) -> None:  # pragma: no cover - simple assignment
        """Initialise scaling factors based on bounds."""
        self.kappa_c = np.asarray(self.kappa_c, dtype=float)
        self.track_half_width = np.asarray(self.track_half_width, dtype=float)
        max_hw = float(np.max(self.track_half_width)) if self.track_half_width.size else 1.0
        self.e_scale = max_hw
        self.kappa_scale = max(abs(self.kappa_bounds[0]), abs(self.kappa_bounds[1])) or 1.0

    # ------------------------------------------------------------------
    # Scaling helpers
    def scale_x(self, x: ca.SX) -> ca.SX:
        """Scale state vector for improved conditioning."""

        return ca.vertcat(
            x[0] / self.e_scale,
            x[1],
            x[2],
            x[3] / self.kappa_scale,
        )

    def unscale_x(self, x: ca.SX) -> ca.SX:
        """Invert :meth:`scale_x` for CasADi vectors."""

        return ca.vertcat(
            x[0] * self.e_scale,
            x[1],
            x[2],
            x[3] * self.kappa_scale,
        )

    def unscale_x_array(self, x: np.ndarray) -> np.ndarray:
        """Invert :meth:`scale_x` for numeric arrays."""

        x = np.array(x, copy=True)
        x[0, :] *= self.e_scale
        x[3, :] *= self.kappa_scale
        return x

    def scale_x_array(self, x: np.ndarray) -> np.ndarray:
        """Scale numeric state arrays."""

        x = np.array(x, copy=True)
        x[0, :] /= self.e_scale
        x[3, :] /= self.kappa_scale
        return x

    # ------------------------------------------------------------------
    # Symbolic variables
    def variables(self) -> Tuple[ca.SX, ca.SX]:
        """Return symbolic state and control vectors."""

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)
        return x, u

    # ------------------------------------------------------------------
    # Dynamics in the arc-length domain
    def dynamics(self, x: ca.SX, u: ca.SX, k: int) -> ca.SX:
        """State derivatives with respect to arc length ``s``.

        Parameters
        ----------
        x, u:
            State and control vectors.
        k:
            Index of the current grid point used to look up the reference
            curvature value.
        """

        x = self.unscale_x(x)
        e, psi, v, kappa = x[0], x[1], x[2], x[3]
        u_kappa, a_x = u[0], u[1]

        kappa_c = self.kappa_c[k] if self.kappa_c.size > 1 else float(self.kappa_c)

        # Nonlinear Frenet relations with angular wrapping for stability
        psi_wrapped = ca.atan2(ca.sin(psi), ca.cos(psi))
        de_ds = ca.sin(psi_wrapped)
        dpsi_ds = kappa - kappa_c
        dv_ds = a_x / ca.fmax(v, self.v_eps)
        dkappa_ds = u_kappa
        return self.scale_x(ca.vertcat(de_ds, dpsi_ds, dv_ds, dkappa_ds))

    # ------------------------------------------------------------------
    # Path constraints
    def _a_power_max(self, v: ca.SX) -> ca.SX:
        """Compute speed-dependent acceleration limit from power."""

        drag = 0.5 * self.rho * self.CdA * v**2
        rr = self.Crr * self.mass * self.g
        # P = F * v  =>  a_max = (P/v - drag - rr) / m
        return (
            self.power / ca.fmax(self.mass * v, self.v_eps)
            - drag / self.mass
            - rr / self.mass
        )

    def path_constraints(self, x: ca.SX, u: ca.SX, k: int) -> ca.SX:
        """Return stacked path-constraint expressions for node ``k``."""

        x = self.unscale_x(x)
        e, _, v, kappa = x[0], x[1], x[2], x[3]
        a_x = u[1]

        ay = v**2 * kappa
        friction = a_x**2 + ay**2
        power_con = a_x - self._a_power_max(v)

        width = self.track_half_width[k] if self.track_half_width.size > 1 else float(self.track_half_width)

        cons = [e / width, friction, power_con]
        if self.phi_max_deg is not None:
            lean = v**2 * ca.fabs(kappa)
            cons.append(lean)

        return ca.vertcat(*cons)

    # ------------------------------------------------------------------
    # Bounds for states, controls and path constraints
    def bounds(
        self,
    ) -> Tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
        max_hw = float(np.max(self.track_half_width)) if self.track_half_width.size else float(self.track_half_width)
        e_min = -max_hw
        e_max = max_hw

        x_min = [e_min / self.e_scale, -ca.inf, 0.0, self.kappa_bounds[0] / self.kappa_scale]
        x_max = [e_max / self.e_scale, ca.inf, ca.inf, self.kappa_bounds[1] / self.kappa_scale]

        u_min = [self.u_kappa_bounds[0], -self.a_brake]
        u_max = [self.u_kappa_bounds[1], self.a_wheelie_max]

        g_min = [-1.0, 0.0, -ca.inf]
        g_max = [1.0, (self.mu * self.g) ** 2, 0.0]

        if self.phi_max_deg is not None:
            g_min.append(0.0)
            g_max.append(self.g * np.tan(np.deg2rad(self.phi_max_deg)))

        return x_min, x_max, u_min, u_max, g_min, g_max

    # ------------------------------------------------------------------
    # Stage cost
    def stage_cost(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """Lap-time objective with control regularisation."""

        x = self.unscale_x(x)
        v = x[2]
        u_kappa, a_x = u[0], u[1]
        inv_v = 1.0 / ca.fmax(v, self.v_eps)
        return inv_v + self.w_u_kappa * u_kappa**2 + self.w_a_x * a_x**2


__all__ = ["OCP"]

