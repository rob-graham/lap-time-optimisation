from __future__ import annotations

"""Definition of a simple optimal control problem for method B.

The :class:`OCP` class bundles symbolic variables, system dynamics, path
constraints and cost terms used by the collocation-based solver in
:mod:`src.method_b.solver`.
"""

from dataclasses import dataclass

import casadi as ca


@dataclass
class OCP:
    """Minimal optimal control problem specification.

    This example problem models a system with three state variables ``x`` and
    two control inputs ``u``.  The particular dynamics are not intended to be a
    realistic vehicle model; they merely provide a concrete system to
    demonstrate the transcription and solver pipeline.
    """

    n_x: int = 3
    """Number of state variables."""

    n_u: int = 2
    """Number of control variables."""

    def variables(self) -> tuple[ca.SX, ca.SX]:
        """Return symbolic state and control vectors."""

        x = ca.SX.sym("x", self.n_x)
        u = ca.SX.sym("u", self.n_u)
        return x, u

    # -- dynamics ---------------------------------------------------------
    def dynamics(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """Right-hand side of the ODE ``\dot{x} = f(x, u)``.

        The toy system used here represents a chain of integrators driven by two
        control inputs interpreted as throttle and brake commands.  The third
        state is an acceleration-like term actuated by the difference of the
        controls.  The exact equations are purposely simple as the focus of
        *method B* is on demonstrating the optimisation workflow rather than the
        model itself.
        """

        s, v, a = x[0], x[1], x[2]
        throttle, brake = u[0], u[1]

        ds = v
        dv = a
        da = throttle - brake
        return ca.vertcat(ds, dv, da)

    # -- path constraints -------------------------------------------------
    def path_constraints(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """Expressions bounded between ``0`` and ``1``.

        The default problem constrains both control inputs to lie in the range
        ``[0, 1]`` by returning them as path constraints.
        """

        throttle, brake = u[0], u[1]
        return ca.vertcat(throttle, brake)

    # -- bounds -----------------------------------------------------------
    def bounds(
        self,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
        """Bounds for state/control variables and path constraints."""

        x_min = [-ca.inf, 0.0, -ca.inf]
        x_max = [ca.inf, ca.inf, ca.inf]
        u_min = [0.0, 0.0]
        u_max = [1.0, 1.0]
        g_min = [0.0, 0.0]
        g_max = [1.0, 1.0]
        return x_min, x_max, u_min, u_max, g_min, g_max

    # -- cost -------------------------------------------------------------
    def stage_cost(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """Quadratic cost on the controls."""

        return ca.sumsqr(u)
