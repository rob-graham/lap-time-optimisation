"""Vehicle model utilities.

This module defines a :class:`Vehicle` that loads parameters from a CSV file
and exposes helper methods to compute basic longitudinal forces.  The engine
tractive force curves for each gear are derived from the supplied torque
characteristic and driveline ratios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import csv
import numpy as np


class Vehicle:
    """Represent a simple motorcycle model.

    Parameters
    ----------
    param_file:
        Path to a CSV file containing vehicle parameters.  The file is expected
        to list ``key,value`` pairs per line.  Optionally a section headed by
        ``rpm,Nm`` may follow to describe the engine torque curve.  If no
        explicit torque curve is provided, a flat curve with value ``T_peak`` up
        to ``shift_rpm`` is assumed.
    """

    def __init__(self, param_file: str | Path = "data/bike_params.csv") -> None:
        self.param_file = Path(param_file)
        self._load_parameters()
        self._compute_tractive_force_curves()

    # ------------------------------------------------------------------
    # Parameter loading
    def _load_parameters(self) -> None:
        params: Dict[str, float | bool | str] = {}
        torque_data: list[Tuple[float, float]] = []
        in_torque_section = False

        with self.param_file.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                key = row[0].strip()
                if key.lower() == "rpm":
                    in_torque_section = True
                    continue
                if in_torque_section:
                    try:
                        rpm = float(row[0])
                        torque = float(row[1])
                    except (IndexError, ValueError):
                        continue
                    torque_data.append((rpm, torque))
                else:
                    try:
                        raw = row[1].strip()
                    except IndexError:
                        continue
                    try:
                        params[key] = float(raw)
                    except ValueError:
                        low = raw.lower()
                        if low in {"true", "false"}:
                            params[key] = low == "true"
                        else:
                            params[key] = raw

        if not torque_data:
            # Fall back to a constant torque curve if none provided.
            t_peak = params.get("T_peak", 0.0)
            shift = params.get("shift_rpm", 0.0)
            torque_data = [(0.0, t_peak), (shift, t_peak)]

        self.params = params

        # Ensure torque points are ordered by increasing RPM to avoid
        # non-monotonic behaviour during interpolation.
        torque_data.sort(key=lambda p: p[0])

        self.rpm = np.array([p[0] for p in torque_data], dtype=float)
        self.torque = np.array([p[1] for p in torque_data], dtype=float)

        # Commonly used parameters as attributes for convenience.
        self.rho = params["rho"]
        self.g = params["g"]
        self.m = params["m"]
        self.CdA = params["CdA"]
        self.Crr = params["Crr"]
        self.rw = params["rw"]
        self.eta = params.get("eta_driveline", 1.0)
        self.primary = params["primary"]
        self.final_drive = params["final_drive"]
        self.mu = params["mu"]
        self.a_wheelie_max = params["a_wheelie_max"]
        self.a_brake = params["a_brake"]

        # Collect gear ratios in numerical order.
        gear_keys = sorted(
            (k for k in params if k.startswith("gear")),
            key=lambda x: int(x[4:]),
        )
        self.gear_ratios = np.array([params[k] for k in gear_keys], dtype=float)

        # Optional caps and flags
        self.phi_max_deg = params.get("phi_max_deg")  # None disables cap
        self.kappa_dot_max = params.get("kappa_dot_max")
        self.use_lean_angle_cap = params.get("use_lean_angle_cap", True)
        self.use_steer_rate_cap = params.get("use_steer_rate_cap", True)

    # ------------------------------------------------------------------
    # Tractive force computation
    def _compute_tractive_force_curves(self) -> None:
        self._speed: Dict[int, np.ndarray] = {}
        self._force: Dict[int, np.ndarray] = {}

        for i, gear_ratio in enumerate(self.gear_ratios, start=1):
            total_ratio = self.primary * gear_ratio * self.final_drive
            speed = self.rpm * 2 * np.pi * self.rw / (60.0 * total_ratio)
            force = self.torque * total_ratio * self.eta / self.rw
            self._speed[i] = speed
            self._force[i] = force

    def tractive_force_vs_speed(self, gear: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return speed and tractive force arrays for ``gear``."""
        if gear not in self._speed:
            raise ValueError(f"invalid gear: {gear}")
        return self._speed[gear], self._force[gear]

    def tractive_force(self, speed: Iterable[float] | float, gear: int) -> np.ndarray:
        """Interpolate available tractive force for a given ``speed`` and ``gear``."""
        speeds, forces = self.tractive_force_vs_speed(gear)
        return np.interp(speed, speeds, forces, left=forces[0], right=0.0)

    # ------------------------------------------------------------------
    # Resistance forces
    def aerodynamic_drag(self, speed: Iterable[float] | float) -> np.ndarray:
        """Compute aerodynamic drag force magnitude at ``speed``."""
        v = np.asarray(speed, dtype=float)
        return 0.5 * self.rho * self.CdA * v**2

    def rolling_resistance(self, speed: Iterable[float] | float | None = None) -> np.ndarray:
        """Compute rolling resistance force magnitude.

        The rolling resistance is modelled as ``Crr * m * g`` and is independent
        of speed.  If ``speed`` is provided, the return value has the same shape
        as ``speed``.
        """
        f_rr = self.Crr * self.m * self.g
        if speed is None:
            return f_rr
        v = np.asarray(speed, dtype=float)
        return np.full_like(v, f_rr, dtype=float)

    # ------------------------------------------------------------------
    # Acceleration helpers
    def max_acceleration(self, speed: float, gear: int, ay: float) -> float:
        """Return the maximum longitudinal acceleration at ``speed`` and ``gear``.

        Parameters
        ----------
        speed:
            Vehicle speed in metres per second.
        gear:
            Selected gear, indexed from 1.
        ay:
            Lateral acceleration in metres per second squared.

        Returns
        -------
        float
            The maximum achievable longitudinal acceleration, limited by
            available engine force, aerodynamic drag, rolling resistance and
            the traction ellipse with wheelie constraint.
        """

        f_drive = (
            self.tractive_force(speed, gear)
            - self.aerodynamic_drag(speed)
            - self.rolling_resistance()
        )
        ax_engine = f_drive / self.m
        ax_limit = min(
            self.a_wheelie_max,
            float(np.sqrt(max((self.mu * self.g) ** 2 - ay**2, 0.0))),
        )
        return min(ax_engine, ax_limit)
