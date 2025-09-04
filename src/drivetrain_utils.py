from __future__ import annotations

"""Utility functions for simple drivetrain calculations."""

from typing import Sequence
import math


def engine_rpm(
    v_mps: float,
    primary: float,
    final_drive: float,
    gear_ratio: float,
    rw: float,
) -> float:
    """Return engine speed in revolutions per minute.

    Parameters
    ----------
    v_mps:
        Vehicle speed in metres per second.
    primary:
        Primary drive ratio.
    final_drive:
        Final drive ratio.
    gear_ratio:
        Selected gearbox ratio.
    rw:
        Effective wheel radius in metres.
    """
    omega_w = v_mps / rw
    omega_e = omega_w * primary * final_drive * gear_ratio
    return omega_e * 60.0 / (2 * math.pi)


def select_gear(
    v_mps: float,
    gears: Sequence[float],
    shift_rpm: float,
    primary: float,
    final_drive: float,
    rw: float,
) -> float:
    """Return the highest gear ratio that keeps engine RPM below the shift point."""
    for g in reversed(gears):
        if engine_rpm(v_mps, primary, final_drive, g, rw) <= shift_rpm:
            return g
    return gears[0]


__all__ = ["engine_rpm", "select_gear"]
