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
    if rw <= 0:
        raise ValueError("wheel radius must be positive")
    if primary <= 0 or final_drive <= 0 or gear_ratio <= 0:
        raise ValueError("ratios must be positive")

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
    """Return the highest gear ratio that keeps engine RPM below ``shift_rpm``."""
    if not gears:
        raise ValueError("gears must be non-empty")
    if any(g <= 0 for g in gears):
        raise ValueError("gears must contain only positive ratios")

    # ``gears`` is expected to be ordered from lowest to highest gear number
    # (i.e. highest to lowest ratio).  Iterate in this order and select the
    # first gear whose engine speed does not exceed the shift point.  If the
    # engine would exceed ``shift_rpm`` even in the highest gear, fall back to
    # that top gear rather than an unrealistic first-gear default.
    for g in gears:
        if engine_rpm(v_mps, primary, final_drive, g, rw) <= shift_rpm:
            return g
    return gears[-1]


__all__ = ["engine_rpm", "select_gear"]
