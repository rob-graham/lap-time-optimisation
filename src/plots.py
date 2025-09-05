from __future__ import annotations

"""Plotting helpers for lap-time optimisation results.

This module contains simple functions for visualising the computed racing line
and associated speed profile.  Plots are produced using :mod:`matplotlib` and
return the :class:`~matplotlib.axes.Axes` instance for further customisation.
"""

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_plan_view(
    x_center: Iterable[float],
    y_center: Iterable[float],
    left_edge: np.ndarray | None = None,
    right_edge: np.ndarray | None = None,
    x_path: Optional[Iterable[float]] = None,
    y_path: Optional[Iterable[float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the track plan view.

    Parameters
    ----------
    x_center, y_center:
        Coordinates of the track centreline.
    left_edge, right_edge:
        Arrays of shape ``(N, 2)`` giving ``x`` and ``y`` coordinates of the
        track boundaries.  Both must be provided; the deprecated
        ``inner_edge``/``outer_edge`` aliases are no longer supported.
    x_path, y_path:
        Optional coordinates of the racing line to overlay.
    ax:
        Existing axes to draw on.  If ``None`` a new figure and axes are
        created.
    """
    if left_edge is None or right_edge is None:
        raise ValueError("Track boundaries must be provided")

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(left_edge[:, 0], left_edge[:, 1], "k--", label="Track edge")
    ax.plot(right_edge[:, 0], right_edge[:, 1], "k--")
    ax.plot(x_center, y_center, color="k", label="Centreline")

    if x_path is not None and y_path is not None:
        ax.plot(x_path, y_path, color="tab:red", label="Racing line")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    return ax


def plot_speed_profile(
    s: Iterable[float],
    speed: Iterable[float],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot speed as a function of distance ``s`` along the track."""
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(s, speed, color="tab:blue")
    ax.set_xlabel("Distance along track [m]")
    ax.set_ylabel("Speed [m/s]")
    return ax


def plot_speed_caps(
    s: Iterable[float],
    v: Iterable[float],
    v_lean: Iterable[float] | None = None,
    v_steer: Iterable[float] | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay speed caps with the final speed profile."""
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(s, v, label="Speed", color="tab:blue")

    if v_lean is not None:
        ax.plot(s, v_lean, label="Lean cap", color="tab:orange", linestyle="--")

    if v_steer is not None:
        ax.plot(s, v_steer, label="Steer cap", color="tab:green", linestyle="--")

    ax.set_xlabel("Distance along track [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.legend()
    return ax


def plot_acceleration_profile(
    s: Iterable[float],
    ax_longitudinal: Iterable[float],
    ay_lateral: Iterable[float],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot longitudinal and lateral acceleration versus distance."""
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(s, ax_longitudinal, label="Longitudinal", color="tab:green")
    ax.plot(s, ay_lateral, label="Lateral", color="tab:orange")
    ax.set_xlabel("Distance along track [m]")
    ax.set_ylabel("Acceleration [m/s$^2$]")
    ax.legend()
    return ax
