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
    inner_edge: np.ndarray | None = None,
    outer_edge: np.ndarray | None = None,
    x_path: Optional[Iterable[float]] = None,
    y_path: Optional[Iterable[float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot the track plan view.

    Parameters
    ----------
    x_center, y_center:
        Coordinates of the track centreline.
    inner_edge, outer_edge:
        Arrays of shape ``(N, 2)`` giving ``x`` and ``y`` coordinates of the
        inner and outer track boundaries.  ``left_edge``/``right_edge`` may be
        supplied instead for backward compatibility.
    x_path, y_path:
        Optional coordinates of the racing line to overlay.
    ax:
        Existing axes to draw on.  If ``None`` a new figure and axes are
        created.
    """
    if inner_edge is None:
        inner_edge = kwargs.get("left_edge")
    if outer_edge is None:
        outer_edge = kwargs.get("right_edge")
    if inner_edge is None or outer_edge is None:
        raise ValueError("Track boundaries must be provided")

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(inner_edge[:, 0], inner_edge[:, 1], "k--", label="Track edge")
    ax.plot(outer_edge[:, 0], outer_edge[:, 1], "k--")
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
