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
    path_label: str = "Racing line",
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
        ax.plot(x_path, y_path, color="tab:red", label=path_label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    return ax


def plot_speed_profile(
    s: Iterable[float],
    speed: Iterable[float],
    label: str | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot speed as a function of distance ``s`` along the track."""
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(s, speed, color="tab:blue", label=label)
    ax.set_xlabel("Distance along track [m]")
    ax.set_ylabel("Speed [m/s]")
    if label is not None:
        ax.legend()
    return ax


def plot_speed_caps(
    s: Iterable[float],
    v: Iterable[float],
    v_lean: Iterable[float] | None = None,
    v_steer: Iterable[float] | None = None,
    label: str = "Speed",
    ax: Optional[plt.Axes] = None,
    max_speed_cap: float | None = None,
) -> plt.Axes:
    """Overlay speed caps with the final speed profile.

    Any non-finite cap values are omitted from the plot to avoid matplotlib
    auto-scaling to extremely large values.  Remaining data are clipped to the
    provided ``max_speed_cap`` (default ``100.0`` m/s) and the axis limits are
    fixed accordingly.
    """
    if ax is None:
        _, ax = plt.subplots()

    if max_speed_cap is None:
        max_speed_cap = 100.0

    s_arr = np.asarray(list(s), dtype=float)
    v_arr = np.asarray(list(v), dtype=float)

    ax.plot(s_arr, np.clip(v_arr, 0.0, max_speed_cap), label=label, color="tab:blue")

    def _plot_cap(data: Iterable[float], *, label: str, color: str) -> None:
        cap_arr = np.asarray(list(data), dtype=float)
        mask = np.isfinite(cap_arr)
        if not np.any(mask):
            return
        cap_arr = cap_arr[mask]
        s_cap = s_arr[mask]
        ax.plot(
            s_cap,
            np.clip(cap_arr, 0.0, max_speed_cap),
            label=label,
            color=color,
            linestyle="--",
        )

    if v_lean is not None:
        _plot_cap(v_lean, label="Lean cap", color="tab:orange")

    if v_steer is not None:
        _plot_cap(v_steer, label="Steer cap", color="tab:green")

    ax.set_xlabel("Distance along track [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_ylim(0.0, max_speed_cap)
    ax.legend()
    return ax


def plot_acceleration_profile(
    s: Iterable[float],
    ax_longitudinal: Iterable[float],
    ay_lateral: Iterable[float],
    label: str | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot longitudinal and lateral acceleration versus distance."""
    if ax is None:
        _, ax = plt.subplots()

    suffix = f" ({label})" if label else ""
    ax.plot(s, ax_longitudinal, label=f"Longitudinal{suffix}", color="tab:green")
    ax.plot(s, ay_lateral, label=f"Lateral{suffix}", color="tab:orange")
    ax.set_xlabel("Distance along track [m]")
    ax.set_ylabel("Acceleration [m/s$^2$]")
    ax.legend()
    return ax
