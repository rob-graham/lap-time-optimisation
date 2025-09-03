"""Track geometry utilities.

This module provides functionality to load a track layout from a CSV file and
interpolate the centreline onto a uniform arc-length grid. The track width is
used to compute coordinates of the left and right boundaries.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def load_track(file_path: str, ds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a track layout and interpolate it onto a uniform grid.

    Parameters
    ----------
    file_path:
        Path to the CSV file containing the track description. The file must
        contain ``x_m``, ``y_m`` and ``width_m`` columns which describe the
        centreline coordinates and track width at each node.
    ds:
        Spacing of the uniform arc-length grid in metres.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing arrays for centreline ``x`` and ``y`` coordinates,
        heading angle ``psi``, curvature ``kappa``, and the coordinates of the
        left and right track boundaries.
    """
    if ds <= 0:
        raise ValueError("ds must be positive")

    df = pd.read_csv(file_path)
    x = df["x_m"].to_numpy()
    y = df["y_m"].to_numpy()
    width = df["width_m"].to_numpy()

    # Ensure the track is closed by appending the first point to the end.
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        width = np.r_[width, width[0]]

    # Compute cumulative arc length along the centreline.
    segment_lengths = np.hypot(np.diff(x), np.diff(y))
    s_nodes = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = s_nodes[-1]
    s_uniform = np.arange(0.0, total_length, ds)

    # Interpolate centreline and width onto the uniform grid.
    x_s = np.interp(s_uniform, s_nodes, x)
    y_s = np.interp(s_uniform, s_nodes, y)
    width_s = np.interp(s_uniform, s_nodes, width)

    # Compute heading and curvature.
    dx = np.gradient(x_s, ds, edge_order=2)
    dy = np.gradient(y_s, ds, edge_order=2)
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = np.gradient(heading, ds, edge_order=2)

    # Compute coordinates of the left and right boundaries.
    half_width = 0.5 * width_s
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    x_left = x_s + half_width * normal_x
    y_left = y_s + half_width * normal_y
    x_right = x_s - half_width * normal_x
    y_right = y_s - half_width * normal_y

    left_edge = np.column_stack((x_left, y_left))
    right_edge = np.column_stack((x_right, y_right))

    return x_s, y_s, heading, curvature, left_edge, right_edge
