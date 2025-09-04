"""Track geometry utilities.

This module provides helpers for working with two simple track layout
descriptions:

``load_track``
    Legacy helper that interpolates a centreline defined by arbitrary nodes on
    to a uniform arc-length grid.
``load_track_layout``
    Parser for the more structured ``track_layout.csv`` format used by the
    command line demo.  Each consecutive row describes either a straight or a
    constant-radius corner which is discretised at the requested spacing.

The :func:`load_track_layout` function returns a :class:`TrackGeometry`
dataclass bundling the centreline and track-edge coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class TrackGeometry:
    """Discrete representation of a race track."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    left_edge: np.ndarray
    right_edge: np.ndarray


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


def load_track_layout(path: str, ds: float) -> TrackGeometry:
    """Load a structured ``track_layout.csv`` file.

    Parameters
    ----------
    path:
        Location of the CSV file describing the track layout.  Each row marks
        the beginning of a segment.  The ``section_type`` column selects either
        ``"straight"`` or ``"corner"`` and ``radius_m`` gives the signed corner
        radius.
    ds:
        Desired arc-length spacing for discretisation.

    Returns
    -------
    TrackGeometry
        Dataclass containing the sampled centreline and track boundaries.
    """
    if ds <= 0:
        raise ValueError("ds must be positive")

    df = pd.read_csv(path)
    x_nodes = df["x_m"].to_numpy(float)
    y_nodes = df["y_m"].to_numpy(float)
    width_nodes = df["width_m"].to_numpy(float)
    radius_nodes = df.get("radius_m", pd.Series(0.0, index=df.index)).to_numpy(float)
    section_types = df["section_type"].astype(str).to_numpy()

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    heading_list: list[np.ndarray] = []
    curvature_list: list[np.ndarray] = []
    width_list: list[np.ndarray] = []

    n = len(df)
    for i in range(n):
        x0, y0 = x_nodes[i], y_nodes[i]
        x1, y1 = x_nodes[(i + 1) % n], y_nodes[(i + 1) % n]
        w0, w1 = width_nodes[i], width_nodes[(i + 1) % n]
        seg_type = section_types[i].lower()

        if seg_type == "straight":
            dx, dy = x1 - x0, y1 - y0
            seg_len = float(np.hypot(dx, dy))
            s_local = np.arange(0.0, seg_len, ds)
            if i != 0:
                s_local = s_local[1:]
            ratio = s_local / seg_len
            x_seg = x0 + ratio * dx
            y_seg = y0 + ratio * dy
            heading_seg = np.full_like(x_seg, np.arctan2(dy, dx))
            curvature_seg = np.zeros_like(x_seg)
            width_seg = w0 + ratio * (w1 - w0)
        elif seg_type == "corner":
            r = radius_nodes[i]
            if r == 0:
                raise ValueError("corner segment requires non-zero radius")
            v = np.array([x1 - x0, y1 - y0], dtype=float)
            chord = float(np.hypot(*v))
            mid = np.array([x0 + x1, y0 + y1], dtype=float) / 2.0
            perp = np.array([-v[1], v[0]]) / chord
            h = np.sqrt(r**2 - (chord / 2.0) ** 2)
            centre = mid + np.sign(r) * perp * h
            phi0 = np.arctan2(y0 - centre[1], x0 - centre[0])
            theta = 2.0 * np.arcsin(chord / (2.0 * abs(r))) * np.sign(r)
            seg_len = abs(r * theta)
            s_local = np.arange(0.0, seg_len, ds)
            if i != 0:
                s_local = s_local[1:]
            phi = phi0 + s_local / r
            x_seg = centre[0] + r * np.cos(phi)
            y_seg = centre[1] + r * np.sin(phi)
            heading_seg = phi + np.sign(r) * (np.pi / 2.0)
            curvature_seg = np.full_like(x_seg, 1.0 / r)
            width_seg = w0 + (w1 - w0) * (s_local / seg_len)
        else:
            raise ValueError(f"unknown section_type '{section_types[i]}'")

        x_list.append(x_seg)
        y_list.append(y_seg)
        heading_list.append(heading_seg)
        curvature_list.append(curvature_seg)
        width_list.append(width_seg)

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    heading = np.concatenate(heading_list)
    curvature = np.concatenate(curvature_list)
    width = np.concatenate(width_list)

    half_width = 0.5 * width
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    left_edge = np.column_stack((x + half_width * normal_x, y + half_width * normal_y))
    right_edge = np.column_stack((x - half_width * normal_x, y - half_width * normal_y))

    return TrackGeometry(x, y, heading, curvature, left_edge, right_edge)
