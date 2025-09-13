from __future__ import annotations

"""Racing line construction using simple cubic spirals.

This module provides helper functions to build a heuristic "racing line" from a
:class:`geometry.TrackGeometry` instance.  Corners are detected from regions of
non‑zero centreline curvature.  For each corner the rider is assumed to start on
the outer track edge, clip the inner apex and return to the outer edge.  The
transition between these points is achieved using cubic polynomials in
arc‑length which ensure continuity of position, heading and curvature
(``G^2`` continuity).  The resulting path is described by arrays of arc length
``s`` and curvature ``kappa`` which can be fed directly to
:func:`speed_solver.solve_speed_profile`.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ``clothoid_path`` can be imported either as part of the ``src`` package or as
# a stand‑alone module.  Try a relative import first and fall back to an absolute
# import for direct execution support.
try:  # pragma: no cover - import shim
    from .geometry import TrackGeometry
    from .path_param import LateralOffsetSpline, path_curvature
except ImportError:  # pragma: no cover - direct execution support
    from geometry import TrackGeometry
    from path_param import LateralOffsetSpline, path_curvature


@dataclass
class Corner:
    """Representation of a discrete corner in the track."""

    start: int  # index of the first point of the corner in the arrays
    end: int  # index of the last point of the corner
    sign: float  # sign of curvature (+1 left, -1 right)
    width: float  # mean track width over the corner


def _find_corners(curvature: np.ndarray) -> List[Tuple[int, int]]:
    """Identify contiguous segments of non‑zero curvature."""

    mask = np.abs(curvature) > 1e-9
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    segments: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segments.append((start, prev))
        start = i
        prev = i
    segments.append((start, prev))
    return segments


def _corner_data(track: TrackGeometry) -> List[Corner]:
    """Extract corner information from ``track``."""

    width = np.linalg.norm(track.left_edge - track.right_edge, axis=1)
    segments = _find_corners(track.curvature)
    corners: List[Corner] = []
    for s, e in segments:
        sign = float(np.sign(np.mean(track.curvature[s : e + 1])))
        if sign == 0:
            sign = 1.0
        w = float(np.mean(width[s : e + 1]))
        corners.append(Corner(s, e, sign, w))
    return corners


def _hermite_step(u: np.ndarray) -> np.ndarray:
    """Cubic step ``3u^2 - 2u^3`` with zero slope at both ends."""

    return 3.0 * u**2 - 2.0 * u**3


def build_clothoid_path(track: TrackGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a simple clothoid racing line for ``track``.

    Parameters
    ----------
    track:
        Track description returned by :func:`geometry.load_track` or
        :func:`geometry.load_track_layout`.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Arrays of lateral offset ``e`` and curvature ``kappa`` along the
        constructed racing line.
    """

    x = np.asarray(track.x, dtype=float)
    y = np.asarray(track.y, dtype=float)
    kappa_c = np.asarray(track.curvature, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("track must contain at least two samples")

    # Arc‑length coordinates of the centreline.
    s = np.zeros(n)
    s[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))

    corners = _corner_data(track)

    # If the track has no corners the racing line follows the centreline.
    if not corners:
        e = np.zeros_like(s)
        spline = LateralOffsetSpline(s, e)
        kappa = path_curvature(s, spline, kappa_c)
        return e, kappa

    width = np.linalg.norm(track.left_edge - track.right_edge, axis=1)
    e = np.zeros(n)

    # Stay on the outer edge before the first corner.
    first = corners[0]
    e_outer_first = -first.sign * first.width / 2.0
    e[: first.start] = e_outer_first

    for i, c in enumerate(corners):
        e_outer = -c.sign * c.width / 2.0
        e_inner = c.sign * c.width / 2.0

        # Fill preceding straight with the correct outer offset.
        if i > 0:
            prev_end = corners[i - 1].end
            e[prev_end + 1 : c.start] = e_outer

        # Determine indices for entry, apex and exit.
        mid = (c.start + c.end) // 2
        s_entry = s[c.start]
        s_apex = s[mid]
        s_exit = s[c.end]

        # Entry spiral: outer -> inner.
        seg1 = slice(c.start, mid + 1)
        u = (s[seg1] - s_entry) / max(s_apex - s_entry, 1e-9)
        h = _hermite_step(u)
        e[seg1] = e_outer + (e_inner - e_outer) * h

        # Exit spiral: inner -> outer.
        seg2 = slice(mid, c.end + 1)
        u = (s[seg2] - s_apex) / max(s_exit - s_apex, 1e-9)
        h = _hermite_step(u)
        e[seg2] = e_inner + (e_outer - e_inner) * h

    # After the last corner remain on the outer edge of the first corner to
    # ensure continuity for closed tracks.
    last_end = corners[-1].end
    e[last_end + 1 :] = e_outer_first

    spline = LateralOffsetSpline(s, e)
    kappa = path_curvature(s, spline, kappa_c)
    return e, kappa


# Provide an alias with a more descriptive name.
def build_racing_line(track: TrackGeometry) -> Tuple[np.ndarray, np.ndarray]:
    return build_clothoid_path(track)

