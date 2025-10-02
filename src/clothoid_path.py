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


def _find_corners(curvature: np.ndarray, tol: float = 1e-9) -> List[Tuple[int, int]]:
    """Identify contiguous segments of non‑zero curvature.

    Segments are split when the curvature drops below ``tol`` or when the
    curvature sign flips, which enables the detection of alternating-sense
    corners such as right–left–right sequences.
    """

    mask = np.abs(curvature) > tol
    if not np.any(mask):
        return []
    idx = np.flatnonzero(mask)
    segments: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    prev_sign = 1.0 if curvature[start] > 0 else -1.0
    sign_tol = max(tol * 1000.0, 1e-6)
    for i in idx[1:]:
        sign = 1.0 if curvature[i] > 0 else -1.0
        if i == prev + 1:
            if sign != prev_sign and (
                abs(curvature[i]) <= sign_tol and abs(curvature[prev]) <= sign_tol
            ):
                sign = prev_sign
            if sign == prev_sign:
                prev = i
                continue
        segments.append((start, prev))
        start = i
        prev = i
        prev_sign = sign
    segments.append((start, prev))
    if not segments:
        return segments

    filtered: List[Tuple[int, int]] = []
    buffer_start: int | None = None
    buffer_end: int | None = None
    for seg_start, seg_end in segments:
        seg_max = float(np.max(np.abs(curvature[seg_start : seg_end + 1])))
        seg_len = seg_end - seg_start + 1
        if seg_len <= 1 or seg_max <= sign_tol:
            if buffer_start is None:
                buffer_start = seg_start
            buffer_end = seg_end
            continue
        if buffer_start is not None:
            seg_start = buffer_start
            buffer_start = None
        filtered.append((seg_start, seg_end))
        buffer_end = None
    if buffer_start is not None and buffer_end is not None:
        filtered.append((buffer_start, buffer_end))
    return filtered


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


def build_clothoid_path(track: TrackGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a simple clothoid racing line for ``track``.

    Parameters
    ----------
    track:
        Track description returned by :func:`geometry.load_track` or
        :func:`geometry.load_track_layout`.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Arrays of arc length ``s``, lateral offset ``e`` and curvature
        ``kappa`` along the constructed racing line.
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
        return s, e, kappa

    width = np.linalg.norm(track.left_edge - track.right_edge, axis=1)
    e = np.zeros(n)

    entry_lengths = (
        np.asarray(track.entry_length, dtype=float)
        if getattr(track, "entry_length", None) is not None
        else np.zeros(n)
    )
    exit_lengths = (
        np.asarray(track.exit_length, dtype=float)
        if getattr(track, "exit_length", None) is not None
        else np.zeros(n)
    )

    apex_fractions = (
        np.asarray(track.apex_fraction, dtype=float)
        if getattr(track, "apex_fraction", None) is not None
        else np.full(n, np.nan)
    )

    apex_radii = (
        np.asarray(track.apex_radius, dtype=float)
        if getattr(track, "apex_radius", None) is not None
        else np.full(n, np.nan)
    )

    entry_offsets = (
        np.asarray(track.entry_offset, dtype=float)
        if getattr(track, "entry_offset", None) is not None
        else np.full(n, np.nan)
    )
    exit_offsets = (
        np.asarray(track.exit_offset, dtype=float)
        if getattr(track, "exit_offset", None) is not None
        else np.full(n, np.nan)
    )
    entry_defined = (
        np.asarray(track.entry_offset_defined, dtype=bool)
        if getattr(track, "entry_offset_defined", None) is not None
        else np.zeros(n, dtype=bool)
    )
    exit_defined = (
        np.asarray(track.exit_offset_defined, dtype=bool)
        if getattr(track, "exit_offset_defined", None) is not None
        else np.zeros(n, dtype=bool)
    )

    # Stay on the outer edge before the first corner.
    first = corners[0]
    corner_offsets: List[dict[str, object]] = []
    for c in corners:
        default_outer = -c.sign * c.width / 2.0
        entry_val = float(entry_offsets[c.start]) if np.isfinite(entry_offsets[c.start]) else default_outer
        exit_val = float(exit_offsets[c.end]) if np.isfinite(exit_offsets[c.end]) else default_outer
        corner_offsets.append(
            {
                "entry": entry_val,
                "exit": exit_val,
                "entry_explicit": bool(entry_defined[c.start] and np.isfinite(entry_offsets[c.start])),
                "exit_explicit": bool(exit_defined[c.end] and np.isfinite(exit_offsets[c.end])),
            }
        )

    for i in range(len(corners) - 1):
        curr = corners[i]
        nxt = corners[i + 1]
        if curr.sign == nxt.sign:
            continue
        if nxt.start > curr.end + 1:
            continue
        gap = curr.end - nxt.start + 1
        blend_to_zero = gap >= 0
        if blend_to_zero:
            if not corner_offsets[i]["exit_explicit"]:
                corner_offsets[i]["exit"] = 0.0
            if not corner_offsets[i + 1]["entry_explicit"]:
                corner_offsets[i + 1]["entry"] = 0.0

    e_outer_first = float(corner_offsets[0]["entry"])

    prev_exit = 0
    prev_offset = e_outer_first
    # Record corner index ranges for post processing of curvature
    corner_sections: List[Tuple[int, int, int, float]] = []

    for i, c in enumerate(corners):
        offsets = corner_offsets[i]
        entry_outer = float(offsets["entry"])
        exit_outer = float(offsets["exit"])
        e_inner = c.sign * c.width / 2.0

        # Determine indices for entry and exit taking into account the
        # requested lengths and track bounds.
        entry_len = float(entry_lengths[c.start])
        exit_len = float(exit_lengths[c.end])

        s_entry_target = s[c.start] - entry_len
        start_idx_raw = int(np.searchsorted(s, s_entry_target, side="left"))
        next_start = corners[i + 1].start if i < len(corners) - 1 else n - 1
        s_exit_target = s[c.end] + exit_len
        end_idx = int(np.searchsorted(s, s_exit_target, side="right") - 1)
        end_idx = min(end_idx, next_start)
        end_idx = max(end_idx, c.end)

        # Blend the preceding straight between the previous and upcoming outer
        # offsets to avoid discontinuities when alternating corner signs.
        overlap = max(prev_exit - start_idx_raw, 0)
        if overlap > 0:
            prev_exit_idx = min(prev_exit, n - 1)
            seg_start = max(prev_exit_idx - overlap, 0)
            seg_end = prev_exit_idx + 1
            if seg_start < seg_end:
                seg = slice(seg_start, seg_end)
                s_start = s[seg_start]
                s_end = s[prev_exit_idx]
                span = max(s_end - s_start, 1e-9)
                u = (s[seg] - s_start) / span
                h = _hermite_step(u)
                e[seg] = prev_offset + (entry_outer - prev_offset) * h
            start_idx = prev_exit
        else:
            start_idx = max(start_idx_raw, prev_exit)
            start_idx = min(start_idx, c.start)

        if start_idx > prev_exit:
            seg = slice(prev_exit, start_idx + 1)
            s_start = s[prev_exit]
            s_end = s[start_idx]
            span = max(s_end - s_start, 1e-9)
            u = (s[seg] - s_start) / span
            h = _hermite_step(u)
            e[seg] = prev_offset + (entry_outer - prev_offset) * h
        elif prev_exit == 0 and i == 0:
            e[:start_idx] = e_outer_first

        start_idx = min(start_idx, end_idx)

        # Determine indices for entry, apex and exit.
        apex_val = float(apex_fractions[c.start])
        if not np.isfinite(apex_val):
            apex_val = 0.5
        apex_val = float(np.clip(apex_val, 0.0, 1.0))
        apex_idx = start_idx + int(apex_val * (end_idx - start_idx))

        s_entry = s[start_idx]
        s_apex = s[apex_idx]
        s_exit = s[end_idx]

        # Entry spiral: outer -> inner.
        seg1 = slice(start_idx, apex_idx + 1)
        u = (s[seg1] - s_entry) / max(s_apex - s_entry, 1e-9)
        h = _hermite_step(u)
        e[seg1] = entry_outer + (e_inner - entry_outer) * h

        # Exit spiral: inner -> outer.
        seg2 = slice(apex_idx, end_idx + 1)
        u = (s[seg2] - s_apex) / max(s_exit - s_apex, 1e-9)
        h = _hermite_step(u)
        e[seg2] = e_inner + (exit_outer - e_inner) * h

        # Store indices for curvature shaping: start, apex and end
        corner_sections.append((start_idx, apex_idx, end_idx, c.sign))

        prev_exit = end_idx + 1
        prev_offset = exit_outer

    # After the last corner remain on the outer edge of the first corner to
    # ensure continuity for closed tracks.
    if prev_exit < n:
        e[prev_exit:] = e_outer_first

    spline = LateralOffsetSpline(s, e)
    kappa = path_curvature(s, spline, kappa_c)

    # Shape curvature using piecewise linear profiles for each corner.
    for start_idx, apex_idx, end_idx, sign in corner_sections:
        start_abs = abs(kappa[start_idx])
        end_abs = abs(kappa[end_idx])
        apex_radius = float(apex_radii[start_idx])
        if np.isfinite(apex_radius) and apex_radius > 0:
            apex_abs = 1.0 / apex_radius
        else:
            apex_abs = abs(kappa[apex_idx])
        apex_abs = max(apex_abs, start_abs, end_abs)

        pre = np.linspace(start_abs, apex_abs, apex_idx - start_idx + 1)
        post = np.linspace(apex_abs, end_abs, end_idx - apex_idx + 1)

        kappa[start_idx : apex_idx + 1] = sign * pre
        kappa[apex_idx : end_idx + 1] = sign * post
    # Reparameterise by the true arc length of the offset path.
    # The trajectory is defined relative to the centreline by a lateral offset
    # ``e(s)``. The derivative of the path position with respect to the
    # centreline arc length is ``(1 - e * kappa_c, e_s)`` in the Frenet frame.
    # Its Euclidean norm gives the local scaling between centreline and path
    # arc length.
    e_s = spline.first_derivative(s)
    scale = np.sqrt((1.0 - e * kappa_c) ** 2 + e_s**2)
    s_true = np.zeros_like(s)
    if len(s) > 1:
        s_true[1:] = np.cumsum(0.5 * (scale[1:] + scale[:-1]) * np.diff(s))

    return s_true, e, kappa


# Provide an alias with a more descriptive name.
def build_racing_line(track: TrackGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_clothoid_path(track)

