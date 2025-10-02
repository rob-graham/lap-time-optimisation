import sys
from pathlib import Path

import numpy as np

# Ensure ``src`` directory is on the import path for the tests
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import TrackGeometry, load_track_layout
from clothoid_path import build_clothoid_path, _corner_data
from io_utils import read_bike_params_csv
from speed_solver import solve_speed_profile


def test_clothoid_path_speed_profile(tmp_path: Path) -> None:
    """Racing line generation and speed profile solve on a sample track."""

    track_csv = tmp_path / "sample_track.csv"
    track_csv.write_text(
        "\n".join(
            [
                "x_m,y_m,section_type,radius_m,width_m,camber_rad,grade_rad,apex_fraction,entry_length_m,exit_length_m",
                "0,0,straight,inf,8,0,0,,0,0",
                "0,50,corner,30,8,0,0,0.5,10,10",
                "50,50,straight,inf,8,0,0,,0,0",
            ]
        )
    )

    geom = load_track_layout(track_csv, ds=1.0, closed=False)
    s, offset, kappa = build_clothoid_path(geom)
    # The returned ``s`` should reflect the arc length of the offset path and
    # therefore differ from the centreline arc length.
    s_center = np.zeros_like(geom.x)
    s_center[1:] = np.cumsum(np.hypot(np.diff(geom.x), np.diff(geom.y)))
    assert not np.isclose(s[-1], s_center[-1])

    assert geom.apex_fraction is not None
    assert np.nanmax(geom.apex_fraction) == 0.5

    params = read_bike_params_csv(Path(__file__).resolve().parents[1] / "data" / "bike_params_r6.csv")
    v, ax, ay, limit, lap_time, iterations, elapsed_s = solve_speed_profile(
        s,
        kappa,
        mu=params["mu"],
        a_wheelie_max=params["a_wheelie_max"],
        a_brake=params["a_brake"],
        v_start=0.0,
        v_end=0.0,
    )

    assert np.all(np.isfinite(v))
    dkappa = np.diff(kappa)
    assert np.all(np.isfinite(dkappa))
    assert np.max(np.abs(dkappa)) < 0.2


def test_entry_exit_lengths(tmp_path: Path) -> None:
    """Entry/exit lengths span multiple samples and respect track bounds."""

    header = (
        "x_m,y_m,section_type,radius_m,width_m,camber_rad,grade_rad,"
        "apex_fraction,entry_length_m,exit_length_m"
    )

    # Entry/exit lengths that fit within the surrounding straights.
    track_csv = tmp_path / "track.csv"
    track_csv.write_text(
        "\n".join(
            [
                header,
                "0,0,straight,inf,8,0,0,,0,0",
                "0,40,corner,30,8,0,0,0.5,5,7",
                "40,40,straight,inf,8,0,0,,0,0",
                "80,40,straight,inf,8,0,0,,0,0",
            ]
        )
    )

    geom = load_track_layout(track_csv, ds=1.0, closed=False)
    s, offset, _ = build_clothoid_path(geom)

    # Locate the corner and expected start/end of the transition region.
    idx = np.flatnonzero(np.abs(geom.curvature) > 1e-9)
    cs, ce = int(idx[0]), int(idx[-1])
    entry_len = float(geom.entry_length[cs])
    exit_len = float(geom.exit_length[ce])
    start_idx = int(np.searchsorted(s, s[cs] - entry_len, side="left"))
    end_idx = int(np.searchsorted(s, s[ce] + exit_len, side="right") - 1)

    assert start_idx < cs
    assert end_idx > ce
    ds = float(s[1] - s[0])
    assert abs((s[cs] - s[start_idx]) - entry_len) <= ds
    assert abs((s[end_idx] - s[ce]) - exit_len) <= ds

    width = np.linalg.norm(geom.left_edge - geom.right_edge, axis=1)
    mean_width = float(np.mean(width[cs : ce + 1]))
    sign = float(np.sign(np.mean(geom.curvature[cs : ce + 1])))
    outer = -sign * mean_width / 2.0
    assert np.allclose(offset[:start_idx], outer)
    assert np.allclose(offset[end_idx + 1 :], outer)

    # Edge case: requested lengths exceed the available straight sections.
    track_csv.write_text(
        "\n".join(
            [
                header,
                "0,0,straight,inf,8,0,0,,0,0",
                "0,20,corner,30,8,0,0,0.5,50,50",
                "20,20,straight,inf,8,0,0,,0,0",
                "40,20,straight,inf,8,0,0,,0,0",
            ]
        )
    )

    geom2 = load_track_layout(track_csv, ds=1.0, closed=False)
    s2, offset2, _ = build_clothoid_path(geom2)

    idx2 = np.flatnonzero(np.abs(geom2.curvature) > 1e-9)
    cs2, ce2 = int(idx2[0]), int(idx2[-1])
    entry_len2 = float(geom2.entry_length[cs2])
    exit_len2 = float(geom2.exit_length[ce2])
    start_idx2 = int(np.searchsorted(s2, s2[cs2] - entry_len2, side="left"))
    end_idx2 = int(np.searchsorted(s2, s2[ce2] + exit_len2, side="right") - 1)

    assert start_idx2 == 0
    assert end_idx2 == len(s2) - 1
    assert (s2[cs2] - s2[start_idx2]) < entry_len2
    assert (s2[end_idx2] - s2[ce2]) < exit_len2


def test_alternating_corners_respect_offsets(tmp_path: Path) -> None:
    """Alternating corners blend defaults but honour explicit offsets."""

    header = (
        "x_m,y_m,section_type,radius_m,width_m,camber_rad,grade_rad,"
        "apex_fraction,entry_length_m,exit_length_m,entry_offset_m,exit_offset_m"
    )

    esse_rows = [
        header,
        "0,0,straight,inf,8,0,0,,0,0,,",
        "0,40,corner,25,8,0,0,,0,0,,",
        "30,70,corner,-25,8,0,0,,0,0,,",
        "60,40,straight,inf,8,0,0,,0,0,,",
        "60,0,straight,inf,8,0,0,,0,0,,",
    ]

    track_csv = tmp_path / "esse.csv"
    track_csv.write_text("\n".join(esse_rows))

    geom = load_track_layout(track_csv, ds=1.0, closed=False)
    s, offset, _ = build_clothoid_path(geom)
    corners = _corner_data(geom)
    assert len(corners) >= 2
    first, second = corners[0], corners[1]

    # Default offsets blended to centre for alternating corners.
    assert np.isclose(offset[first.end], 0.0, atol=1e-6)
    assert np.isclose(offset[second.start], 0.0, atol=1e-6)

    # Explicit offsets should be respected.
    esse_rows_explicit = [
        header,
        "0,0,straight,inf,8,0,0,,0,0,,",
        "0,40,corner,25,8,0,0,,0,0,-3,-3",
        "30,70,corner,-25,8,0,0,,0,0,3,3",
        "60,40,straight,inf,8,0,0,,0,0,,",
        "60,0,straight,inf,8,0,0,,0,0,,",
    ]

    track_csv.write_text("\n".join(esse_rows_explicit))
    geom2 = load_track_layout(track_csv, ds=1.0, closed=False)
    _, offset2, _ = build_clothoid_path(geom2)
    corners2 = _corner_data(geom2)
    first2, second2 = corners2[0], corners2[1]

    assert np.isclose(offset2[first2.end], -3.0, atol=1e-6)
    assert np.isclose(offset2[second2.start], 3.0, atol=1e-6)

def test_apex_fraction_shifts_apex(tmp_path: Path) -> None:
    """Apex fraction moves the apex away from the midpoint."""

    track_csv = tmp_path / "asym.csv"
    track_csv.write_text(
        "\n".join(
            [
                "x_m,y_m,section_type,radius_m,width_m,camber_rad,grade_rad,apex_fraction,entry_length_m,exit_length_m",
                "0,0,straight,inf,8,0,0,,0,0",
                "0,40,corner,30,8,0,0,0.2,0,0",
                "40,40,straight,inf,8,0,0,,0,0",
            ]
        )
    )

    geom = load_track_layout(track_csv, ds=1.0, closed=False)
    s, offset, _ = build_clothoid_path(geom)

    idx = np.flatnonzero(np.abs(geom.curvature) > 1e-9)
    start_idx, end_idx = int(idx[0]), int(idx[-1])
    width = np.linalg.norm(geom.left_edge - geom.right_edge, axis=1)
    mean_width = float(np.mean(width[start_idx : end_idx + 1]))
    sign = float(np.sign(np.mean(geom.curvature[start_idx : end_idx + 1])))
    e_inner = sign * mean_width / 2.0

    expected_apex = start_idx + int(0.2 * (end_idx - start_idx))
    if e_inner > 0:
        apex_idx = start_idx + int(np.argmax(offset[start_idx : end_idx + 1]))
    else:
        apex_idx = start_idx + int(np.argmin(offset[start_idx : end_idx + 1]))
    assert apex_idx == expected_apex
    assert np.isclose(offset[expected_apex], e_inner)


def test_nan_apex_fraction_midpoint(tmp_path: Path) -> None:
    """NaN apex fraction falls back to the midpoint."""

    track_csv = tmp_path / "nan_apex.csv"
    track_csv.write_text(
        "\n".join(
            [
                "x_m,y_m,section_type,radius_m,width_m,camber_rad,grade_rad,apex_fraction,entry_length_m,exit_length_m",
                "0,0,straight,inf,8,0,0,,0,0",
                "0,40,corner,30,8,0,0,nan,0,0",
                "40,40,straight,inf,8,0,0,,0,0",
            ]
        )
    )

    geom = load_track_layout(track_csv, ds=1.0, closed=False)
    assert np.isnan(geom.apex_fraction).any()
    s, offset, _ = build_clothoid_path(geom)

    idx = np.flatnonzero(np.abs(geom.curvature) > 1e-9)
    start_idx, end_idx = int(idx[0]), int(idx[-1])
    width = np.linalg.norm(geom.left_edge - geom.right_edge, axis=1)
    mean_width = float(np.mean(width[start_idx : end_idx + 1]))
    sign = float(np.sign(np.mean(geom.curvature[start_idx : end_idx + 1])))
    e_inner = sign * mean_width / 2.0

    expected_apex = start_idx + int(0.5 * (end_idx - start_idx))
    if e_inner > 0:
        apex_idx = start_idx + int(np.argmax(offset[start_idx : end_idx + 1]))
    else:
        apex_idx = start_idx + int(np.argmin(offset[start_idx : end_idx + 1]))
    assert apex_idx == expected_apex
    assert np.isclose(offset[expected_apex], e_inner)


def test_overlapping_entries_blend_monotonically() -> None:
    """Overlapping alternating corners blend outer offsets smoothly."""

    n = 80
    x = np.arange(n, dtype=float)
    y = np.zeros(n)
    heading = np.zeros(n)
    curvature = np.zeros(n)
    width = 10.0

    curvature[20:25] = 0.02
    curvature[40:45] = -0.02

    left_edge = np.column_stack((x, np.full(n, width / 2.0)))
    right_edge = np.column_stack((x, np.full(n, -width / 2.0)))

    apex_fraction = np.full(n, 0.5)
    entry_length = np.zeros(n)
    exit_length = np.zeros(n)

    entry_length[20] = 8.0
    exit_length[24] = 8.0
    entry_length[40] = 8.0
    exit_length[44] = 8.0

    geom = TrackGeometry(
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
        left_edge=left_edge,
        right_edge=right_edge,
        apex_fraction=apex_fraction,
        entry_length=entry_length,
        exit_length=exit_length,
        apex_radius=None,
    )

    s, offset, _ = build_clothoid_path(geom)
    corners = _corner_data(geom)
    assert len(corners) >= 2
    assert corners[0].sign != corners[1].sign

    arc = np.zeros_like(geom.x, dtype=float)
    arc[1:] = np.cumsum(np.hypot(np.diff(geom.x), np.diff(geom.y)))
    entry = np.asarray(geom.entry_length, dtype=float)
    exit = np.asarray(geom.exit_length, dtype=float)

    prev_exit = 0
    prev_outer_offset = -corners[0].sign * corners[0].width / 2.0
    overlap_info = None
    for idx, corner in enumerate(corners):
        entry_len = float(entry[corner.start])
        exit_len = float(exit[corner.end])

        s_entry_target = arc[corner.start] - entry_len
        start_idx_raw = int(np.searchsorted(arc, s_entry_target, side="left"))

        next_start = corners[idx + 1].start if idx < len(corners) - 1 else len(arc) - 1
        s_exit_target = arc[corner.end] + exit_len
        end_idx = int(np.searchsorted(arc, s_exit_target, side="right") - 1)
        end_idx = min(end_idx, next_start)
        end_idx = max(end_idx, corner.end)

        overlap = max(prev_exit - start_idx_raw, 0)
        if overlap > 0:
            shared_start = max(prev_exit - overlap, 0)
            shared_end = min(prev_exit, len(offset) - 1)
            overlap_info = (
                shared_start,
                shared_end,
                prev_outer_offset,
                -corner.sign * corner.width / 2.0,
            )
            start_idx = prev_exit
        else:
            start_idx = max(start_idx_raw, prev_exit)
            start_idx = min(start_idx, corner.start)

        prev_exit = end_idx + 1
        prev_outer_offset = -corner.sign * corner.width / 2.0

    assert overlap_info is not None
    shared_start, shared_end, prev_outer, next_outer = overlap_info
    segment = offset[shared_start : shared_end + 1]
    assert segment.size >= 2
    assert np.isclose(segment[0], prev_outer)
    assert np.isclose(segment[-1], next_outer)

    diffs = np.diff(segment)
    if next_outer >= prev_outer:
        assert np.all(diffs >= -1e-9)
    else:
        assert np.all(diffs <= 1e-9)

def test_right_left_right_corners_split_and_hit_apexes() -> None:
    """Alternating corners are detected separately and reach their apices."""

    n = 200
    x = np.linspace(0.0, 199.0, n)
    y = np.zeros(n)
    heading = np.zeros(n)
    curvature = np.zeros(n)

    width = 10.0
    left_edge = np.column_stack((x, np.full(n, width / 2.0)))
    right_edge = np.column_stack((x, np.full(n, -width / 2.0)))

    apex_fraction = np.full(n, 0.5)
    entry_length = np.zeros(n)
    exit_length = np.zeros(n)

    segments = [
        (40, 60, -0.02),
        (80, 100, 0.02),
        (120, 140, -0.02),
    ]

    for start, end, kappa in segments:
        curvature[start : end + 1] = kappa
        entry_length[start] = 8.0
        exit_length[end] = 8.0

    geom = TrackGeometry(
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
        left_edge=left_edge,
        right_edge=right_edge,
        apex_fraction=apex_fraction,
        entry_length=entry_length,
        exit_length=exit_length,
        apex_radius=None,
    )

    corners = _corner_data(geom)
    signs = [c.sign for c in corners]

    esse_start = None
    for idx in range(len(signs) - 2):
        if signs[idx : idx + 3] == [-1.0, 1.0, -1.0]:
            esse_start = idx
            break

    assert esse_start is not None, "Expected right-left-right sequence not found"

    esse_corners = corners[esse_start : esse_start + 3]
    assert len(esse_corners) == 3
    assert esse_corners[0].end < esse_corners[1].start
    assert esse_corners[1].end < esse_corners[2].start

    s, offset, _ = build_clothoid_path(geom)

    arc = np.zeros_like(geom.x, dtype=float)
    arc[1:] = np.cumsum(np.hypot(np.diff(geom.x), np.diff(geom.y)))
    entry = np.asarray(geom.entry_length, dtype=float)
    exit = np.asarray(geom.exit_length, dtype=float)
    apex_frac = np.asarray(geom.apex_fraction, dtype=float)

    prev_exit = 0
    outer_offsets = []
    corner_ranges = []
    for idx, corner in enumerate(corners):
        entry_len = float(entry[corner.start])
        exit_len = float(exit[corner.end])
        s_entry_target = arc[corner.start] - entry_len
        start_idx = int(np.searchsorted(arc, s_entry_target, side="left"))
        start_idx = max(start_idx, prev_exit)
        start_idx = min(start_idx, corner.start)

        next_start = corners[idx + 1].start if idx < len(corners) - 1 else len(arc) - 1
        s_exit_target = arc[corner.end] + exit_len
        end_idx = int(np.searchsorted(arc, s_exit_target, side="right") - 1)
        end_idx = min(end_idx, next_start)
        end_idx = max(end_idx, corner.end)

        if esse_start <= idx < esse_start + 3:
            inner = corner.sign * corner.width / 2.0
            seg = offset[start_idx : end_idx + 1]
            if corner.sign > 0:
                apex_value = np.max(seg)
            else:
                apex_value = np.min(seg)
            assert np.isclose(apex_value, inner, atol=1e-6)

        apex_val = float(apex_frac[corner.start])
        if not np.isfinite(apex_val):
            apex_val = 0.5
        apex_val = float(np.clip(apex_val, 0.0, 1.0))
        apex_idx = start_idx + int(apex_val * (end_idx - start_idx))
        assert start_idx <= apex_idx <= end_idx

        outer_offsets.append(-corner.sign * corner.width / 2.0)
        corner_ranges.append((start_idx, end_idx))
        prev_exit = end_idx + 1

    # Offsets between alternating corners should blend smoothly instead of
    # jumping when the outer edge changes side.
    assert len(corner_ranges) == len(outer_offsets) == len(corners)
    for idx in range(esse_start, esse_start + 2):
        start_idx, end_idx = corner_ranges[idx]
        next_start_idx, _ = corner_ranges[idx + 1]
        if next_start_idx <= end_idx + 1:
            continue
        straight = offset[end_idx + 1 : next_start_idx + 1]
        prev_outer = outer_offsets[idx]
        next_outer = outer_offsets[idx + 1]
        assert np.isclose(straight[0], prev_outer, atol=1e-6)
        assert np.isclose(straight[-1], next_outer, atol=1e-6)
        if not np.isclose(prev_outer, next_outer, atol=1e-6):
            # Ensure the offset actually transitions between the two outer edges
            # with a smooth, monotonic variation.
            diffs = np.diff(straight)
            assert np.any(np.abs(diffs) > 1e-6)
            assert np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6)


def test_curvature_radius_profile(tmp_path: Path) -> None:
    """Curvature decreases radius to the apex and increases afterwards."""

    track_csv = tmp_path / "corner.csv"
    track_csv.write_text(
        "\n".join(
            [
                "x_m,y_m,section_type,radius_m,apex_radius_m,width_m,camber_rad,grade_rad,apex_fraction,entry_length_m,exit_length_m",
                "0,0,straight,inf,inf,8,0,0,,0,0",
                "0,50,corner,30,20,8,0,0,0.5,0,0",
                "50,50,straight,inf,inf,8,0,0,,0,0",
            ]
        )
    )

    geom = load_track_layout(track_csv, ds=1.0, closed=False)
    s, _, kappa = build_clothoid_path(geom)

    idx = np.flatnonzero(np.abs(geom.curvature) > 1e-9)
    start_idx, end_idx = int(idx[0]), int(idx[-1])
    apex_idx = start_idx + (end_idx - start_idx) // 2

    before = np.abs(kappa[start_idx : apex_idx + 1])
    after = np.abs(kappa[apex_idx : end_idx + 1])

    # Curvature varies linearly towards and away from the apex
    assert np.allclose(np.diff(before), before[1] - before[0])
    assert np.allclose(np.diff(after), after[1] - after[0])
    # Apex has the maximum curvature magnitude
    assert before[-1] >= before[0]
    assert after[0] >= after[-1]

