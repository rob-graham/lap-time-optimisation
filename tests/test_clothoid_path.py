import sys
from pathlib import Path

import numpy as np

# Ensure ``src`` directory is on the import path for the tests
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track_layout
from clothoid_path import build_clothoid_path
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

