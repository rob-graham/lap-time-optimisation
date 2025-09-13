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
    s, kappa = build_clothoid_path(geom)
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

