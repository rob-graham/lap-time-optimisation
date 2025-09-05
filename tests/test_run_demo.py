import sys
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import pytest

# Ensure src package on path for test discovery when pytest runs directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.run_demo import run


def test_one_corner_track_open_track(capfd) -> None:
    lap_time, out_dir = run(
        "data/oneCornerTrack.csv",
        "data/bike_params_sv650.csv",
        ds=1.0,
        buffer=0.5,
        n_ctrl=20,
        closed=False,
        speed_tol=1e-2,
        path_tol=1e-2,
    )

    out = capfd.readouterr().out
    assert re.search(r"Path optimisation: \d+ iterations", out)
    assert re.search(r"Speed solver: \d+ iterations", out)
    assert re.search(r"Total runtime: [0-9.]+ s", out)

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    with summary_path.open() as f:
        summary = json.load(f)
    assert summary["lap_time_s"] > 0
    assert np.isclose(summary["lap_time_s"], lap_time)
    geom = pd.read_csv(out_dir / "geometry.csv")
    x = geom["x_center_m"].to_numpy()
    y = geom["y_center_m"].to_numpy()
    assert not np.allclose([x[0], y[0]], [x[-1], y[-1]])
    length = geom["s_m"].iloc[-1]
    assert np.isclose(length, 327.0, atol=1e-6)
    for col in ["x_left_m", "y_left_m", "x_right_m", "y_right_m"]:
        assert col in geom.columns
    for col in ["x_inner_m", "y_inner_m", "x_outer_m", "y_outer_m"]:
        assert col not in geom.columns
    results = pd.read_csv(out_dir / "results.csv")
    for col in ["x_left_m", "y_left_m", "x_right_m", "y_right_m"]:
        assert col in results.columns


def test_rpm_capped_top_gear(tmp_path) -> None:
    """RPM in top gear should not exceed the shift limit."""

    # Create a bike parameter file with a very low shift RPM so that
    # the unconstrained speed solver would exceed it in top gear.
    bike_src = Path("data/bike_params_sv650.csv").read_text().splitlines()
    modified = [
        "shift_rpm,2000" if line.startswith("shift_rpm") else line
        for line in bike_src
    ]
    bike_file = tmp_path / "bike_params_low_shift.csv"
    bike_file.write_text("\n".join(modified))

    lap_time, out_dir = run(
        "data/track_layout.csv",
        str(bike_file),
        ds=1.0,
        buffer=0.5,
        n_ctrl=20,
        closed=True,
        path_tol=1e-2,
    )

    results = pd.read_csv(out_dir / "results.csv")
    shift_rpm = 2000
    top_gear = results["gear"].max()
    rpm_top = results.loc[results["gear"] == top_gear, "rpm"]
    assert np.all(rpm_top <= shift_rpm)


def test_run_raises_without_gears(tmp_path) -> None:
    """run() should raise a clear error when no gear ratios are supplied."""

    bike_src = Path("data/bike_params_sv650.csv").read_text().splitlines()
    modified = [line for line in bike_src if not line.startswith("gear")]
    bike_file = tmp_path / "bike_params_no_gears.csv"
    bike_file.write_text("\n".join(modified))

    with pytest.raises(ValueError, match="No gear ratios provided"):
        run(
            "data/oneCornerTrack.csv",
            str(bike_file),
            ds=1.0,
            buffer=0.5,
            n_ctrl=20,
            closed=False,
            speed_tol=1e-2,
            path_tol=1e-2,
        )
