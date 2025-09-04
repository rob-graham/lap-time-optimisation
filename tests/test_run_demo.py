import sys
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

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
