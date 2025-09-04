import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src package on path for test discovery when pytest runs directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.run_demo import run


def test_one_corner_track_open_track() -> None:
    out_dir = run(
        "data/oneCornerTrack.csv",
        "data/bike_params_sv650.csv",
        ds=1.0,
        buffer=0.5,
        n_ctrl=20,
    )
    geom = pd.read_csv(out_dir / "geometry.csv")
    x = geom["x_center_m"].to_numpy()
    y = geom["y_center_m"].to_numpy()
    assert not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1]))
    length = geom["s_m"].iloc[-1]
    assert np.isclose(length, 327.0, atol=1e-6)
