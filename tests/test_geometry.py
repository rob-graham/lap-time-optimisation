import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src directory on path for test discovery when pytest runs directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track


def test_circle_track_length_and_curvature(tmp_path: Path) -> None:
    R = 50.0
    theta = np.linspace(0.0, 2 * np.pi, 100, endpoint=False)
    df = pd.DataFrame(
        {
            "x_m": R * np.cos(theta),
            "y_m": R * np.sin(theta),
            "width_m": np.full(theta.shape, 10.0),
        }
    )
    track_file = tmp_path / "circle.csv"
    df.to_csv(track_file, index=False)

    x, y, heading, curvature, left, right = load_track(track_file, ds=1.0)

    length = np.sum(
        np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
    )
    assert np.isclose(length, 2 * np.pi * R, atol=1.0)
    assert np.all(curvature > -1e-6)
