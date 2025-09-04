import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src directory on path for test discovery when pytest runs directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track_layout


def test_circle_track_length_and_curvature(tmp_path: Path) -> None:
    R = 50.0
    data = [
        {"x_m": R, "y_m": 0.0, "section_type": "corner", "radius_m": R, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
        {"x_m": 0.0, "y_m": R, "section_type": "corner", "radius_m": R, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
        {"x_m": -R, "y_m": 0.0, "section_type": "corner", "radius_m": R, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
        {"x_m": 0.0, "y_m": -R, "section_type": "corner", "radius_m": R, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
    ]
    df = pd.DataFrame(data)
    track_file = tmp_path / "circle.csv"
    df.to_csv(track_file, index=False)

    geom = load_track_layout(track_file, ds=1.0)

    x, y = geom.x, geom.y
    length = np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])))
    assert np.isclose(length, 2 * np.pi * R, atol=1.0)
    assert np.allclose(geom.curvature, 1.0 / R, atol=1e-2)


def test_right_hand_corner_continuity(tmp_path: Path) -> None:
    data = [
        {"x_m": 0.0, "y_m": 0.0, "section_type": "straight", "radius_m": 0.0, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
        {"x_m": 10.0, "y_m": 0.0, "section_type": "corner", "radius_m": -5.0, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
        {"x_m": 10.0, "y_m": -10.0, "section_type": "straight", "radius_m": 0.0, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
        {"x_m": 0.0, "y_m": -10.0, "section_type": "corner", "radius_m": -5.0, "width_m": 10.0,
         "camber_rad": 0.0, "grade_rad": 0.0},
    ]
    df = pd.DataFrame(data)
    track_file = tmp_path / "right_hand.csv"
    df.to_csv(track_file, index=False)

    geom = load_track_layout(track_file, ds=1.0)

    node = np.array([data[1]["x_m"], data[1]["y_m"]])
    curvature = geom.curvature
    idx = np.where(curvature != 0)[0][0]
    corner_pt = np.array([geom.x[idx], geom.y[idx]])
    prev_pt = np.array([geom.x[idx - 1], geom.y[idx - 1]])

    assert np.isclose(np.linalg.norm(corner_pt - node), 1.0, atol=5e-3)
    assert np.isclose(np.linalg.norm(prev_pt - node), 1.0, atol=5e-3)
