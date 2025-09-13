import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src directory on path for test discovery when pytest runs directly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track_layout, load_track


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
    dist_left = np.hypot(geom.left_edge[:, 0], geom.left_edge[:, 1])
    dist_right = np.hypot(geom.right_edge[:, 0], geom.right_edge[:, 1])
    assert np.allclose(dist_left, R - 5.0, atol=1.0)
    assert np.allclose(dist_right, R + 5.0, atol=1.0)


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
    width = np.linalg.norm(geom.left_edge - geom.right_edge, axis=1)
    assert np.allclose(width, 10.0)
    assert np.linalg.norm(geom.right_edge[idx] - node) < np.linalg.norm(
        geom.left_edge[idx] - node
    )


def test_open_track_endpoints_and_length(tmp_path: Path) -> None:
    data = {
        "x_m": [0.0, 0.0, 100.0],
        "y_m": [0.0, 100.0, 100.0],
        "width_m": [10.0, 10.0, 10.0],
    }
    df = pd.DataFrame(data)
    track_file = tmp_path / "open.csv"
    df.to_csv(track_file, index=False)

    ds = 1.0
    geom = load_track(str(track_file), ds=ds, closed=False)
    x, y = geom.x, geom.y

    # Start and end points should be distinct for an open track.
    assert not np.allclose([x[0], y[0]], [x[-1], y[-1]])

    # The path length should match the length computed from the nodes.
    expected_length = np.sum(
        np.hypot(np.diff(df["x_m"].to_numpy()), np.diff(df["y_m"].to_numpy()))
    )
    length = np.sum(np.hypot(np.diff(x), np.diff(y))) + ds
    assert np.isclose(length, expected_length, atol=1.0e-6)
    width = np.linalg.norm(geom.left_edge - geom.right_edge, axis=1)
    assert np.allclose(width, 10.0)


def test_load_track_missing_columns(tmp_path: Path) -> None:
    data = {"x_m": [0.0, 1.0], "y_m": [0.0, 1.0]}
    df = pd.DataFrame(data)
    track_file = tmp_path / "missing.csv"
    df.to_csv(track_file, index=False)

    with pytest.raises(ValueError, match="width_m"):
        load_track(str(track_file), ds=1.0)


def test_load_track_layout_missing_columns(tmp_path: Path) -> None:
    data = {"x_m": [0.0], "y_m": [0.0], "section_type": ["straight"], "radius_m": [0.0]}
    df = pd.DataFrame(data)
    track_file = tmp_path / "missing_layout.csv"
    df.to_csv(track_file, index=False)

    with pytest.raises(ValueError, match="width_m"):
        load_track_layout(str(track_file), ds=1.0)


def test_optional_corner_metadata(tmp_path: Path) -> None:
    data = [
        {"x_m": 0.0, "y_m": 0.0, "section_type": "straight", "radius_m": 0.0, "width_m": 8.0},
        {
            "x_m": 10.0,
            "y_m": 0.0,
            "section_type": "corner",
            "radius_m": 10.0,
            "width_m": 8.0,
            "apex_fraction": 0.3,
        },
        {"x_m": 10.0, "y_m": 10.0, "section_type": "straight", "radius_m": 0.0, "width_m": 8.0},
    ]
    df = pd.DataFrame(data)
    track_file = tmp_path / "apex.csv"
    df.to_csv(track_file, index=False)
    geom = load_track_layout(track_file, ds=1.0, closed=False)

    mask = np.isfinite(geom.apex_fraction)
    assert mask.any()
    assert np.allclose(geom.apex_fraction[mask], 0.3)
    assert np.allclose(geom.entry_length[mask], 0.0)
    assert np.allclose(geom.exit_length[mask], 0.0)
    assert geom.apex_radius is not None
    assert np.isnan(geom.apex_radius).all()

    data2 = [
        {"x_m": 0.0, "y_m": 0.0, "section_type": "straight", "radius_m": 0.0, "width_m": 8.0},
        {
            "x_m": 10.0,
            "y_m": 0.0,
            "section_type": "corner",
            "radius_m": 10.0,
            "width_m": 8.0,
            "entry_length_m": 2.0,
            "exit_length_m": 3.0,
        },
        {"x_m": 10.0, "y_m": 10.0, "section_type": "straight", "radius_m": 0.0, "width_m": 8.0},
    ]
    df2 = pd.DataFrame(data2)
    track_file2 = tmp_path / "entry_exit.csv"
    df2.to_csv(track_file2, index=False)
    geom2 = load_track_layout(track_file2, ds=1.0, closed=False)

    mask2 = np.isfinite(geom2.apex_fraction)
    assert mask2.any()
    assert np.allclose(geom2.apex_fraction[mask2], 0.5)
    assert np.allclose(geom2.entry_length[mask2], 2.0)
    assert np.allclose(geom2.exit_length[mask2], 3.0)
    assert geom2.apex_radius is not None
    assert np.isnan(geom2.apex_radius).all()

    data3 = [
        {"x_m": 0.0, "y_m": 0.0, "section_type": "straight", "radius_m": 0.0, "apex_radius_m": np.nan, "width_m": 8.0},
        {
            "x_m": 10.0,
            "y_m": 0.0,
            "section_type": "corner",
            "radius_m": 10.0,
            "apex_radius_m": 5.0,
            "width_m": 8.0,
        },
        {"x_m": 10.0, "y_m": 10.0, "section_type": "straight", "radius_m": 0.0, "apex_radius_m": np.nan, "width_m": 8.0},
    ]
    df3 = pd.DataFrame(data3)
    track_file3 = tmp_path / "apex_radius.csv"
    df3.to_csv(track_file3, index=False)
    geom3 = load_track_layout(track_file3, ds=1.0, closed=False)
    assert geom3.apex_radius is not None
    mask3 = np.isfinite(geom3.apex_radius)
    assert mask3.any()
    assert np.allclose(geom3.apex_radius[mask3], 5.0)
