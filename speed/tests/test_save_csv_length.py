import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import TrackPoint, save_csv


def _make_points(n: int):
    return [TrackPoint(0.0, 0.0, "straight", 0.0, 0.0) for _ in range(n)]


def test_save_csv_length_mismatch(tmp_path):
    pts = _make_points(2)
    dists = [0.0, 1.0]
    speeds = [10.0, 20.0]
    gears = [1, 2]
    rpms = [1000.0, 2000.0]
    curvatures = [0.0, 0.0]
    limiters = ["limit"]
    with pytest.raises(AssertionError):
        save_csv(tmp_path / "out.csv", pts, dists, speeds, gears, rpms, curvatures, limiters)