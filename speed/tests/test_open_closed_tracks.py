import pathlib
import sys
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from speed_profile import load_csv, resample, compute_speed_profile, BikeParams


def test_open_track_start_end_speed_differ():
    csv_path = pathlib.Path(__file__).resolve().parent.parent / "oneCornerTrack.csv"
    pts = load_csv(csv_path)
    pts = resample(pts, step=5.0, closed=False)
    speeds, _, _, _ = compute_speed_profile(pts, BikeParams(), closed_loop=False)
    assert not math.isclose(speeds[0], speeds[-1], rel_tol=1e-3)


def test_closed_track_start_end_speed_match():
    csv_path = pathlib.Path(__file__).resolve().parent.parent / "track_layout.csv"
    pts = load_csv(csv_path)
    pts = resample(pts, step=5.0, closed=True)
    speeds, _, _, _ = compute_speed_profile(pts, BikeParams(), closed_loop=True)
    assert speeds[0] == speeds[-1]

