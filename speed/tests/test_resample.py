import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import load_csv, resample

def _angle_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)

def test_first_corner_tangent():
    pts = load_csv("sample_track.csv")
    csv_path = pathlib.Path(__file__).resolve().parent.parent / "sample_track.csv"
    pts = load_csv(csv_path)
    res = resample(pts, step=5.0)
    idx = next(
        i
        for i in range(1, len(res))
        if res[i - 1].section == "straight" and res[i].section == "corner"
    )
    heading_in = math.atan2(
        res[idx - 1].y - res[idx - 2].y, res[idx - 1].x - res[idx - 2].x
    )
    heading_out = math.atan2(
        res[idx].y - res[idx - 1].y, res[idx].x - res[idx - 1].x
    )
    expected = 5.0 / (2 * abs(res[idx].radius_m))
    assert abs(_angle_diff(heading_in, heading_out) - expected) < 1e-2