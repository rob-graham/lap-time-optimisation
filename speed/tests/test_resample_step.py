import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import TrackPoint, resample


def _make_points():
    return [
        TrackPoint(0.0, 0.0, "straight", 0.0, 0.0),
        TrackPoint(1.0, 0.0, "straight", 0.0, 0.0),
    ]


@pytest.mark.parametrize("bad_step", [0.0, -1.0])
def test_resample_invalid_step(bad_step):
    pts = _make_points()
    with pytest.raises(ValueError, match="step must be positive"):
        resample(pts, step=bad_step)
