import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import BikeParams, traction_circle_cap


def test_traction_circle_cap_handles_negative_radius():
    bp = BikeParams()
    v = 10.0
    acc_pos = traction_circle_cap(v, 20.0, bp, 0.0, 0.0)
    acc_neg = traction_circle_cap(v, -20.0, bp, 0.0, 0.0)
    assert acc_pos > 0
    assert acc_neg > 0
    assert acc_pos == pytest.approx(acc_neg)