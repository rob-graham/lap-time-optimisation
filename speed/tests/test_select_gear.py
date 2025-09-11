import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import BikeParams, select_gear


def _make_bp():
    return BikeParams(gears=(1.150, 1.286, 1.444, 1.667, 2.0, 2.583))


def test_select_gear_low_speed():
    bp = _make_bp()
    assert select_gear(10.0, bp) == 2.583


def test_select_gear_mid_speed():
    bp = _make_bp()
    assert select_gear(55.0, bp) == 1.444


def test_select_gear_high_speed():
    bp = _make_bp()
    assert select_gear(70.0, bp) == 1.15
