import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from io_utils import read_bike_params_csv


def test_read_bike_params_csv_bool_and_float() -> None:
    base_path = Path(__file__).resolve().parents[1]
    params = read_bike_params_csv(base_path / "data" / "bike_params_r6.csv")
    assert isinstance(params["mu"], float)
    assert isinstance(params["use_lean_angle_cap"], bool)
    assert params["use_lean_angle_cap"] is True
    assert params["use_steer_rate_cap"] is True
