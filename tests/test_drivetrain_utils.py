import csv
import sys
from pathlib import Path

import pytest

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from drivetrain_utils import engine_rpm, select_gear


def _load_params() -> tuple[dict[str, float], list[float]]:
    """Load SV650 parameters and ordered gear ratios."""

    params: dict[str, float] = {}
    path = Path(__file__).resolve().parents[1] / "data" / "bike_params_sv650.csv"
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                params[row[0]] = float(row[1])
            except ValueError:
                continue
    gears = [params[f"gear{i}"] for i in range(1, 7)]
    return params, gears


def test_engine_rpm_known_value() -> None:
    params, gears = _load_params()
    rpm = engine_rpm(
        30.0,
        params["primary"],
        params["final_drive"],
        gears[2],  # third gear
        params["rw"],
    )
    assert rpm == pytest.approx(8565.910961476091)


@pytest.mark.parametrize(
    "speed, expected_idx, expected_rpm",
    [
        (10.0, 0, 5089.559829348139),
        (25.0, 1, 9189.483024637575),
        (35.0, 2, 9993.562788388772),
        (45.0, 3, 10467.395506442812),
        (50.0, 4, 9940.546544603194),
        (55.0, 5, 9687.246688945941),
        # Even above ``shift_rpm`` the highest gear should be selected
        (60.0, 5, 10567.905478850122),
    ],
)
def test_select_gear(speed: float, expected_idx: int, expected_rpm: float) -> None:
    params, gears = _load_params()
    gear = select_gear(
        speed,
        gears,
        params["shift_rpm"],
        params["primary"],
        params["final_drive"],
        params["rw"],
    )
    assert gear == pytest.approx(gears[expected_idx])
    rpm = engine_rpm(
        speed,
        params["primary"],
        params["final_drive"],
        gear,
        params["rw"],
    )
    assert rpm == pytest.approx(expected_rpm)
    if speed <= 55.0:
        assert rpm <= params["shift_rpm"]
    else:
        assert rpm > params["shift_rpm"]

