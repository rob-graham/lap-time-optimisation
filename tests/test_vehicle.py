import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from vehicle import Vehicle


def _create_csv(tmp_path: Path) -> Path:
    content = """
rho,1.225
g,9.81
m,200
CdA,0.3
Crr,0.01
rw,0.3
shift_rpm,10000
primary,2.0
gear1,2.0
final_drive,3.0
eta_driveline,0.9
T_peak,50
rpm,Nm
0,0
5000,50
10000,0
"""
    file = tmp_path / "bike_params.csv"
    file.write_text(content.strip())
    return file


def test_drag_and_rolling(tmp_path: Path) -> None:
    csv_path = _create_csv(tmp_path)
    vehicle = Vehicle(csv_path)
    speed = 10.0
    drag = vehicle.aerodynamic_drag(speed)
    expected_drag = 0.5 * 1.225 * 0.3 * speed**2
    assert np.isclose(drag, expected_drag)
    rr = vehicle.rolling_resistance()
    expected_rr = 0.01 * 200 * 9.81
    assert np.isclose(rr, expected_rr)


def test_tractive_force(tmp_path: Path) -> None:
    csv_path = _create_csv(tmp_path)
    vehicle = Vehicle(csv_path)
    speeds, forces = vehicle.tractive_force_vs_speed(1)
    rpm = np.array([0.0, 5000.0, 10000.0])
    torque = np.array([0.0, 50.0, 0.0])
    total_ratio = 2.0 * 2.0 * 3.0
    expected_speeds = rpm * 2 * np.pi * 0.3 / (60.0 * total_ratio)
    expected_forces = torque * total_ratio * 0.9 / 0.3
    assert np.allclose(speeds, expected_speeds)
    assert np.allclose(forces, expected_forces)

def test_tractive_force_envelope_decreases() -> None:
    param_file = Path(__file__).resolve().parents[1] / "data" / "bike_params_sv650.csv"
    vehicle = Vehicle(param_file)
    shift_rpm = vehicle.params["shift_rpm"]
    ratio1 = vehicle.primary * vehicle.gear_ratios[0] * vehicle.final_drive
    v_shift = shift_rpm * 2 * np.pi * vehicle.rw / (60.0 * ratio1)

    def envelope(speed: float) -> float:
        gears = range(1, len(vehicle.gear_ratios) + 1)
        return max(vehicle.tractive_force(speed, g) for g in gears)

    before = envelope(0.99 * v_shift)
    after = envelope(1.01 * v_shift)
    assert after < before
