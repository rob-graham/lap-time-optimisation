import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from speed_solver import solve_speed_profile
from geometry import load_track_layout
from io_utils import read_bike_params_csv


def test_straight_line_profile() -> None:
    s = np.linspace(0.0, 100.0, 11)
    kappa = np.zeros_like(s)
    v, ax, ay, limit, lap_time = solve_speed_profile(
        s,
        kappa,
        mu=1.2,
        a_wheelie_max=9.81,
        a_brake=11.772,
        v_start=0.0,
        v_end=0.0,
    )
    assert v.shape == s.shape
    assert limit.shape == s.shape
    assert np.allclose(ay, 0.0)
    mid = len(s) // 2
    expected_vmax = np.sqrt(9.81 * 100.0)
    assert np.isclose(v[mid], expected_vmax, atol=0.5)
    assert np.isclose(v[0], 0.0, atol=1e-6)
    assert np.isclose(v[-1], 0.0, atol=1e-6)
    assert np.isclose(ax[1], 9.81, rtol=1e-2)
    assert np.isclose(ax[-2], -11.772, rtol=1e-2)
    # verify limiter reasons across the straight segment
    assert limit[0] == "accel"
    assert limit[mid] == "wheelie"
    assert limit[-2] == "stoppie"
    assert limit[-1] == "braking"


def test_straight_line_low_initial_speed_accelerates() -> None:
    s = np.linspace(0.0, 100.0, 11)
    kappa = np.zeros_like(s)
    v_init = np.zeros_like(s)
    v_init_orig = v_init.copy()
    v, ax, ay, limit, lap_time = solve_speed_profile(
        s,
        kappa,
        mu=1.2,
        a_wheelie_max=9.81,
        a_brake=11.772,
        v_init=v_init,
        v_start=0.0,
        v_end=0.0,
    )
    mid = len(s) // 2
    expected_vmax = np.sqrt(9.81 * 100.0)
    assert np.isclose(v[mid], expected_vmax, atol=0.5)
    # Ensure the solver accelerates from the low initial guess
    assert v[mid] > v_init_orig[mid]

def test_circular_track_speed_limit() -> None:
    R = 50.0
    s = np.linspace(0.0, 1000.0, 5001)
    kappa = np.full_like(s, 1.0 / R)
    # Include straight segments at start and end so the solver can reach the
    # steady-state speed before entering the circular section.
    kappa[:200] = 0.0
    kappa[-200:] = 0.0
    mu = 1.2
    v, ax, ay, limit, lap_time = solve_speed_profile(
        s,
        kappa,
        mu=mu,
        a_wheelie_max=9.81,
        a_brake=11.772,
    )
    expected = np.sqrt(mu * 9.81 * R)
    mid = len(s) // 2
    assert np.isclose(v[mid], expected, atol=0.5)
    assert limit[mid] == "corner"


def test_closed_circular_track_convergence() -> None:
    base_path = Path(__file__).resolve().parents[1]
    geom = load_track_layout(str(base_path / "data" / "track_layout.csv"), ds=1.0)
    params = read_bike_params_csv(base_path / "data" / "bike_params_r6.csv")
    s = np.arange(geom.x.size)
    v, ax, ay, limit, lap_time = solve_speed_profile(
        s,
        geom.curvature,
        mu=params["mu"],
        a_wheelie_max=params["a_wheelie_max"],
        a_brake=params["a_brake"],
        closed_loop=True,
    )
    assert np.isclose(v[0], v[-1], atol=1e-3)
    # Speeds should vary around the lap on a closed loop.
    assert np.ptp(v) > 5.0
    straight = np.isclose(geom.curvature, 0.0)
    # Ensure the track contains straight sections and that the solver applies
    # acceleration or braking on them.
    assert straight.any()
    assert np.any(np.abs(ax[straight]) > 0.1)
    assert np.isclose(v.max(), 50.0, atol=15.0)


def test_open_track_has_free_end_speeds() -> None:
    base_path = Path(__file__).resolve().parents[1]
    geom = load_track_layout(base_path / "data" / "oneCornerTrack.csv", ds=1.0, closed=False)
    params = read_bike_params_csv(base_path / "data" / "bike_params_r6.csv")
    s = np.arange(geom.x.size)
    v, ax, ay, limit, lap_time = solve_speed_profile(
        s,
        geom.curvature,
        mu=params["mu"],
        a_wheelie_max=params["a_wheelie_max"],
        a_brake=params["a_brake"],
    )
    assert v[0] > 0.0
    assert v[-1] > 0.0
    corner_idx = np.where(np.abs(geom.curvature) > 0)[0][0]
    assert np.argmin(v) == corner_idx
    assert v[corner_idx] < v[0]
    assert v[corner_idx] < v[-1]
