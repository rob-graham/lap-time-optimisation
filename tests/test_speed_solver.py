import sys
from pathlib import Path

import numpy as np
import pytest

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from speed_solver import solve_speed_profile
from geometry import load_track_layout
from io_utils import read_bike_params_csv


def test_invalid_max_iterations_raises() -> None:
    s = np.array([0.0, 1.0])
    kappa = np.zeros_like(s)
    with pytest.raises(ValueError):
        solve_speed_profile(
            s,
            kappa,
            mu=1.0,
            a_wheelie_max=9.81,
            a_brake=11.772,
            max_iterations=0,
        )


def test_invalid_tol_raises() -> None:
    s = np.array([0.0, 1.0])
    kappa = np.zeros_like(s)
    with pytest.raises(ValueError):
        solve_speed_profile(
            s,
            kappa,
            mu=1.0,
            a_wheelie_max=9.81,
            a_brake=11.772,
            tol=0.0,
        )


def test_straight_line_profile() -> None:
    s = np.linspace(0.0, 100.0, 11)
    kappa = np.zeros_like(s)
    v, ax, ay, limit, lap_time, iterations, elapsed_s = solve_speed_profile(
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
    assert iterations >= 1
    assert elapsed_s >= 0.0


def test_straight_line_low_initial_speed_accelerates() -> None:
    s = np.linspace(0.0, 100.0, 11)
    kappa = np.zeros_like(s)
    v_init = np.zeros_like(s)
    v_init_orig = v_init.copy()
    v, ax, ay, limit, lap_time, _, _ = solve_speed_profile(
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
    v, ax, ay, limit, lap_time, _, _ = solve_speed_profile(
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
    v, ax, ay, limit, lap_time, _, _ = solve_speed_profile(
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
    v, ax, ay, limit, lap_time, _, _ = solve_speed_profile(
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


def test_lean_angle_cap_limits_speed() -> None:
    s = np.linspace(0.0, 100.0, 101)
    kappa = np.full_like(s, 0.1)
    v, ax, ay, limit, _, _, _ = solve_speed_profile(
        s,
        kappa,
        mu=10.0,
        a_wheelie_max=50.0,
        a_brake=50.0,
        v_start=0.0,
        v_end=0.0,
        phi_max_deg=30.0,
        use_steer_rate_cap=False,
    )
    mid = len(s) // 2
    expected = np.sqrt(9.81 * np.tan(np.deg2rad(30.0)) / 0.1)
    assert np.isclose(v[mid], expected, atol=0.5)
    assert limit[mid] == "lean"


def test_steer_rate_cap_limits_speed() -> None:
    s = np.linspace(0.0, 10.0, 101)
    kappa = np.linspace(0.0, 10.0, 101)
    v, ax, ay, limit, _, _, _ = solve_speed_profile(
        s,
        kappa,
        mu=100.0,
        a_wheelie_max=50.0,
        a_brake=50.0,
        v_start=0.0,
        v_end=0.0,
        kappa_dot_max=0.5,
        use_lean_angle_cap=False,
    )
    mid = len(s) // 2
    expected = 0.5 / 1.0
    assert np.isclose(v[mid], expected, atol=0.1)
    assert limit[mid] == "steer"


def test_lean_angle_small_phi_matches_theory() -> None:
    """Lean angle cap enforces theoretical speed for small angles."""
    s = np.linspace(0.0, 50.0, 101)
    kappa = np.full_like(s, 0.05)
    phi_max_deg = 10.0
    v, ax, ay, limit, _, _, _ = solve_speed_profile(
        s,
        kappa,
        mu=100.0,
        a_wheelie_max=50.0,
        a_brake=50.0,
        v_start=0.0,
        v_end=0.0,
        phi_max_deg=phi_max_deg,
        use_steer_rate_cap=False,
    )
    expected = np.sqrt(9.81 * np.tan(np.deg2rad(phi_max_deg)) / np.abs(kappa[0]))
    mask = limit == "lean"
    assert mask.any()
    assert np.allclose(v[mask], expected, atol=1e-6)
    assert np.all(limit[mask] == "lean")


def test_steer_rate_ramp_matches_theory() -> None:
    """Steer rate cap limits speed based on curvature rate."""
    s = np.linspace(0.0, 1.0, 101)
    kappa = np.linspace(0.0, 10.0, 101)
    kappa_dot_max = 5.0
    v, ax, ay, limit, _, _, _ = solve_speed_profile(
        s,
        kappa,
        mu=100.0,
        a_wheelie_max=50.0,
        a_brake=50.0,
        v_start=0.0,
        v_end=0.0,
        kappa_dot_max=kappa_dot_max,
        use_lean_angle_cap=False,
    )
    dkappa_ds = np.gradient(kappa, s, edge_order=2)
    window = 5 if kappa.size >= 5 else 3 if kappa.size >= 3 else 1
    if window > 1:
        dkappa_ds = np.convolve(dkappa_ds, np.ones(window) / window, mode="same")
    expected = kappa_dot_max / np.maximum(np.abs(dkappa_ds), 1e-6)
    mask = limit == "steer"
    assert mask.any()
    assert np.allclose(v[mask], expected[mask], atol=1e-6)
    assert np.all(limit[mask] == "steer")
