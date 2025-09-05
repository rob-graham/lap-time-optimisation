import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track_layout
import path_optim
from path_optim import optimise_lateral_offset
from path_param import path_curvature
from speed_solver import solve_speed_profile


def test_optimisation_respects_bounds():
    geom = load_track_layout("data/track_layout.csv", ds=10.0)
    s = np.arange(len(geom.x)) * 10.0
    s_control = np.linspace(s[0], s[-1], 8)

    offset, iterations = optimise_lateral_offset(
        s, geom.curvature, geom.left_edge, geom.right_edge, s_control, buffer=0.5
    )

    assert iterations > 0
    e_vals = offset(s)
    half_width = 0.5 * np.linalg.norm(geom.left_edge - geom.right_edge, axis=1) - 0.5

    assert np.all(e_vals <= half_width + 1e-6)
    assert np.all(e_vals >= -half_width - 1e-6)


def test_curvature_cost_performs_multiple_iterations():
    """Ensure curvature cost optimisation iterates more than once.

    With ``path_tol=1e-6`` on ``track_layout.csv`` the SLSQP solver typically
    performs around 20 iterations before converging.  The exact count may vary
    slightly with ``scipy`` versions, but anything less than 2 indicates the
    optimisation terminated prematurely and warrants investigation."""

    geom = load_track_layout("data/track_layout.csv", ds=10.0)
    s = np.arange(len(geom.x)) * 10.0
    s_control = np.linspace(s[0], s[-1], 8)

    _, iterations = optimise_lateral_offset(
        s,
        geom.curvature,
        geom.left_edge,
        geom.right_edge,
        s_control,
        buffer=0.5,
        cost="curvature",
        path_tol=1e-6,
    )

    assert iterations > 1, f"Expected more than one iteration, got {iterations}"


def test_lap_time_cost_reduces_lap_time():
    geom = load_track_layout("data/oneCornerTrack.csv", ds=10.0)
    s = np.arange(len(geom.x)) * 10.0
    s_control = np.linspace(s[0], s[-1], 6)

    mu = 1.0
    a_wheelie_max = 9.81
    a_brake = 9.81

    offset_spline, _ = optimise_lateral_offset(
        s,
        geom.curvature,
        geom.left_edge,
        geom.right_edge,
        s_control,
        buffer=0.5,
        cost="lap_time",
        mu=mu,
        a_wheelie_max=a_wheelie_max,
        a_brake=a_brake,
        speed_max_iterations=20,
        speed_tol=1e-2,
        path_tol=1e-2,
        v_start=0.0,
        v_end=0.0,
        fd_step=1e-3,
    )

    kappa_opt = path_curvature(s, offset_spline, geom.curvature)
    _, _, _, _, lap_time_opt, _, _ = solve_speed_profile(
        s, kappa_opt, mu, a_wheelie_max, a_brake, v_start=0.0, v_end=0.0
    )

    _, _, _, _, lap_time_center, _, _ = solve_speed_profile(
        s, geom.curvature, mu, a_wheelie_max, a_brake, v_start=0.0, v_end=0.0
    )

    assert lap_time_opt < lap_time_center


def test_fd_step_forwarded_to_minimize(monkeypatch):
    geom = load_track_layout("data/track_layout.csv", ds=10.0)
    s = np.arange(len(geom.x)) * 10.0
    s_control = np.linspace(s[0], s[-1], 8)

    captured: list[dict] = []

    def fake_minimize(fun, x0, method=None, constraints=None, options=None, tol=None):
        captured.append({"options": options, "tol": tol})
        class Result:
            success = True
            x = x0
            nit = 0
        return Result()

    monkeypatch.setattr(path_optim, "minimize", fake_minimize)

    # Explicitly pass None to trigger the default value forwarding
    optimise_lateral_offset(
        s,
        geom.curvature,
        geom.left_edge,
        geom.right_edge,
        s_control,
        fd_step=None,
        path_tol=1e-2,
    )
    # Default should be forwarded as 1e-2
    assert captured and captured[0]["options"].get("eps") == 1e-2
    assert captured[0]["options"].get("ftol") == 1e-2
    assert captured[0]["tol"] is None

    # Explicit value should override the default
    optimise_lateral_offset(
        s,
        geom.curvature,
        geom.left_edge,
        geom.right_edge,
        s_control,
        fd_step=1e-3,
        path_tol=1e-2,
    )
    assert captured[1]["options"].get("eps") == 1e-3
