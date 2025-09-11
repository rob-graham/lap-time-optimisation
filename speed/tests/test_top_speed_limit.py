import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from speed_profile import BikeParams, TrackPoint, compute_speed_profile, max_speed


def test_top_speed_limited_by_shift_rpm():
    bp = BikeParams()
    bp.mu = 10.0
    bp.a_wheelie_max = 50.0
    bp.a_brake = 50.0
    bp.T_peak = 500.0

    pts = [TrackPoint(i * 10.0, 0.0, "straight", 0.0, 0.0) for i in range(101)]
    speeds, _, _, _ = compute_speed_profile(
        pts,
        bp,
        sweeps=10,
        curv_smoothing=0,
        speed_smoothing=0,
        phi_max_deg=None,
        kappa_dot_max=None,
        use_traction_circle=False,
        trail_braking=False,
    )
    v_top = max_speed(bp)
    assert max(speeds) <= v_top + 1e-6
