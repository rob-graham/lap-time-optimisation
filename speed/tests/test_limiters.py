import csv
import math
import pathlib
import sys
import numpy as np

# Allow importing from the speed package when tests are executed directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from speed_profile import BikeParams, TrackPoint, compute_speed_profile


def load_bike_params(path: pathlib.Path) -> BikeParams:
    bp = BikeParams()
    gear_rows = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key, value = row[0], row[1]
            if key.startswith("gear") and key[4:].isdigit():
                gear_rows[int(key[4:])] = float(value)
            elif key in {"use_lean_angle_cap", "use_steer_rate_cap"}:
                setattr(bp, key, value.strip().lower() == "true")
            elif hasattr(bp, key) and value:
                setattr(bp, key, float(value))
    if gear_rows:
        bp.gears = tuple(gear_rows[i] for i in sorted(gear_rows))
    return bp


def test_phi_max_deg_limits_speed_in_corner():
    params_path = pathlib.Path(__file__).resolve().parent.parent / "bike_params_r6.csv"
    bp = load_bike_params(params_path)
    bp.mu = 100.0
    bp.a_wheelie_max = 50.0
    bp.a_brake = 50.0
    bp.phi_max_deg = 30.0
    bp.use_lean_angle_cap = True
    bp.use_steer_rate_cap = False

    R = 50.0
    n = 20
    pts = [
        TrackPoint(R * math.cos(2 * math.pi * i / n),
                   R * math.sin(2 * math.pi * i / n),
                   "corner", 0.0, 0.0, R, 0.0)
        for i in range(n)
    ]
    pts.append(pts[0])

    speeds, _, _, limits = compute_speed_profile(
        pts,
        bp,
        sweeps=0,
        curv_smoothing=0,
        speed_smoothing=0,
        phi_max_deg=bp.phi_max_deg,
        kappa_dot_max=None,
        use_lean_angle_cap=True,
        use_steer_rate_cap=False,
    )
    expected = math.sqrt(bp.g * math.tan(math.radians(bp.phi_max_deg)) * R)
    assert np.allclose(speeds, expected, atol=1e-6)
    assert all(lim == "lean" for lim in limits)


def test_kappa_dot_max_limits_speed_with_curvature_rate():
    params_path = pathlib.Path(__file__).resolve().parent.parent / "bike_params_r6.csv"
    bp = load_bike_params(params_path)
    bp.mu = 100.0
    bp.a_wheelie_max = 50.0
    bp.a_brake = 50.0
    bp.use_lean_angle_cap = False
    bp.use_steer_rate_cap = True
    bp.kappa_dot_max = 0.5

    pts = [
        TrackPoint(0.0, 0.0, "straight", 0.0, 0.0, 0.0, 0.0),
        TrackPoint(1.0, 0.0, "corner", 0.0, 0.0, 1000.0, 0.0),
        TrackPoint(2.0, 0.0, "corner", 0.0, 0.0, 10.0, 0.0),
        TrackPoint(3.0, 0.0, "corner", 0.0, 0.0, 1000.0, 0.0),
        TrackPoint(4.0, 0.0, "straight", 0.0, 0.0, 0.0, 0.0),
    ]
    pts.append(TrackPoint(0.0, 0.0, "straight", 0.0, 0.0, 0.0, 0.0))

    speeds, _, _, limits = compute_speed_profile(
        pts,
        bp,
        sweeps=0,
        curv_smoothing=0,
        speed_smoothing=0,
        phi_max_deg=None,
        kappa_dot_max=bp.kappa_dot_max,
        use_lean_angle_cap=False,
        use_steer_rate_cap=True,
    )

    x = [p.x for p in pts]
    y = [p.y for p in pts]
    ds = [math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(len(pts) - 1)]
    s_vals = [0.0]
    for d in ds:
        s_vals.append(s_vals[-1] + d)
    s = np.array(s_vals)

    R = [math.inf] * len(pts)
    for i, p in enumerate(pts):
        if p.section == "corner" and p.radius_m != 0.0:
            R[i] = abs(p.radius_m)
    kappa = np.zeros(len(pts))
    for i, r in enumerate(R):
        if math.isfinite(r):
            kappa[i] = 1.0 / max(r, 1e-9)

    dkappa_ds = np.gradient(kappa, s, edge_order=2)
    window = 5 if len(pts) >= 5 else 3 if len(pts) >= 3 else 1
    if window > 1:
        kernel = np.ones(window) / window
        dkappa_ds = np.convolve(dkappa_ds, kernel, mode="same")
    expected = bp.kappa_dot_max / np.maximum(np.abs(dkappa_ds), 1e-6)

    steer_indices = [i for i, lim in enumerate(limits) if lim == "steer"]
    assert steer_indices
    assert np.allclose([speeds[i] for i in steer_indices], expected[steer_indices], atol=1e-6)
