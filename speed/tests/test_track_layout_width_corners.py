import pathlib
import sys
import numpy as np

# Add the parent directory to the path so speed_profile can be imported when
# running the tests directly from this subdirectory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from speed_profile import load_csv, resample, compute_speed_profile, BikeParams


def _load_params():
    # Helper to load bike parameters for tests.
    return BikeParams()


def test_load_csv_ignores_width_and_handles_single_point_corners():
    csv_path = pathlib.Path(__file__).resolve().parent.parent / "track_layout.csv"
    pts = load_csv(csv_path)

    # width_m column should be parsed but not affect geometry
    assert {pt.width_m for pt in pts} == {8.0}

    bp = _load_params()
    res = resample(pts, step=5.0)
    _, _, curv, _ = compute_speed_profile(res, bp)

    # Changing width should have no effect on curvature or speeds
    for pt in pts:
        pt.width_m = 1.0
    res2 = resample(pts, step=5.0)
    _, _, curv2, _ = compute_speed_profile(res2, bp)
    assert np.allclose(curv, curv2)

    # Ensure each corner is expanded into multiple resampled points and the
    # sequence of radii matches the input specification
    orig_radii = [pt.radius_m for pt in load_csv(csv_path) if pt.section == "corner"]

    res_radii = []
    current = None
    count = 0
    counts = []
    for p in res:
        if p.section == "corner":
            if current is None:
                current = p.radius_m
            elif p.radius_m != current:
                res_radii.append(current)
                current = p.radius_m
            count += 1
        else:
            if current is not None:
                res_radii.append(current)
                counts.append(count)
                current = None
                count = 0
    if current is not None:
        res_radii.append(current)
        counts.append(count)

    assert res_radii == orig_radii
    assert counts and all(c > 1 for c in counts)
