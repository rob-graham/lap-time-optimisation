import pathlib
import sys
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import load_csv, resample, compute_speed_profile, BikeParams


def _corner_apices(points, curvatures):
    indices = []
    start = None
    for i, pt in enumerate(points):
        if pt.section == "corner" and start is None:
            start = i
        elif pt.section != "corner" and start is not None:
            end = i
            apex = max(range(start, end), key=lambda k: curvatures[k])
            indices.append(apex)
            start = None
    if start is not None:
        apex = max(range(start, len(points)), key=lambda k: curvatures[k])
        indices.append(apex)
    return indices


def _map_indices(src_pts, dst_pts, src_indices):
    mapped = []
    for idx in src_indices:
        x, y = src_pts[idx].x, src_pts[idx].y
        j = min(
            range(len(dst_pts)),
            key=lambda k: (dst_pts[k].x - x) ** 2 + (dst_pts[k].y - y) ** 2,
        )
        mapped.append(j)
    return mapped


def test_step_curvature_consistency():
    csv_path = pathlib.Path(__file__).resolve().parent.parent / "sample_track.csv"
    pts = load_csv(csv_path)
    bp = BikeParams()

    res1 = resample(pts, step=1.0)
    speeds1, _, curv1, _ = compute_speed_profile(res1, bp)

    res8 = resample(pts, step=8.0)
    speeds8, _, curv8, _ = compute_speed_profile(res8, bp)

    apex8 = _corner_apices(res8, curv8)
    apex1 = _map_indices(res8, res1, apex8)

    for i8, i1 in zip(apex8, apex1):
        r8 = 1.0 / max(curv8[i8], 1e-9)
        r1 = 1.0 / max(curv1[i1], 1e-9)
        assert r1 == pytest.approx(r8, abs=1.0)

        s8 = speeds8[i8]
        s1 = speeds1[i1]
        assert abs(s1 - s8) < 4.0