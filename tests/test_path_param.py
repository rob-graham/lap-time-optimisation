import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from path_param import LateralOffsetSpline, path_curvature


def test_constant_offset_curvature():
    s = np.linspace(0.0, 10.0, 50)
    kappa_c = np.full_like(s, 0.1)
    offset = LateralOffsetSpline([0.0, 5.0, 10.0], [1.0, 1.0, 1.0])
    kappa = path_curvature(s, offset, kappa_c)
    expected = 0.1 / (1 - 0.1 * 1.0)
    assert np.allclose(kappa, expected)


def test_curvature_matches_straight_centerline():
    s = np.linspace(0.0, 2 * np.pi, 200)
    kappa_c = np.zeros_like(s)
    e_vals = 0.5 * np.sin(s)
    offset = LateralOffsetSpline(s, e_vals, bc_type="periodic")
    kappa = path_curvature(s, offset, kappa_c)

    e_prime = 0.5 * np.cos(s)
    e_second = -0.5 * np.sin(s)
    expected = e_second / (1 + e_prime**2) ** 1.5
    assert np.allclose(kappa, expected, atol=1e-3)
