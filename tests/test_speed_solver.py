import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from speed_solver import solve_speed_profile


def test_straight_line_profile() -> None:
    s = np.linspace(0.0, 100.0, 11)
    kappa = np.zeros_like(s)
    v, ax, ay = solve_speed_profile(
        s,
        kappa,
        mu=1.2,
        a_wheelie_max=9.81,
        a_brake=11.772,
    )
    assert v.shape == s.shape
    assert np.allclose(ay, 0.0)
    mid = len(s) // 2
    expected_vmax = np.sqrt(9.81 * 100.0)
    assert np.isclose(v[mid], expected_vmax, atol=0.5)
    assert np.isclose(v[0], 0.0, atol=1e-6)
    assert np.isclose(v[-1], 0.0, atol=1e-6)
    assert np.isclose(ax[1], 9.81, rtol=1e-2)
    assert np.isclose(ax[-2], -11.772, rtol=1e-2)
