import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track
from path_optim import optimise_lateral_offset


def test_optimisation_respects_bounds():
    x, y, heading, curvature, left, right = load_track(
        "data/track_layout.csv", ds=10.0
    )
    s = np.arange(len(x)) * 10.0
    s_control = np.linspace(s[0], s[-1], 8)

    offset = optimise_lateral_offset(s, curvature, left, right, s_control, buffer=0.5)

    e_vals = offset(s)
    half_width = 0.5 * np.linalg.norm(left - right, axis=1) - 0.5

    assert np.all(e_vals <= half_width + 1e-6)
    assert np.all(e_vals >= -half_width - 1e-6)
