import sys
from pathlib import Path

import numpy as np

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geometry import load_track_layout
from path_optim import optimise_lateral_offset


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
