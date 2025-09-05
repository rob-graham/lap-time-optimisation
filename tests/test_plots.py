import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

# Add the ``src`` directory to the import path for test execution.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from plots import plot_plan_view


def _track_data():
    x_center = np.array([0.0, 1.0])
    y_center = np.array([0.0, 0.0])
    left_edge = np.stack((x_center, y_center - 1.0), axis=1)
    right_edge = np.stack((x_center, y_center + 1.0), axis=1)
    return x_center, y_center, left_edge, right_edge


def test_plot_plan_view_requires_edges():
    x_center, y_center, left_edge, right_edge = _track_data()
    with pytest.raises(ValueError):
        plot_plan_view(x_center, y_center)
    with pytest.raises(ValueError):
        plot_plan_view(x_center, y_center, left_edge, None)
    with pytest.raises(ValueError):
        plot_plan_view(x_center, y_center, None, right_edge)


def test_plot_plan_view_rejects_aliases():
    x_center, y_center, left_edge, right_edge = _track_data()
    with pytest.raises(TypeError):
        plot_plan_view(x_center, y_center, inner_edge=left_edge, outer_edge=right_edge)


def test_plot_plan_view_returns_axes():
    x_center, y_center, left_edge, right_edge = _track_data()
    ax = plot_plan_view(x_center, y_center, left_edge, right_edge)
    assert ax.get_xlabel() == "x [m]"
    plt.close(ax.figure)
