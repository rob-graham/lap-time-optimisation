import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

# Ensure the repository root is on the path so ``src`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.method_b.solver import SolverResult
from src.method_b import post


def _dummy_result() -> SolverResult:
    grid = np.array([0.0, 1.0])
    # States: e, psi, v, kappa
    x = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    # Controls: u_kappa, a_x
    u = np.zeros((2, 2))
    return SolverResult(x=x, u=u, grid=grid, stats={})


def test_save_outputs_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _dummy_result()
    out_dir = post.save_outputs(result, timestamp="123")
    expected = Path("outputs") / "123" / "method_b"
    assert out_dir == expected
    assert (out_dir / "solution.csv").is_file()
    assert (out_dir / "speed_profile.png").is_file()
    assert (out_dir / "acceleration_profile.png").is_file()
