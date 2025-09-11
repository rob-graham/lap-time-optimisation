import sys
from pathlib import Path

import sys
from pathlib import Path

import numpy as np


# Ensure the repository root is on the path so ``src`` is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.method_b.ocp import OCP
from src.method_b import solver


def _write_warm_start_csv(path: Path, n: int) -> None:
    path.write_text(
        "e,kappa,v,psi\n" + "\n".join(
            f"{0.1*i},{0.01*i},{1+0.1*i},{0.2*i}" for i in range(n)
        )
    )


def test_load_method_a_results_from_file(tmp_path):
    grid = np.linspace(0.0, 1.0, 5)
    kappa = np.zeros_like(grid)
    width = np.full_like(grid, 5.0)
    ocp_def = OCP(kappa_c=kappa, track_half_width=width)
    csv_file = tmp_path / "warm.csv"
    _write_warm_start_csv(csv_file, grid.size)

    warm = solver._load_method_a_results(ocp_def, grid, csv_file)
    assert "x0" in warm
    assert warm["x0"].shape[0] == (ocp_def.n_x + ocp_def.n_u) * grid.size


def test_run_passes_warm_start_path(monkeypatch, tmp_path):
    from run_method_b import run

    csv_file = tmp_path / "ws.csv"
    csv_file.touch()

    called = {}

    def fake_solve(*args, **kwargs):
        called.update(kwargs)

        class Dummy:
            pass

        return Dummy()

    def fake_save(_result):
        return Path("out")

    monkeypatch.setattr(solver, "solve", fake_solve)
    monkeypatch.setattr("src.method_b.post.save_outputs", fake_save)

    track_csv = str(Path("data/track_layout.csv"))
    bike_csv = str(Path("data/bike_params_r6.csv"))
    run(track_csv, bike_csv, 1.0, warm_start=str(csv_file))

    assert called["warm_start_from_method_a"] is True
    assert Path(called["warm_start_file"]) == csv_file
