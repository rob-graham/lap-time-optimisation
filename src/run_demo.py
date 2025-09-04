from __future__ import annotations

"""Command line demo for the lap-time optimisation pipeline.

Running ``python -m src.run_demo`` executes a simple workflow that
parses a track layout, optimises a racing line and computes a feasible
speed profile.  Results are written to time-stamped CSV files under the
``outputs`` directory.
"""

from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import pandas as pd

from .io_utils import read_track_csv, read_bike_params_csv, write_csv
from .geometry import load_track_layout
from .path_optim import optimise_lateral_offset
from .path_param import path_curvature
from .speed_solver import solve_speed_profile


def run(track_file: str, bike_file: str, ds: float, buffer: float, n_ctrl: int) -> Path:
    """Execute the optimisation pipeline and return the output directory."""
    # Load input data
    df = read_track_csv(track_file)
    bike_params = read_bike_params_csv(bike_file)

    closed = (
        np.hypot(df["x_m"].iloc[0] - df["x_m"].iloc[-1], df["y_m"].iloc[0] - df["y_m"].iloc[-1])
        <= ds
    )
    geom = load_track_layout(track_file, ds, closed=closed)
    x, y, psi, kappa_c = geom.x, geom.y, geom.heading, geom.curvature
    left_edge, right_edge = geom.left_edge, geom.right_edge
    s = np.arange(x.size) * ds

    # Path optimisation
    s_control = np.linspace(s[0], s[-1], n_ctrl)
    offset_spline = optimise_lateral_offset(
        s, kappa_c, left_edge, right_edge, s_control, buffer=buffer
    )
    offset = offset_spline(s)
    kappa_path = path_curvature(s, offset_spline, kappa_c)
    normal_x = -np.sin(psi)
    normal_y = np.cos(psi)
    x_path = x + offset * normal_x
    y_path = y + offset * normal_y

    # Speed solver
    mu = float(bike_params.get("mu", 1.0))
    a_wheelie_max = float(bike_params.get("a_wheelie_max", 9.81))
    a_brake = float(bike_params.get("a_brake", 9.81))
    v, ax, ay = solve_speed_profile(
        s, kappa_path, mu, a_wheelie_max, a_brake, closed_loop=closed
    )

    # Write outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_df = pd.DataFrame(
        {
            "s_m": s,
            "x_center_m": x,
            "y_center_m": y,
            "heading_rad": psi,
            "curvature_1pm": kappa_c,
            "x_left_m": left_edge[:, 0],
            "y_left_m": left_edge[:, 1],
            "x_right_m": right_edge[:, 0],
            "y_right_m": right_edge[:, 1],
        }
    )
    results_df = pd.DataFrame(
        {
            "s_m": s,
            "x_path_m": x_path,
            "y_path_m": y_path,
            "offset_m": offset,
            "curvature_1pm": kappa_path,
            "speed_mps": v,
            "ax_mps2": ax,
            "ay_mps2": ay,
        }
    )

    write_csv(geometry_df, out_dir / "geometry.csv")
    write_csv(results_df, out_dir / "results.csv")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run lap-time optimisation demo")
    parser.add_argument("--track", default="data/track_layout.csv", help="Track layout CSV")
    parser.add_argument("--bike", default="data/bike_params_sv650.csv", help="Bike parameter CSV")
    parser.add_argument("--ds", type=float, default=1.0, help="Track interpolation spacing")
    parser.add_argument("--buffer", type=float, default=0.5, help="Track edge buffer")
    parser.add_argument(
        "--ctrl-points", type=int, default=20, help="Number of lateral offset control points"
    )
    args = parser.parse_args(argv)

    out_dir = run(args.track, args.bike, args.ds, args.buffer, args.ctrl_points)
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
