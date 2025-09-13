# Lap Time Optimisation

This project aims to explore motorcycle lap time optimisation using physics-based models. Bike and track parameters are provided as CSV files and future code in `src/` will load these inputs to simulate laps and analyse performance.

## Project Structure

```
.
├── data/              # Bike and track CSV inputs
├── outputs/           # Generated results and figures
├── src/               # Source code for simulations and CLI
└── tests/             # Test suite
```

## Setup

1. Create a Python virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## CLI Usage

Once implementation is complete, run simulations via the command line. Example:

```bash
python -m src.run_demo --track data/track_layout.csv --bike data/bike_params_r6.csv
```

Replace `track_layout.csv` and `bike_params_r6.csv` with the desired files. By
default, the command prints the simulated lap time; pass `--quiet-lap-time` to
silence this summary output.

The path optimisation and speed profile solver accuracy can be traded for
faster execution using `--max-iter` and `--path-tol` for the former and
`--speed-max-iter` and `--speed-tol` for the latter. Smaller iteration counts or
looser convergence tolerances reduce run time but may slightly increase lap
time estimates. The path optimiser defaults to a convergence tolerance of
`1e-6`; relaxing this (for example to `1e-3`) speeds up execution at the cost of
path accuracy.

By default the path optimiser uses a finite-difference step size of `1e-2` when
approximating gradients. This value can be adjusted through the `fd_step`
argument of `optimise_lateral_offset` if finer control over the optimisation is
needed.

### Clothoid racing line

Passing `--clothoid` skips the numerical path optimisation and instead builds a
simple racing line from cubic spirals (clothoids). This heuristic approach is
much faster but may not produce the minimum‑time trajectory. The subsequent
speed profile optimisation still runs on the generated path. This flag only
affects the Method A pipeline in ``run_demo`` and has no effect on the Method B
optimal control solver.

```bash
python -m src.run_demo --track data/track_layout.csv --bike data/bike_params_r6.csv --clothoid
```

The clothoid line relies only on the track geometry; it does not use the path
optimisation parameters above.

### Lean-angle and steer-rate limits

The speed solver can enforce physical limits on the bike dynamics. When a
maximum lean angle (``phi_max_deg``) is specified and lean capping is enabled,
the solver reduces the speed in corners so that the required lean angle does not
exceed this limit. Similarly, providing a maximum steer rate (``kappa_dot_max``)
limits the speed through fast direction changes to ensure the steering rate
remains within bounds.

These parameters can be supplied in the bike parameter CSV via
``phi_max_deg`` and ``kappa_dot_max`` along with boolean fields
``use_lean_angle_cap`` and ``use_steer_rate_cap`` to enable or disable the caps.
They may also be overridden from the command line:

```bash
python -m src.run_demo --phi-max-deg 55 --kappa-dot-max 0.8
```

``--phi-max-deg`` and ``--kappa-dot-max`` set the respective limits directly,
while ``--no-lean-cap`` and ``--no-steer-cap`` disable enforcing the lean and
steer caps even if values are present in the bike file.

The track layout file follows the ``track_layout.csv`` format where each row
describes the start of either a straight or constant-radius corner section. The
required columns ``x_m``, ``y_m``, ``section_type``, ``radius_m`` and ``width_m``
define the basic geometry. Additional optional fields such as ``camber_rad``,
``grade_rad``, ``apex_fraction``, ``entry_length_m`` and ``exit_length_m`` can be
included and are used by some features like the clothoid racing line.

Each run writes two CSV files in a time-stamped folder under ``outputs``:

``geometry.csv``
    Discretised centreline with heading, curvature and track edges.
``results.csv``
    Optimised path with speed profile and the same track edges for
    convenience.

## Method B

Method B formulates lap-time optimisation as a single optimal control problem
solved with CasADi and IPOPT. Unlike Method A's sequential path and speed
optimisation, this approach collocates the dynamics and optimises curvature and
velocity simultaneously. It therefore introduces a dependency on CasADi,
included in ``requirements.txt``.

Example usage:

```bash
python -m run_method_b --track_csv data/sample_track.csv --bike_csv data/bike_params.csv --ds 0.5
```

Outputs are written under ``outputs/<timestamp>/method_b/`` containing the
solution CSV and diagnostic plots.

Additional solver options include ``--linear-solver`` to choose the IPOPT linear
solver, ``--ipopt-opts`` for further IPOPT configuration, ``--warm-start`` to
initialise from Method A results, and flags such as ``--use-slacks`` and
``--no-auto-slack`` controlling slack variable behaviour. Tolerance and
verbosity can be adjusted via ``--tol`` and ``--print-level``.

## License

MIT License (to be defined).
