# Speed Profile

This project computes a motorcycle speed profile around a race track.

The core script, `speed_profile.py`, loads a CSV file containing `x_m`, `y_m`,
`section_type`, optional `radius_m`, `camber_rad` and `grade_rad` columns that
describe a racing line. The `radius_m` column allows corners to be specified by
their signed radius (positive for left turns, negative for right). The path is
resampled to a fixed spacing and a physics model including track camber and
gradient estimates the maximum speed, gear and engine RPM at each point. The
result is written to a new CSV file along with the cumulative distance
travelled.

Consecutive corner points automatically join tangentially with the preceding
straight; track files do not need extra intermediate points to ensure smooth
arcs at the start of a corner sequence.

## Usage

```
python speed_profile.py input.csv output.csv [options]
```

### Common options

* `--step` – resampling distance in metres (default: 2)
* `--params-file` – load motorcycle and environment parameters from CSV
* `--traction-circle` – limit acceleration by lateral grip
* `--trail-braking` – apply traction limit while braking
* `--sweeps` – number of forward/backward passes for the solver
* `--curv-smooth` – neighbour-averaging passes for corner radius (helps on noisy tracks)
* `--speed-smooth` – neighbour-averaging passes for final speed profile
* Command line options override any values loaded from the parameter CSV.

The parameter CSV may include a `gears` row containing a comma-separated list
of gear ratios (e.g. `gears,"2.583,2.000,1.667,1.444,1.286,1.150"`).
Alternatively, individual `gear1`…`gear6` entries can be supplied.

The output CSV contains `x`/`y` coordinates, cumulative distance, speed in both
m/s and km/h, selected gear, engine RPM, curvature, section type, camber, grade
and the limiting factor for that point.

For very small tracks or when using a tight `--step`, curvature calculations can
pick up noise that leads to a jerky speed trace. Increasing `--curv-smooth` or
`--speed-smooth` applies additional neighbour averaging to the corner radius or
speed profile to mitigate this.

## Development notes

The physics model and solver are contained in `compute_speed_profile` and the
`BikeParams` dataclass. To experiment with different motorcycles or track
conditions, adjust the fields in `BikeParams` or expose additional command-line
arguments in `main`.

Tests are not provided; when modifying the code, run `python speed_profile.py --help`
to ensure the script loads correctly.