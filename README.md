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

The track layout file follows the ``track_layout.csv`` format where each row
describes the start of either a straight or constant-radius corner section. The
columns ``x_m``, ``y_m``, ``section_type`` and ``radius_m`` define the geometry
while ``width_m`` gives the track width used to compute the left and right
edges.

## License

MIT License (to be defined).
