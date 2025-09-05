from __future__ import annotations

"""Utility functions for reading and writing CSV data.

This module centralises basic I/O helpers used by the demo script.  It
provides convenience wrappers around :mod:`pandas` for loading track
layouts and saving results as well as a light-weight parser for the bike
parameter CSV files used by :class:`~src.vehicle.Vehicle` and the speed
solver.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping

import csv
import pandas as pd


def read_track_csv(path: str | Path) -> pd.DataFrame:
    """Read a track layout CSV into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    path:
        Location of the CSV file describing the track.
    """
    return pd.read_csv(path)


def read_bike_params_csv(path: str | Path) -> Dict[str, float | bool]:
    """Read motorcycle parameters from ``path``.

    Values of ``true``/``false`` are interpreted as booleans while other entries
    are parsed as floating point numbers.  A section starting with a row whose
    first entry is ``rpm`` (used for torque curves) is ignored as it is not
    required for the speed solver.
    """
    params: Dict[str, float | bool] = {}
    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            key = row[0].strip()
            if key.lower() == "rpm":
                break
            try:
                raw_value = row[1].strip()
            except IndexError:
                continue

            value_lower = raw_value.lower()
            if value_lower == "true":
                params[key] = True
            elif value_lower == "false":
                params[key] = False
            else:
                try:
                    params[key] = float(raw_value)
                except ValueError:
                    continue
    return params


def write_csv(data: Mapping[str, Iterable] | pd.DataFrame, file_path: str | Path) -> None:
    """Write ``data`` to ``file_path`` ensuring parent directories exist."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False)
    else:
        pd.DataFrame(data).to_csv(file_path, index=False)
