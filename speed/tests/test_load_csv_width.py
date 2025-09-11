import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import load_csv


def test_width_column_parsed(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("x_m,y_m,width_m\n0,0,7.5\n1,0,8.5\n")
    pts = load_csv(str(p))
    assert [pt.width_m for pt in pts] == [7.5, 8.5]


def test_width_defaults_to_zero(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("x_m,y_m\n0,0\n1,0\n")
    pts = load_csv(str(p))
    assert [pt.width_m for pt in pts] == [0.0, 0.0]
