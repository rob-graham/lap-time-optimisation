import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speed_profile import load_csv


def test_missing_section_type_defaults_to_corner(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("x_m,y_m\n0,0\n1,1\n")
    pts = load_csv(str(p))
    assert [pt.section for pt in pts] == ["corner", "corner"]


def test_blank_section_type_defaults_to_corner(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("x_m,y_m,section_type\n0,0,straight\n1,1,\n")
    pts = load_csv(str(p))
    assert [pt.section for pt in pts] == ["straight", "corner"]
