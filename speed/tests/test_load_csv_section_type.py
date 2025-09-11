import pathlib
import sys
import pytest

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


def test_valid_section_types(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("x_m,y_m,section_type\n0,0,straight\n1,1,corner\n")
    pts = load_csv(str(p))
    assert [pt.section for pt in pts] == ["straight", "corner"]


def test_invalid_section_type_raises(tmp_path):
    p = tmp_path / "track.csv"
    p.write_text("x_m,y_m,section_type\n0,0,straight\n1,1,foo\n")
    with pytest.raises(ValueError) as excinfo:
        load_csv(str(p))
    assert "row 3" in str(excinfo.value)
