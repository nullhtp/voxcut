"""Tests for Fragment value object."""
from voxcut.fragment import Fragment


def test_duration():
    f = Fragment(1.0, 3.5)
    assert f.duration == 2.5


def test_duration_zero():
    f = Fragment(5.0, 5.0)
    assert f.duration == 0.0


def test_label():
    f = Fragment(1.0, 3.5)
    label = f.label(0)
    assert "1." in label
    assert "00:01.000" in label
    assert "00:03.500" in label


def test_frozen():
    f = Fragment(1.0, 2.0)
    try:
        f.start = 0.0
        raise AssertionError("should raise")
    except AttributeError:
        pass
