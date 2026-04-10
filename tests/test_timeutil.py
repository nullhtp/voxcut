"""Tests for timeutil module."""
from voxcut.timeutil import fmt_time


def test_zero():
    assert fmt_time(0.0) == "00:00.000"


def test_seconds():
    assert fmt_time(5.123) == "00:05.123"


def test_minutes():
    assert fmt_time(65.5) == "01:05.500"


def test_hours():
    assert fmt_time(3725.123) == "1:02:05.123"


def test_negative_clamps():
    assert fmt_time(-1.0) == "00:00.000"


def test_nan_clamps():
    assert fmt_time(float("nan")) == "00:00.000"


def test_none_clamps():
    assert fmt_time(None) == "00:00.000"
