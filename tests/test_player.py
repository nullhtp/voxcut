"""Tests for the Player class."""
import numpy as np

from sam3_audio.player import Player


def test_creation():
    data = np.zeros((1600, 1), dtype=np.float32)
    p = Player(data, 16000)
    assert p.duration == 0.1
    assert p.position == 0.0
    assert not p.playing
    p.close()


def test_seek():
    data = np.zeros((16000, 1), dtype=np.float32)
    p = Player(data, 16000)
    p.seek(0.5)
    assert abs(p.position - 0.5) < 0.001
    p.close()


def test_seek_to():
    data = np.zeros((16000, 1), dtype=np.float32)
    p = Player(data, 16000)
    p.seek_to(0.75)
    assert abs(p.position - 0.75) < 0.001
    p.close()


def test_seek_clamps():
    data = np.zeros((16000, 1), dtype=np.float32)
    p = Player(data, 16000)
    p.seek(-100)
    assert p.position == 0.0
    p.seek(1000)
    assert p.position <= p.duration
    p.close()


def test_speed():
    data = np.zeros((16000, 1), dtype=np.float32)
    p = Player(data, 16000)
    p.set_speed(1.5)
    assert p.speed == 1.5
    p.set_speed(0.1)
    assert p.speed == Player.MIN_SPEED
    p.set_speed(10.0)
    assert p.speed == Player.MAX_SPEED
    p.close()


def test_gain():
    data = np.zeros((16000, 1), dtype=np.float32)
    p = Player(data, 16000)
    p.set_gain(6.0)
    assert p.gain_db == 6.0
    assert abs(p._gain_linear - 2.0) < 0.01
    p.set_gain(-100)
    assert p.gain_db == Player.MIN_GAIN_DB
    p.close()


def test_1d_input():
    data = np.zeros(16000, dtype=np.float32)
    p = Player(data, 16000)
    assert p.channels == 1
    assert p.data.ndim == 2
    p.close()
