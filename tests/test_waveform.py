"""Tests for waveform rendering."""
import numpy as np

from voxcut.waveform import compute_peaks, render


def test_compute_peaks_shape():
    mono = np.sin(np.linspace(0, 100, 48000)).astype(np.float32)
    peaks = compute_peaks(mono, 80)
    assert peaks.shape == (80,)
    assert peaks.max() <= 1.0
    assert peaks.min() >= 0.0


def test_compute_peaks_empty():
    peaks = compute_peaks(np.array([], dtype=np.float32), 40)
    assert peaks.shape == (40,)
    assert (peaks == 0).all()


def test_render_basic():
    peaks = np.ones(20, dtype=np.float32)
    text = render(peaks, cursor_frac=0.5)
    assert len(text) == 20


def test_render_with_fragments():
    peaks = np.ones(40, dtype=np.float32)
    text = render(
        peaks,
        cursor_frac=0.5,
        duration=10.0,
        fragments=[(1.0, 3.0)],
    )
    assert len(text) == 40


def test_render_with_ghost():
    peaks = np.ones(40, dtype=np.float32)
    text = render(
        peaks,
        cursor_frac=0.5,
        duration=10.0,
        in_point=2.0,
        cursor_sec=5.0,
    )
    assert len(text) == 40
