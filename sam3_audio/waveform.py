"""Precomputed peak-waveform strip, rendered as Rich text."""
from __future__ import annotations

import numpy as np
from rich.text import Text

_BLOCKS = " ▁▂▃▄▅▆▇█"


def compute_peaks(mono: np.ndarray, width: int) -> np.ndarray:
    """Downsample ``mono`` to per-column peak magnitudes in [0, 1]."""
    width = max(1, width)
    n = mono.shape[0]
    if n == 0:
        return np.zeros(width, dtype=np.float32)
    edges = np.linspace(0, n, width + 1, dtype=np.int64)
    peaks = np.empty(width, dtype=np.float32)
    for i in range(width):
        a, b = edges[i], edges[i + 1]
        peaks[i] = float(np.abs(mono[a:b]).max()) if b > a else 0.0
    m = peaks.max()
    if m > 0:
        peaks /= m
    return peaks


def render(peaks: np.ndarray, cursor_frac: float) -> Text:
    width = int(peaks.shape[0])
    cursor_col = min(width - 1, max(0, int(cursor_frac * width)))
    text = Text()
    levels = len(_BLOCKS) - 1
    for i, p in enumerate(peaks):
        ch = _BLOCKS[int(round(float(p) * levels))]
        style = "reverse bold yellow" if i == cursor_col else "cyan"
        text.append(ch, style=style)
    return text
