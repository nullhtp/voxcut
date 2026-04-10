"""Precomputed peak-waveform strip, rendered as Rich text."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from rich.text import Text

_BLOCKS = " ▁▂▃▄▅▆▇█"

# Style priority (highest wins):  cursor > in-point > fragment > default
_STYLE_DEFAULT = "cyan"
_STYLE_FRAGMENT = "green"
_STYLE_IN_POINT = "bold yellow"
_STYLE_GHOST = "dim yellow"
_STYLE_CURSOR = "reverse bold yellow"


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


def _col(frac: float, width: int) -> int:
    return min(width - 1, max(0, int(frac * width)))


def render(
    peaks: np.ndarray,
    cursor_frac: float,
    duration: float = 0.0,
    fragments: Sequence[tuple[float, float]] = (),
    in_point: Optional[float] = None,
    cursor_sec: float = 0.0,
) -> Text:
    """Render the waveform strip with overlays.

    ``fragments`` is a sequence of ``(start_sec, end_sec)`` pairs.
    ``in_point`` is the pending mark-in position in seconds.
    ``cursor_sec`` is the absolute cursor position — used together with
    *in_point* to render a ghost selection region.
    """
    width = int(peaks.shape[0])
    if width == 0:
        return Text()

    # Pre-compute per-column style
    styles = [_STYLE_DEFAULT] * width
    levels = len(_BLOCKS) - 1

    if duration > 0:
        for start, end in fragments:
            a = _col(start / duration, width)
            b = _col(end / duration, width)
            for j in range(a, b + 1):
                styles[j] = _STYLE_FRAGMENT
        if in_point is not None:
            # Ghost region from in-point to cursor
            a = _col(in_point / duration, width)
            b = _col(cursor_sec / duration, width)
            lo, hi = min(a, b), max(a, b)
            for j in range(lo, hi + 1):
                if styles[j] == _STYLE_DEFAULT:
                    styles[j] = _STYLE_GHOST
            styles[a] = _STYLE_IN_POINT

    cursor_col = _col(cursor_frac, width)
    styles[cursor_col] = _STYLE_CURSOR

    text = Text()
    for i, p in enumerate(peaks):
        ch = _BLOCKS[int(round(float(p) * levels))]
        text.append(ch, style=styles[i])
    return text
