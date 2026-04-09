"""Time formatting helpers."""
from __future__ import annotations

import math


def fmt_time(t: float) -> str:
    """Format seconds as ``M:SS.mmm`` (or ``H:MM:SS.mmm`` past one hour)."""
    if t is None or math.isnan(t) or t < 0:
        t = 0.0
    hours, rem = divmod(t, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{int(hours):d}:{int(minutes):02d}:{seconds:06.3f}"
    return f"{int(minutes):02d}:{seconds:06.3f}"
