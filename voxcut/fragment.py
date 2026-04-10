"""Audio fragment value object."""
from __future__ import annotations

from dataclasses import dataclass

from .timeutil import fmt_time


@dataclass(frozen=True)
class Fragment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def label(self, idx: int) -> str:
        return (
            f"{idx + 1:>2}. {fmt_time(self.start)} → {fmt_time(self.end)}  "
            f"({fmt_time(self.duration)})"
        )
