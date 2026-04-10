"""Per-file sidecar JSON for fragment persistence."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .fragment import Fragment

_SUFFIX = ".sam3.json"
_VERSION = 1


def sidecar_path(audio: Path) -> Path:
    return audio.with_suffix(audio.suffix + _SUFFIX)


_MAX_HISTORY = 10


@dataclass
class Session:
    fragments: list[Fragment] = field(default_factory=list)
    last_description: str = ""
    description_history: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.fragments
            and not self.last_description
            and not self.description_history
        )

    def add_description(self, desc: str) -> None:
        """Push *desc* to the front of history, deduplicating."""
        desc = desc.strip()
        if not desc:
            return
        if desc in self.description_history:
            self.description_history.remove(desc)
        self.description_history.insert(0, desc)
        self.description_history = self.description_history[:_MAX_HISTORY]
        self.last_description = desc

    def to_dict(self) -> dict:
        return {
            "version": _VERSION,
            "fragments": [
                {"start": f.start, "end": f.end} for f in self.fragments
            ],
            "last_description": self.last_description,
            "description_history": self.description_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        frags_raw = data.get("fragments", [])
        frags = [Fragment(float(f["start"]), float(f["end"])) for f in frags_raw]
        return cls(
            fragments=frags,
            last_description=str(data.get("last_description", "")),
            description_history=list(data.get("description_history", [])),
        )


def load(audio: Path) -> Session:
    path = sidecar_path(audio)
    if not path.exists():
        return Session()
    try:
        return Session.from_dict(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return Session()


def save(audio: Path, session: Session) -> None:
    path = sidecar_path(audio)
    if session.is_empty():
        path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps(session.to_dict(), indent=2))
