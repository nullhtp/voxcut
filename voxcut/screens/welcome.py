"""First-launch welcome / quick-start guide."""
from __future__ import annotations

from pathlib import Path

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

_FLAG_DIR = Path.home() / ".config" / "voxcut"
_FLAG_FILE = _FLAG_DIR / "welcomed"

_GUIDE = """\
[b cyan]Welcome to voxcut![/b cyan]

[b]1. Listen[/b]
   space = play/pause    ← / → = seek ±5s    - / + = speed

[b]2. Mark a fragment[/b]
   Seek to the start, press [b]i[/b] (in-point) — yellow marker appears.
   Seek to the end, press [b]o[/b] (out-point) — green fragment created.

[b]3. Fine-tune[/b]
   [b]p[/b] = play fragment    [b][ ][/b] = expand    [b]{ }[/b] = contract
   [b]g[/b] / [b]G[/b] = jump to start/end    [b]S[/b] = split    [b]m[/b] = merge

[b]4. Isolate a voice[/b]
   [b]d[/b] → describe the voice (e.g. "man speaking") → Enter.
   Audition: [b]t[/b] = target  [b]r[/b] = residual  [b]o[/b] = original
   [b]k[/b] = keep    [b]l[/b] = load target back into editor

[b]5. Save[/b]
   [b]s[/b] → pick mode (concat/separate) + format + path → Save.

Press [b]?[/b] anytime for the full keybinding reference.

[dim]Press any key to start.[/dim]
"""


def should_show_welcome() -> bool:
    return not _FLAG_FILE.exists()


def mark_welcomed() -> None:
    _FLAG_DIR.mkdir(parents=True, exist_ok=True)
    _FLAG_FILE.touch()


class WelcomeScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    WelcomeScreen { align: center middle; }
    WelcomeScreen > Vertical {
        width: 68; height: auto; padding: 1 2;
        border: tall $accent; background: $panel;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(_GUIDE)

    def on_key(self, event: events.Key) -> None:
        mark_welcomed()
        self.dismiss()

    def on_click(self, event: events.Click) -> None:
        mark_welcomed()
        self.dismiss()
