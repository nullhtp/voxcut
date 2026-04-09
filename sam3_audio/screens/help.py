"""Help overlay with keybinding reference."""
from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

_HELP = """\
[b cyan]sam3-audio — keybindings[/b cyan]

[b]Playback[/b]
  space           play / pause
  ← / →           seek ±5s
  shift+← / →     seek ±1s
  - / +           speed 0.5× .. 2.0×
  click bar       seek to clicked position

[b]Fragments[/b]
  i               mark in-point
  o               mark out-point (adds fragment)
  enter           play selected fragment (auto-stops at out-point)
  x               delete selected fragment
  u               undo last delete
  s               save fragments (concat / separate)

[b]Voice isolation[/b]
  d               isolate voice (selected fragment or whole file)
  ctrl+k          cancel in-flight isolation (result is dropped)

[b]Files & misc[/b]
  f               open another file
  ?               show this help
  q               quit

Press any key to dismiss.
"""


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Vertical {
        width: 64; height: auto; padding: 1 2;
        border: tall $accent; background: $panel;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(_HELP)

    def on_key(self, event: events.Key) -> None:  # noqa: ARG002
        self.dismiss()

    def on_click(self, event: events.Click) -> None:  # noqa: ARG002
        self.dismiss()
