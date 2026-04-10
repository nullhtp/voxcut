"""Modal save dialog: mode + output path + format."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Select

_FORMAT_CHOICES = [
    ("Same as input", "same"),
    ("WAV (.wav)", ".wav"),
    ("MP3 (.mp3)", ".mp3"),
    ("FLAC (.flac)", ".flac"),
    ("OGG (.ogg)", ".ogg"),
]


@dataclass(frozen=True)
class SaveRequest:
    mode: str  # "concat" | "separate"
    path: Path
    format: str  # "same" | ".wav" | ".mp3" | ".flac" | ".ogg"


class SaveDialog(ModalScreen[SaveRequest | None]):
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    DEFAULT_CSS = """
    SaveDialog { align: center middle; }
    SaveDialog > Vertical {
        width: 76; height: auto; padding: 1 2;
        border: tall $warning; background: $panel;
    }
    SaveDialog RadioSet { margin: 1 0; width: 100%; }
    SaveDialog Input { margin-top: 1; }
    SaveDialog Select { margin-top: 1; width: 100%; }
    SaveDialog #buttons {
        height: auto; align: right middle; margin-top: 1;
    }
    SaveDialog Button { margin-left: 2; }
    """

    def __init__(self, src: Path) -> None:
        super().__init__()
        self._src = src

    def _defaults(self, mode: str, fmt: str) -> str:
        ext = self._src.suffix if fmt == "same" else fmt
        if mode == "concat":
            return str(self._src.with_name(self._src.stem + "_cut" + ext))
        return str(self._src.with_name(self._src.stem + "_cut"))

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("[b]Save fragments[/b]")
            with RadioSet(id="mode"):
                yield RadioButton(
                    "Concatenate into a single file", value=True, id="concat"
                )
                yield RadioButton(
                    "Separate files in a directory", id="separate"
                )
            yield Label("Output format:")
            yield Select(_FORMAT_CHOICES, value="same", id="format")
            yield Label("Output path:")
            yield Input(value=self._defaults("concat", "same"), id="path")
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="save")

    def _update_path(self) -> None:
        mode = "concat" if self.query_one("#concat", RadioButton).value else "separate"
        fmt = self.query_one("#format", Select).value
        self.query_one("#path", Input).value = self._defaults(mode, str(fmt))

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._update_path()

    def on_select_changed(self, event: Select.Changed) -> None:
        self._update_path()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        mode = (
            "concat"
            if self.query_one("#concat", RadioButton).value
            else "separate"
        )
        fmt = str(self.query_one("#format", Select).value)
        raw = self.query_one("#path", Input).value.strip()
        if not raw:
            raw = self._defaults(mode, fmt)
        self.dismiss(SaveRequest(mode=mode, path=Path(raw), format=fmt))

    def action_cancel(self) -> None:
        self.dismiss(None)
