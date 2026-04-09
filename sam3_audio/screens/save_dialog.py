"""Modal save dialog: mode + output path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet


@dataclass(frozen=True)
class SaveRequest:
    mode: str  # "concat" | "separate"
    path: Path


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
    SaveDialog #buttons {
        height: auto; align: right middle; margin-top: 1;
    }
    SaveDialog Button { margin-left: 2; }
    """

    def __init__(self, src: Path) -> None:
        super().__init__()
        self._src = src
        self._concat_default = src.with_name(src.stem + "_cut" + src.suffix)
        self._split_default = src.with_name(src.stem + "_cut")

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
            yield Label("Output path:")
            yield Input(value=str(self._concat_default), id="path")
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", variant="primary", id="save")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        path_input = self.query_one("#path", Input)
        if event.pressed.id == "concat":
            path_input.value = str(self._concat_default)
        else:
            path_input.value = str(self._split_default)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:  # noqa: ARG002
        self._submit()

    def _submit(self) -> None:
        mode = (
            "concat"
            if self.query_one("#concat", RadioButton).value
            else "separate"
        )
        raw = self.query_one("#path", Input).value.strip()
        if not raw:
            raw = str(
                self._concat_default if mode == "concat" else self._split_default
            )
        self.dismiss(SaveRequest(mode=mode, path=Path(raw)))

    def action_cancel(self) -> None:
        self.dismiss(None)
