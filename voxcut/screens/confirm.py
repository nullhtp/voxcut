"""Simple yes/no confirmation modal."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("escape", "no", "no"),
        Binding("y", "yes", "yes"),
        Binding("n", "no", "no"),
    ]

    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Vertical {
        width: 50; height: auto; padding: 1 2;
        border: tall $warning; background: $panel;
    }
    ConfirmScreen #buttons { height: auto; align: right middle; margin-top: 1; }
    ConfirmScreen Button { margin-left: 2; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._message)
            with Horizontal(id="buttons"):
                yield Button("No (n)", id="no")
                yield Button("Yes (y)", variant="primary", id="yes")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)
