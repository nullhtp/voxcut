"""Single-line modal prompt for a voice description."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class DescribePrompt(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    DEFAULT_CSS = """
    DescribePrompt { align: center middle; }
    DescribePrompt > Vertical {
        width: 72; height: auto; padding: 1 2;
        border: tall $warning; background: $panel;
    }
    DescribePrompt Input { margin-top: 1; }
    """

    def __init__(self, scope: str, default: str = "") -> None:
        super().__init__()
        self._scope = scope
        self._default = default

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"[b]Isolate voice on {self._scope}[/b]")
            yield Label("Describe the voice you want to keep:")
            yield Input(
                value=self._default,
                placeholder="e.g. 'man speaking', 'woman singing', 'child laughing'",
                id="desc",
            )

    def on_mount(self) -> None:
        self.query_one("#desc", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
