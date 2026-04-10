"""Single-line modal prompt for a voice description, with history."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView


class DescribePrompt(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    DEFAULT_CSS = """
    DescribePrompt { align: center middle; }
    DescribePrompt > Vertical {
        width: 72; height: auto; padding: 1 2;
        border: tall $warning; background: $panel;
    }
    DescribePrompt Input { margin-top: 1; }
    DescribePrompt #history {
        height: auto; max-height: 8; margin-top: 1;
        border: tall $primary; display: none;
    }
    DescribePrompt #history_label { display: none; }
    """

    def __init__(
        self,
        scope: str,
        default: str = "",
        history: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._scope = scope
        self._default = default
        self._history = history or []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"[b]Isolate voice on {self._scope}[/b]")
            yield Label("Describe the voice you want to keep:")
            yield Input(
                value=self._default,
                placeholder="e.g. 'man speaking', 'woman singing', 'child laughing'",
                id="desc",
            )
            yield Label("[dim]Recent descriptions (click or Enter):[/dim]", id="history_label")
            yield ListView(id="history")

    def on_mount(self) -> None:
        self.query_one("#desc", Input).focus()
        if self._history:
            lv = self.query_one("#history", ListView)
            lv.styles.display = "block"
            self.query_one("#history_label").styles.display = "block"
            for desc in self._history:
                lv.append(ListItem(Label(desc)))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.dismiss(value or None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        label = event.item.query_one(Label)
        picked = str(label.renderable).strip()
        if picked:
            self.query_one("#desc", Input).value = picked
            self.dismiss(picked)

    def action_cancel(self) -> None:
        self.dismiss(None)
