"""Audition / keep / re-run a SAM-Audio separation result."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ..player import Player
from ..separator import save_wav


@dataclass
class SeparationResultData:
    description: str
    sample_rate: int
    target: np.ndarray    # mono float32
    residual: np.ndarray  # mono float32
    out_base: Path        # destination basename (no suffix)


@dataclass(frozen=True)
class ResultDecision:
    action: str  # "keep" | "rerun"
    kept_paths: tuple[Path, Path] | None = None


class SeparationResultScreen(ModalScreen[ResultDecision | None]):
    BINDINGS = [
        Binding("escape", "discard", "discard"),
        Binding("t", "play_target", "play target"),
        Binding("r", "play_residual", "play residual"),
        Binding("space", "stop", "stop"),
        Binding("k", "keep", "keep"),
        Binding("shift+r", "rerun", "re-run"),
    ]

    DEFAULT_CSS = """
    SeparationResultScreen { align: center middle; }
    SeparationResultScreen > Vertical {
        width: 80; height: auto; padding: 1 2;
        border: tall $accent; background: $panel;
    }
    SeparationResultScreen #play_row, SeparationResultScreen #choice_row {
        height: auto; align: center middle; margin-top: 1;
    }
    SeparationResultScreen Button { margin: 0 1; }
    """

    def __init__(self, data: SeparationResultData) -> None:
        super().__init__()
        self._data = data
        self._target_player: Optional[Player] = None
        self._residual_player: Optional[Player] = None
        self._active: Optional[Player] = None

    # --- layout ---

    def compose(self) -> ComposeResult:
        d = self._data
        t_dur = d.target.shape[0] / d.sample_rate if d.sample_rate else 0.0
        r_dur = d.residual.shape[0] / d.sample_rate if d.sample_rate else 0.0
        with Vertical():
            yield Label(f"[b]Separation result[/b] — {d.description!r}")
            yield Static(
                f"target:   {t_dur:6.2f}s    residual: {r_dur:6.2f}s"
            )
            yield Static("", id="play_status")
            with Horizontal(id="play_row"):
                yield Button("▶ Target (t)", id="b_target")
                yield Button("▶ Residual (r)", id="b_residual")
                yield Button("■ Stop (␣)", id="b_stop")
            with Horizontal(id="choice_row"):
                yield Button("Keep (k)", variant="primary", id="b_keep")
                yield Button("Re-run (shift+r)", id="b_rerun")
                yield Button("Discard (esc)", id="b_discard")

    def on_mount(self) -> None:
        self._target_player = Player(self._data.target, self._data.sample_rate)
        self._residual_player = Player(self._data.residual, self._data.sample_rate)

    def on_unmount(self) -> None:
        self._stop()
        for p in (self._target_player, self._residual_player):
            if p is not None:
                p.close()

    # --- playback ---

    def _play(self, which: str) -> None:
        self._stop()
        p = self._target_player if which == "target" else self._residual_player
        if p is None:
            return
        p.seek_to(0.0)
        if not p.playing:
            p.toggle()
        self._active = p
        self.query_one("#play_status", Static).update(f"playing: {which}")

    def _stop(self) -> None:
        if self._active and self._active.playing:
            self._active.toggle()
        self._active = None
        try:
            self.query_one("#play_status", Static).update("")
        except Exception:
            pass

    # --- actions ---

    def action_play_target(self) -> None: self._play("target")
    def action_play_residual(self) -> None: self._play("residual")
    def action_stop(self) -> None: self._stop()

    def action_keep(self) -> None:
        self._stop()
        d = self._data
        target_path = d.out_base.with_name(d.out_base.name + "_target.wav")
        residual_path = d.out_base.with_name(d.out_base.name + "_residual.wav")
        save_wav(target_path, d.target, d.sample_rate)
        save_wav(residual_path, d.residual, d.sample_rate)
        self.dismiss(
            ResultDecision(action="keep", kept_paths=(target_path, residual_path))
        )

    def action_rerun(self) -> None:
        self._stop()
        self.dismiss(ResultDecision(action="rerun"))

    def action_discard(self) -> None:
        self._stop()
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "b_target": self._play("target")
            case "b_residual": self._play("residual")
            case "b_stop": self._stop()
            case "b_keep": self.action_keep()
            case "b_rerun": self.action_rerun()
            case "b_discard": self.action_discard()
