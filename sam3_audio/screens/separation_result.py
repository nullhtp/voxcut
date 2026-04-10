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
from ..timeutil import fmt_time

_BAR_WIDTH = 30


@dataclass
class SeparationResultData:
    description: str
    sample_rate: int
    target: np.ndarray    # mono float32
    residual: np.ndarray  # mono float32
    out_base: Path        # destination basename (no suffix)
    original: np.ndarray | None = None   # mono float32, source segment
    original_sr: int = 0


@dataclass(frozen=True)
class ResultDecision:
    action: str  # "keep" | "rerun" | "load"
    kept_paths: tuple[Path, Path] | None = None


class SeparationResultScreen(ModalScreen[ResultDecision | None]):
    BINDINGS = [
        Binding("escape", "discard", "discard"),
        Binding("t", "play_target", "play target"),
        Binding("r", "play_residual", "play residual"),
        Binding("o", "play_original", "play original"),
        Binding("space", "stop", "stop"),
        Binding("k", "keep", "keep"),
        Binding("l", "load_target", "load target"),
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
    SeparationResultScreen #playback_bar { color: $accent; margin-top: 1; }
    """

    def __init__(self, data: SeparationResultData) -> None:
        super().__init__()
        self._data = data
        self._players: dict[str, Player] = {}
        self._active_name: str = ""
        self._timer = None

    # --- layout ---

    def compose(self) -> ComposeResult:
        d = self._data
        t_dur = d.target.shape[0] / d.sample_rate if d.sample_rate else 0.0
        r_dur = d.residual.shape[0] / d.sample_rate if d.sample_rate else 0.0
        with Vertical():
            yield Label(f"[b]Separation result[/b] — {d.description!r}")
            yield Static(
                f"target: {t_dur:.2f}s    residual: {r_dur:.2f}s"
            )
            yield Static("", id="playback_bar")
            with Horizontal(id="play_row"):
                yield Button("▶ Target (t)", id="b_target")
                yield Button("▶ Residual (r)", id="b_residual")
                if d.original is not None:
                    yield Button("▶ Original (o)", id="b_original")
                yield Button("■ Stop (␣)", id="b_stop")
            with Horizontal(id="choice_row"):
                yield Button("Keep (k)", variant="primary", id="b_keep")
                yield Button("Load target (l)", variant="success", id="b_load")
                yield Button("Re-run (shift+r)", id="b_rerun")
                yield Button("Discard (esc)", id="b_discard")

    def on_mount(self) -> None:
        d = self._data
        self._players["target"] = Player(d.target, d.sample_rate)
        self._players["residual"] = Player(d.residual, d.sample_rate)
        if d.original is not None and d.original_sr > 0:
            self._players["original"] = Player(d.original, d.original_sr)
        self._timer = self.set_interval(0.1, self._tick)

    def on_unmount(self) -> None:
        self._stop()
        for p in self._players.values():
            p.close()

    # --- playback ---

    def _play(self, name: str) -> None:
        self._stop()
        p = self._players.get(name)
        if p is None:
            self.app.bell(); return
        p.seek_to(0.0)
        if not p.playing:
            p.toggle()
        self._active_name = name

    def _stop(self) -> None:
        if self._active_name:
            p = self._players.get(self._active_name)
            if p and p.playing:
                p.toggle()
        self._active_name = ""
        self._update_bar()

    def _tick(self) -> None:
        self._update_bar()

    def _update_bar(self) -> None:
        try:
            bar_widget = self.query_one("#playback_bar", Static)
        except Exception:
            return
        if not self._active_name:
            bar_widget.update("")
            return
        p = self._players.get(self._active_name)
        if p is None:
            bar_widget.update("")
            return
        pos, dur = p.position, p.duration
        if not p.playing and pos >= dur - 0.05:
            # finished
            self._active_name = ""
            bar_widget.update("")
            return
        frac = min(1.0, pos / dur) if dur > 0 else 0.0
        filled = int(_BAR_WIDTH * frac)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        bar_widget.update(
            f"▶ {self._active_name}  {bar}  "
            f"{fmt_time(pos)} / {fmt_time(dur)}"
        )

    # --- actions ---

    def action_play_target(self) -> None: self._play("target")
    def action_play_residual(self) -> None: self._play("residual")
    def action_play_original(self) -> None: self._play("original")
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

    def action_load_target(self) -> None:
        self._stop()
        d = self._data
        target_path = d.out_base.with_name(d.out_base.name + "_target.wav")
        residual_path = d.out_base.with_name(d.out_base.name + "_residual.wav")
        save_wav(target_path, d.target, d.sample_rate)
        save_wav(residual_path, d.residual, d.sample_rate)
        self.dismiss(
            ResultDecision(action="load", kept_paths=(target_path, residual_path))
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
            case "b_original": self._play("original")
            case "b_stop": self._stop()
            case "b_keep": self.action_keep()
            case "b_load": self.action_load_target()
            case "b_rerun": self.action_rerun()
            case "b_discard": self.action_discard()
