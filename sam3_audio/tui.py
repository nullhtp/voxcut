"""Textual TUI: play, cut fragments, isolate voices."""
from __future__ import annotations

import enum
import tempfile
import threading
from pathlib import Path
from typing import Optional

import soundfile as sf
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from . import ffmpeg
from .fragment import Fragment
from .player import Player
from .separator import SamSeparator, load_mono, slice_mono
from .timeutil import fmt_time

try:
    from textual_fspicker import FileOpen, Filters
    _HAS_FSPICKER = True
except ImportError:
    _HAS_FSPICKER = False

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aiff"}
_PROGRESS_BAR_WIDTH = 40


class _Prompt(enum.Enum):
    NONE = enum.auto()
    SAVE_MODE = enum.auto()
    SAVE_PATH = enum.auto()
    SEP_DESC = enum.auto()


def _safe_slug(text: str, limit: int = 32) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:limit]


class AudioTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #status { height: 3; padding: 1 2; background: $boost; }
    #progress { height: 1; padding: 0 2; color: $accent; }
    #marks { height: 3; padding: 0 2; }
    #list { height: 1fr; border: tall $primary; }
    #prompt { height: 3; border: tall $warning; display: none; }
    """

    BINDINGS = [
        Binding("space", "toggle_play", "play/pause"),
        Binding("right", "seek(5)", "+5s"),
        Binding("left", "seek(-5)", "-5s"),
        Binding("shift+right", "seek(1)", "+1s"),
        Binding("shift+left", "seek(-1)", "-1s"),
        Binding("plus,equals_sign,equal", "speed(0.1)", "speed+"),
        Binding("minus", "speed(-0.1)", "speed-"),
        Binding("i", "mark_in", "mark in"),
        Binding("o", "mark_out", "mark out"),
        Binding("x", "del_fragment", "delete"),
        Binding("s", "save", "save cuts"),
        Binding("d", "separate", "isolate voice"),
        Binding("f", "open_file", "open file"),
        Binding("q", "quit", "quit"),
    ]

    position = reactive(0.0)

    def __init__(self, src: Path, wav: Path, separator: SamSeparator):
        super().__init__()
        self.src = src
        self.wav_path = wav
        self.separator = separator
        self.fragments: list[Fragment] = []
        self.in_point: Optional[float] = None
        self.player = self._load_player(wav)
        self._prompt_mode = _Prompt.NONE
        self._save_mode: Optional[str] = None
        self._default_out: str = ""

    @staticmethod
    def _load_player(wav: Path) -> Player:
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        return Player(data, sr)

    # --- layout ---

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(self._header_text(), id="status")
        yield Static("", id="progress")
        yield Static("", id="marks")
        yield ListView(id="list")
        yield Input(placeholder="", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)
        self._refresh_marks()
        self._refresh_list()

    def on_unmount(self) -> None:
        self.player.close()

    # --- widget accessors (keeps IDs in one place) ---

    @property
    def _w_status(self) -> Static: return self.query_one("#status", Static)
    @property
    def _w_progress(self) -> Static: return self.query_one("#progress", Static)
    @property
    def _w_marks(self) -> Static: return self.query_one("#marks", Static)
    @property
    def _w_list(self) -> ListView: return self.query_one("#list", ListView)
    @property
    def _w_prompt(self) -> Input: return self.query_one("#prompt", Input)

    # --- view helpers ---

    def _header_text(self) -> str:
        return f"File: {self.src.name}   duration: {fmt_time(self.player.duration)}"

    def _tick(self) -> None:
        self.position = self.player.position
        pos, dur = self.player.position, self.player.duration
        filled = int(_PROGRESS_BAR_WIDTH * (pos / dur)) if dur else 0
        bar = "█" * filled + "░" * (_PROGRESS_BAR_WIDTH - filled)
        state = "▶" if self.player.playing else "⏸"
        self._w_progress.update(
            f"{state} {bar} {fmt_time(pos)} / {fmt_time(dur)}   "
            f"speed {self.player.speed:.1f}x"
        )

    def _refresh_marks(self) -> None:
        ip = fmt_time(self.in_point) if self.in_point is not None else "—"
        self._w_marks.update(
            f"in: {ip}    (i=mark in, o=mark out, s=save cuts, d=isolate voice)"
        )

    def _refresh_list(self) -> None:
        lv = self._w_list
        lv.clear()
        for i, frag in enumerate(self.fragments):
            lv.append(ListItem(Label(frag.label(i))))

    def _set_status(self, msg: str) -> None:
        self._w_status.update(msg)

    def _selected_fragment(self) -> Optional[Fragment]:
        idx = self._w_list.index
        if idx is None or not (0 <= idx < len(self.fragments)):
            return None
        return self.fragments[idx]

    # --- prompt helpers ---

    def _open_prompt(self, mode: _Prompt, placeholder: str) -> None:
        self._prompt_mode = mode
        inp = self._w_prompt
        inp.placeholder = placeholder
        inp.value = ""
        inp.styles.display = "block"
        inp.focus()

    def _close_prompt(self) -> None:
        self._prompt_mode = _Prompt.NONE
        self._w_prompt.styles.display = "none"

    # --- actions ---

    def action_toggle_play(self) -> None:
        self.player.toggle()

    def action_seek(self, delta: float) -> None:
        self.player.seek(delta)

    def action_speed(self, delta: float) -> None:
        self.player.set_speed(round(self.player.speed + delta, 2))

    def action_mark_in(self) -> None:
        self.in_point = self.player.position
        self._refresh_marks()

    def action_mark_out(self) -> None:
        if self.in_point is None:
            self.bell(); return
        start, end = self.in_point, self.player.position
        if end <= start:
            self.bell(); return
        self.fragments.append(Fragment(start, end))
        self.fragments.sort(key=lambda f: f.start)
        self.in_point = None
        self._refresh_marks()
        self._refresh_list()

    def action_del_fragment(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            self.bell(); return
        self.fragments.remove(frag)
        self._refresh_list()

    def action_save(self) -> None:
        if not self.fragments:
            self.bell(); return
        self._open_prompt(
            _Prompt.SAVE_MODE,
            "save mode: (c)oncat single file  /  (s)eparate files",
        )

    def action_separate(self) -> None:
        frag = self._selected_fragment()
        if frag is not None:
            scope = (
                f"fragment #{self.fragments.index(frag) + 1} "
                f"({fmt_time(frag.start)}–{fmt_time(frag.end)})"
            )
        else:
            scope = "whole file"
        self._open_prompt(
            _Prompt.SEP_DESC,
            f"describe voice to isolate on {scope} (e.g. 'man speaking')",
        )

    def action_open_file(self) -> None:
        if not _HAS_FSPICKER:
            self._set_status("install textual-fspicker: pip install textual-fspicker")
            self.bell(); return
        filters = Filters(
            ("Audio", lambda p: p.suffix.lower() in _AUDIO_EXTS),
            ("All", lambda p: True),
        )
        self.push_screen(
            FileOpen(location=str(self.src.parent), filters=filters),
            self._on_file_chosen,
        )

    def _on_file_chosen(self, path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            self._set_status(f"loading {path.name} …")
            new_tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            ffmpeg.decode_to_wav(path, new_tmp)
            new_player = self._load_player(new_tmp)
            self.player.close()
            old_tmp = self.wav_path
            self.player = new_player
            self.src = path
            self.wav_path = new_tmp
            try: old_tmp.unlink()
            except OSError: pass
            self.fragments.clear()
            self.in_point = None
            self._refresh_marks()
            self._refresh_list()
            self._set_status(self._header_text())
        except Exception as e:
            self._set_status(f"open failed: {e}")

    # --- prompt dispatch ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        match self._prompt_mode:
            case _Prompt.SAVE_MODE:
                self._handle_save_mode(value)
            case _Prompt.SAVE_PATH:
                self._handle_save_path(value)
            case _Prompt.SEP_DESC:
                self._handle_sep_desc(value)
            case _Prompt.NONE:
                pass

    def _handle_save_mode(self, value: str) -> None:
        first = value[:1].lower()
        if first == "c":
            self._save_mode = "concat"
            default = self.src.with_name(self.src.stem + "_cut" + self.src.suffix)
        elif first == "s":
            self._save_mode = "separate"
            default = self.src.with_name(self.src.stem + "_cut")
        else:
            self.bell(); return
        self._default_out = str(default)
        self._open_prompt(
            _Prompt.SAVE_PATH,
            f"output path (default: {self._default_out}) — enter to accept",
        )

    def _handle_save_path(self, value: str) -> None:
        out = Path(value or self._default_out)
        self._close_prompt()
        try:
            if self._save_mode == "concat":
                ffmpeg.concat_cuts(self.src, self.fragments, out)
            else:
                ffmpeg.split_cuts(self.src, self.fragments, out)
            self._set_status(f"saved → {out}")
        except Exception as e:
            self._set_status(f"save failed: {e}")

    def _handle_sep_desc(self, value: str) -> None:
        if not value:
            self.bell(); return
        frag = self._selected_fragment()
        self._close_prompt()
        self._set_status(f"isolating {value!r} … (this can take a while)")
        threading.Thread(
            target=self._run_separation,
            args=(value, frag),
            daemon=True,
        ).start()

    def _run_separation(self, description: str, frag: Optional[Fragment]) -> None:
        try:
            self.call_from_thread(self._set_status, "loading SAM-Audio …")
            mono, sr = load_mono(self.wav_path)
            if frag is not None:
                mono = slice_mono(mono, sr, frag.start, frag.end)
                tag = f"frag{self.fragments.index(frag) + 1:02d}"
            else:
                tag = "full"
            self.call_from_thread(
                self._set_status, f"separating {description!r} on {tag} …"
            )
            out_base = self.src.with_name(
                f"{self.src.stem}_{tag}_{_safe_slug(description)}"
            )
            result = self.separator.separate(mono, sr, description, out_base)
            self.call_from_thread(
                self._set_status,
                f"isolated → {result.target_path.name} + {result.residual_path.name}",
            )
        except Exception as e:
            self.call_from_thread(self._set_status, f"separate failed: {e}")
