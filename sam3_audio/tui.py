"""Textual TUI: play, cut fragments, isolate voices."""
from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from . import ffmpeg, session, waveform
from .fragment import Fragment
from .player import Player
from .screens import (
    DescribePrompt,
    HelpScreen,
    ResultDecision,
    SaveDialog,
    SaveRequest,
    SeparationResultData,
    SeparationResultScreen,
)
from .separator import SamSeparator, slice_mono
from .timeutil import fmt_time

try:
    from textual_fspicker import FileOpen, Filters
    _HAS_FSPICKER = True
except ImportError:
    _HAS_FSPICKER = False

_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".m4a", ".ogg",
    ".opus", ".aac", ".wma", ".aiff",
}
_PROGRESS_BAR_WIDTH = 40
_WAVEFORM_DEFAULT_WIDTH = 80


def _safe_slug(text: str, limit: int = 32) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:limit]


class _CancelledSeparation(Exception):
    """Raised from the progress callback to hard-abort an in-flight run."""


@dataclass
class _PendingSeparation:
    """Context carried across describe → run → result → optional rerun."""
    description: str
    mono: np.ndarray
    sr: int
    tag: str
    out_base: Path
    fragment: Optional[Fragment]


class AudioTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #status { height: 1; padding: 0 2; background: $boost; }
    #progress { height: 1; padding: 0 2; color: $accent; }
    #waveform { height: 1; padding: 0 2; }
    #marks { height: 1; padding: 0 2; color: $text-muted; }
    #sep_progress { height: 1; padding: 0 2; color: $warning; display: none; }
    #list { height: 1fr; border: tall $primary; }
    #log { height: 6; border: tall $accent-lighten-1; }
    """

    SEP_BAR_WIDTH = 30

    BINDINGS = [
        Binding("space", "toggle_play", "play/pause"),
        Binding("right", "seek(5)", "+5s"),
        Binding("left", "seek(-5)", "-5s"),
        Binding("shift+right", "seek(1)", "+1s"),
        Binding("shift+left", "seek(-1)", "-1s"),
        Binding("plus,equals_sign,equal", "speed(0.1)", "speed+"),
        Binding("minus", "speed(-0.1)", "speed-"),
        Binding("v", "gain(3)", "vol+"),
        Binding("V", "gain(-3)", "vol-"),
        Binding("i", "mark_in", "mark in"),
        Binding("o", "mark_out", "mark out"),
        Binding("p", "play_fragment", "play fragment"),
        Binding("x", "del_fragment", "delete"),
        Binding("u", "undo_delete", "undo"),
        Binding("S", "split_fragment", "split at cursor"),
        Binding("m", "merge_fragments", "merge with next"),
        Binding("left_square_bracket", "nudge_start(-0.1)", "in ←"),
        Binding("right_square_bracket", "nudge_end(0.1)", "out →"),
        Binding("left_curly_bracket", "nudge_start(0.1)", "in →"),
        Binding("right_curly_bracket", "nudge_end(-0.1)", "out ←"),
        Binding("s", "save", "save cuts"),
        Binding("d", "separate", "isolate voice"),
        Binding("D", "separate_all", "isolate all"),
        Binding("ctrl+k", "cancel_separation", "cancel isolation"),
        Binding("f", "open_file", "open file"),
        Binding("question_mark,?", "help", "help"),
        Binding("q", "quit_app", "quit"),
    ]

    position = reactive(0.0)

    def __init__(
        self,
        src: Optional[Path],
        wav: Optional[Path],
        separator: SamSeparator,
    ):
        super().__init__()
        self.src: Optional[Path] = src
        self.wav_path: Optional[Path] = wav
        self.separator = separator
        self.fragments: list[Fragment] = []
        self.in_point: Optional[float] = None
        self.player = self._load_player(wav) if wav is not None else self._silent_player()
        self._peaks = self._compute_peaks()
        self._last_description = ""
        self._description_history: list[str] = []
        self._undo_stack: list[Fragment] = []
        self._play_until: Optional[float] = None
        self._playing_frag_idx: int | None = None
        self._sep_busy = False
        self._sep_generation = 0

    # --- file loading ---

    @staticmethod
    def _load_player(wav: Path) -> Player:
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        return Player(data, sr)

    @staticmethod
    def _silent_player() -> Player:
        return Player(np.zeros((1, 1), dtype=np.float32), 44100)

    def _waveform_width(self) -> int:
        try:
            w = self._w_waveform.size.width
            # subtract CSS padding (0 2 = 2 chars each side)
            return max(10, w - 4) if w > 4 else _WAVEFORM_DEFAULT_WIDTH
        except Exception:
            return _WAVEFORM_DEFAULT_WIDTH

    def _compute_peaks(self) -> np.ndarray:
        data = self.player.data
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return waveform.compute_peaks(mono, self._waveform_width())

    # --- layout ---

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(self._header_text(), id="status")
        yield Static("", id="progress")
        yield Static("", id="waveform")
        yield Static("", id="marks")
        yield Static("", id="sep_progress")
        yield ListView(id="list")
        log = RichLog(id="log", highlight=False, markup=True, wrap=True)
        log.can_focus = False
        yield log
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)
        self._refresh_marks()
        self._refresh_list()
        self.call_after_refresh(self._recompute_peaks_if_needed)
        if self.src is None:
            self._log("no file loaded — [b]f[/b] to open, [b]?[/b] for help")
            self.call_after_refresh(self._prompt_initial_file)
            return
        self._load_session()
        self._log(
            f"[dim]opened[/dim] {self.src.name} "
            f"([dim]{fmt_time(self.player.duration)}[/dim])"
        )
        self._log("press [b]?[/b] for help")

    def on_resize(self, event: events.Resize) -> None:  # noqa: ARG002
        self._recompute_peaks_if_needed()

    def _recompute_peaks_if_needed(self) -> None:
        new_w = self._waveform_width()
        if self._peaks.shape[0] != new_w:
            self._peaks = self._compute_peaks()

    def on_unmount(self) -> None:
        self._save_session()
        self.player.close()

    def _prompt_initial_file(self) -> None:
        if not _HAS_FSPICKER:
            self._log("[red]no file given and textual-fspicker is missing[/red]")
            return
        filters = Filters(
            ("Audio", lambda p: p.suffix.lower() in _AUDIO_EXTS),
            ("All", lambda p: True),
        )
        self.push_screen(
            FileOpen(location=str(Path.cwd()), filters=filters),
            self._on_initial_file_chosen,
        )

    def _on_initial_file_chosen(self, path: Optional[Path]) -> None:
        if path is None:
            self.exit()
            return
        self._on_file_chosen(path)

    # --- widget accessors ---

    @property
    def _w_status(self) -> Static: return self.query_one("#status", Static)
    @property
    def _w_progress(self) -> Static: return self.query_one("#progress", Static)
    @property
    def _w_waveform(self) -> Static: return self.query_one("#waveform", Static)
    @property
    def _w_marks(self) -> Static: return self.query_one("#marks", Static)
    @property
    def _w_list(self) -> ListView: return self.query_one("#list", ListView)
    @property
    def _w_log(self) -> RichLog: return self.query_one("#log", RichLog)
    @property
    def _w_sep_progress(self) -> Static:
        return self.query_one("#sep_progress", Static)

    # --- view helpers ---

    def _header_text(self) -> str:
        if self.src is None:
            return "No file loaded — press f to open"
        return (
            f"File: {self.src.name}   "
            f"duration: {fmt_time(self.player.duration)}"
        )

    def _require_file(self) -> bool:
        if self.src is None:
            self._log("[yellow]no file loaded — press f to open[/yellow]")
            self.bell()
            return False
        return True

    def _tick(self) -> None:
        self.position = self.player.position
        pos, dur = self.player.position, self.player.duration

        # auto-stop at fragment end
        if self._play_until is not None and pos >= self._play_until:
            if self.player.playing:
                self.player.toggle()
            self._play_until = None
            self._playing_frag_idx = None
            self._refresh_list()

        filled = int(_PROGRESS_BAR_WIDTH * (pos / dur)) if dur else 0
        bar = "█" * filled + "░" * (_PROGRESS_BAR_WIDTH - filled)
        state = "▶" if self.player.playing else "⏸"
        gain_str = f"   {self.player.gain_db:+.0f}dB" if self.player.gain_db else ""
        self._w_progress.update(
            f"{state} {bar} {fmt_time(pos)} / {fmt_time(dur)}   "
            f"speed {self.player.speed:.1f}x{gain_str}"
        )
        cursor_frac = (pos / dur) if dur else 0.0
        self._w_waveform.update(waveform.render(
            self._peaks,
            cursor_frac,
            duration=dur,
            fragments=[(f.start, f.end) for f in self.fragments],
            in_point=self.in_point,
        ))

    def _refresh_marks(self) -> None:
        ip = fmt_time(self.in_point) if self.in_point is not None else "—"
        n = len(self.fragments)
        total = sum(f.duration for f in self.fragments)
        frag_info = f"{n} fragments ({fmt_time(total)})" if n else "no fragments"
        self._w_marks.update(
            f"in: {ip}    {frag_info} — enter to play, ? for help"
        )

    def _refresh_list(self, select: int | None = None) -> None:
        lv = self._w_list
        prev = select if select is not None else lv.index
        lv.clear()
        for i, frag in enumerate(self.fragments):
            prefix = "▶" if i == self._playing_frag_idx else " "
            lv.append(ListItem(Label(f"{prefix}{frag.label(i)}")))
        if self.fragments and prev is not None:
            lv.index = min(prev, len(self.fragments) - 1)

    def _log(self, msg: str) -> None:
        self._w_log.write(msg)

    def _selected_fragment(self) -> Optional[Fragment]:
        idx = self._w_list.index
        if idx is None or not (0 <= idx < len(self.fragments)):
            return None
        return self.fragments[idx]

    # --- actions ---

    def action_toggle_play(self) -> None:
        self.player.toggle()
        if not self.player.playing:
            self._play_until = None
            if self._playing_frag_idx is not None:
                self._playing_frag_idx = None
                self._refresh_list()

    def action_seek(self, delta: float) -> None:
        self.player.seek(delta)
        self._play_until = None
        if self._playing_frag_idx is not None:
            self._playing_frag_idx = None
            self._refresh_list()

    def action_speed(self, delta: float) -> None:
        self.player.set_speed(round(self.player.speed + delta, 2))
        self._log(f"speed → {self.player.speed:.1f}x")

    def action_gain(self, db: float) -> None:
        self.player.set_gain(round(self.player.gain_db + db, 1))
        self._log(f"gain → {self.player.gain_db:+.0f} dB")

    def action_mark_in(self) -> None:
        self.in_point = self.player.position
        self._refresh_marks()
        self._log(f"in = {fmt_time(self.in_point)}")

    def action_mark_out(self) -> None:
        if self.in_point is None:
            self.bell(); return
        start, end = self.in_point, self.player.position
        if end <= start:
            self._log("[yellow]out must be after in[/yellow]")
            self.bell(); return
        frag = Fragment(start, end)
        self.fragments.append(frag)
        self.fragments.sort(key=lambda f: f.start)
        new_idx = self.fragments.index(frag)
        self.in_point = None
        self._refresh_marks()
        self._refresh_list(select=new_idx)
        self._log(f"+ fragment {fmt_time(frag.start)} → {fmt_time(frag.end)}")

    def action_del_fragment(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            self.bell(); return
        self.fragments.remove(frag)
        self._undo_stack.append(frag)
        self._refresh_list()
        self._refresh_marks()
        self._log(f"– fragment (u to undo)")

    def action_undo_delete(self) -> None:
        if not self._undo_stack:
            self.bell(); return
        frag = self._undo_stack.pop()
        self.fragments.append(frag)
        self.fragments.sort(key=lambda f: f.start)
        self._refresh_list()
        self._refresh_marks()
        self._log(f"undo → {fmt_time(frag.start)} → {fmt_time(frag.end)}")

    def action_split_fragment(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            self.bell(); return
        pos = self.player.position
        if pos <= frag.start or pos >= frag.end:
            self._log("[yellow]cursor must be inside the fragment to split[/yellow]")
            self.bell(); return
        idx = self.fragments.index(frag)
        left = Fragment(frag.start, pos)
        right = Fragment(pos, frag.end)
        self.fragments[idx:idx + 1] = [left, right]
        self._refresh_list(select=idx)
        self._refresh_marks()
        self._log(
            f"split → {fmt_time(left.start)}–{fmt_time(left.end)} | "
            f"{fmt_time(right.start)}–{fmt_time(right.end)}"
        )

    def action_merge_fragments(self) -> None:
        idx = self._w_list.index
        if idx is None or not (0 <= idx < len(self.fragments) - 1):
            self._log("[yellow]select a fragment that has a next to merge with[/yellow]")
            self.bell(); return
        a = self.fragments[idx]
        b = self.fragments[idx + 1]
        merged = Fragment(min(a.start, b.start), max(a.end, b.end))
        self.fragments[idx:idx + 2] = [merged]
        self._refresh_list(select=idx)
        self._refresh_marks()
        self._log(
            f"merged → {fmt_time(merged.start)} – {fmt_time(merged.end)}"
        )

    def _preview_around(self, t: float, radius: float = 0.25) -> None:
        """Seek near *t* and play a short preview snippet."""
        start = max(0.0, t - radius)
        end = min(self.player.duration, t + radius)
        self.player.seek_to(start)
        self._play_until = end
        self._playing_frag_idx = None
        if not self.player.playing:
            self.player.toggle()

    def action_nudge_start(self, delta: float) -> None:
        frag = self._selected_fragment()
        if frag is None:
            self.bell(); return
        new_start = max(0.0, frag.start + delta)
        if new_start >= frag.end:
            self.bell(); return
        idx = self.fragments.index(frag)
        self.fragments[idx] = Fragment(new_start, frag.end)
        self._refresh_list()
        self._preview_around(new_start)

    def action_nudge_end(self, delta: float) -> None:
        frag = self._selected_fragment()
        if frag is None:
            self.bell(); return
        new_end = min(self.player.duration, frag.end + delta)
        if new_end <= frag.start:
            self.bell(); return
        idx = self.fragments.index(frag)
        self.fragments[idx] = Fragment(frag.start, new_end)
        self._refresh_list()
        self._preview_around(new_end)

    def action_save(self) -> None:
        if not self._require_file():
            return
        if not self.fragments:
            self._log("[yellow]no fragments to save[/yellow]")
            self.bell(); return
        self.push_screen(SaveDialog(self.src), self._handle_save)

    def action_separate(self) -> None:
        if not self._require_file():
            return
        if self._sep_busy:
            self._log("[yellow]isolation already in progress — ctrl+k to cancel[/yellow]")
            self.bell(); return
        frag = self._selected_fragment()
        if frag is not None:
            scope = (
                f"fragment #{self.fragments.index(frag) + 1} "
                f"({fmt_time(frag.start)}–{fmt_time(frag.end)})"
            )
        else:
            scope = "whole file"
        self.push_screen(
            DescribePrompt(
                scope=scope,
                default=self._last_description,
                history=self._description_history,
            ),
            lambda value: self._on_describe_initial(value, frag),
        )

    def action_separate_all(self) -> None:
        if not self._require_file():
            return
        if self._sep_busy:
            self._log("[yellow]isolation already in progress — ctrl+k to cancel[/yellow]")
            self.bell(); return
        if not self.fragments:
            self._log("[yellow]no fragments to isolate[/yellow]")
            self.bell(); return
        self.push_screen(
            DescribePrompt(
                scope=f"all {len(self.fragments)} fragments",
                default=self._last_description,
                history=self._description_history,
            ),
            self._on_describe_batch,
        )

    def _on_describe_batch(self, description: str | None) -> None:
        if not description:
            self._log("[dim]batch isolation cancelled[/dim]")
            return
        frags = list(self.fragments)
        pendings: list[_PendingSeparation] = []
        for i, frag in enumerate(frags):
            try:
                mono, sr, tag, out_base = self._prepare_audio(frag)
            except Exception as e:
                self._log(f"[red]prepare failed for fragment {i + 1}:[/red] {e}")
                continue
            pendings.append(_PendingSeparation(
                description=description, mono=mono, sr=sr,
                tag=tag, out_base=out_base, fragment=frag,
            ))
        if not pendings:
            self._log("[red]no fragments to process[/red]")
            return
        self._start_batch_separation(pendings)

    def _start_batch_separation(self, pendings: list[_PendingSeparation]) -> None:
        self._sep_busy = True
        self._sep_generation += 1
        generation = self._sep_generation
        desc = pendings[0].description
        self._last_description = desc
        # update history
        d = desc.strip()
        if d in self._description_history:
            self._description_history.remove(d)
        self._description_history.insert(0, d)
        self._description_history = self._description_history[:10]
        self._log(
            f"batch isolating [b]{desc!r}[/b] on {len(pendings)} fragments — "
            f"[dim]ctrl+k to cancel[/dim]"
        )
        threading.Thread(
            target=self._run_batch_separation,
            args=(pendings, generation),
            daemon=True,
        ).start()

    def _run_batch_separation(
        self, pendings: list[_PendingSeparation], generation: int
    ) -> None:
        from .separator import save_wav

        total = len(pendings)
        for idx, pending in enumerate(pendings):
            if generation != self._sep_generation:
                self.call_from_thread(self._finish_sep_cancelled, generation)
                return

            def progress_cb(done_s: float, total_s: float) -> None:
                if generation != self._sep_generation:
                    raise _CancelledSeparation
                self.call_from_thread(
                    self._update_batch_progress,
                    idx, total, done_s, total_s, generation,
                )

            self.call_from_thread(
                self._log,
                f"[dim]fragment {idx + 1}/{total}: {pending.tag}[/dim]",
            )
            try:
                target, residual, target_sr = self.separator.separate_arrays(
                    pending.mono, pending.sr, pending.description,
                    progress_cb=progress_cb,
                )
            except _CancelledSeparation:
                self.call_from_thread(self._finish_sep_cancelled, generation)
                return
            except Exception as e:
                self.call_from_thread(self._finish_sep_error, generation, str(e))
                return

            slug = _safe_slug(pending.description)
            out_base = pending.out_base.with_name(f"{pending.out_base.name}_{slug}")
            target_path = out_base.with_name(out_base.name + "_target.wav")
            residual_path = out_base.with_name(out_base.name + "_residual.wav")
            save_wav(target_path, target, target_sr)
            save_wav(residual_path, residual, target_sr)
            self.call_from_thread(
                self._log,
                f"[green]→[/green] {target_path.name} + {residual_path.name}",
            )

        self.call_from_thread(self._finish_batch_ok, generation, total)

    def _update_batch_progress(
        self, frag_idx: int, frag_total: int,
        done_s: float, total_s: float, generation: int,
    ) -> None:
        if generation != self._sep_generation:
            return
        frac = min(1.0, done_s / total_s) if total_s > 0 else 0.0
        filled = int(self.SEP_BAR_WIDTH * frac)
        bar = "█" * filled + "░" * (self.SEP_BAR_WIDTH - filled)
        pct = int(frac * 100)
        self._w_sep_progress.update(
            f"[b]batch[/b] {frag_idx + 1}/{frag_total}  "
            f"{bar} {pct:3d}%   "
            f"{done_s:5.1f}s / {total_s:5.1f}s   "
            f"[dim]ctrl+k to cancel[/dim]"
        )
        self._w_sep_progress.styles.display = "block"

    def _finish_batch_ok(self, generation: int, total: int) -> None:
        if generation != self._sep_generation:
            return
        self._sep_busy = False
        self._hide_sep_progress()
        self._log(f"[green]batch complete:[/green] {total} fragments isolated")

    def action_cancel_separation(self) -> None:
        if not self._sep_busy:
            self.bell(); return
        self._sep_generation += 1
        self._sep_busy = False
        self._hide_sep_progress()
        self._log("[yellow]cancelling isolation…[/yellow]")

    def action_open_file(self) -> None:
        if not _HAS_FSPICKER:
            self._log("install textual-fspicker: pip install textual-fspicker")
            self.bell(); return
        filters = Filters(
            ("Audio", lambda p: p.suffix.lower() in _AUDIO_EXTS),
            ("All", lambda p: True),
        )
        location = str(self.src.parent) if self.src is not None else str(Path.cwd())
        self.push_screen(
            FileOpen(location=location, filters=filters),
            self._on_file_chosen,
        )

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_quit_app(self) -> None:
        self.exit()

    # --- file switching ---

    def _on_file_chosen(self, path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            if self.src is not None:
                self._save_session()
            self._log(f"loading [b]{path.name}[/b] …")
            new_tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            ffmpeg.decode_to_wav(path, new_tmp)
            new_player = self._load_player(new_tmp)
            self.player.close()
            old_tmp = self.wav_path
            self.player = new_player
            self.src = path
            self.wav_path = new_tmp
            self._peaks = self._compute_peaks()
            if old_tmp is not None:
                try: old_tmp.unlink()
                except OSError: pass
            self.fragments.clear()
            self._undo_stack.clear()
            self.in_point = None
            self._play_until = None
            self._load_session()
            self._refresh_marks()
            self._refresh_list()
            self._w_status.update(self._header_text())
            self._log(
                f"opened {self.src.name} ({fmt_time(self.player.duration)})"
            )
        except Exception as e:
            self._log(f"[red]open failed:[/red] {e}")

    # --- fragment audition ---

    def action_play_fragment(self) -> None:
        self._play_selected_fragment()

    def on_list_view_selected(self, event: ListView.Selected) -> None:  # noqa: ARG002
        self._play_selected_fragment()

    def _play_selected_fragment(self) -> None:
        frag = self._selected_fragment()
        if frag is None:
            self.bell(); return
        self.player.seek_to(frag.start)
        self._play_until = frag.end
        self._playing_frag_idx = self.fragments.index(frag)
        if not self.player.playing:
            self.player.toggle()
        self._refresh_list()
        self._log(
            f"▶ fragment {fmt_time(frag.start)} → {fmt_time(frag.end)}"
        )

    # --- click-to-seek ---

    @on(events.Click, "#progress")
    @on(events.Click, "#waveform")
    def _seek_click(self, event: events.Click) -> None:
        width = event.widget.size.width if event.widget else 0
        dur = self.player.duration
        if width <= 0 or dur <= 0:
            return
        # strip padding: 2 chars on each side per CSS
        inner_x = event.x - 2
        inner_w = max(1, width - 4)
        frac = max(0.0, min(1.0, inner_x / inner_w))
        self.player.seek_to(frac * dur)
        self._play_until = None

    # --- save dialog result ---

    def _handle_save(self, request: Optional[SaveRequest]) -> None:
        if request is None:
            return
        out_ext = None if request.format == "same" else request.format
        try:
            if request.mode == "concat":
                ffmpeg.concat_cuts(
                    self.src, self.fragments, request.path, out_ext=out_ext,
                )
            else:
                ffmpeg.split_cuts(
                    self.src, self.fragments, request.path, out_ext=out_ext,
                )
            self._log(f"[green]saved →[/green] {request.path}")
        except Exception as e:
            self._log(f"[red]save failed:[/red] {e}")

    # --- separation flow ---

    def _on_describe_initial(
        self, description: Optional[str], frag: Optional[Fragment]
    ) -> None:
        if not description:
            self._log("[dim]isolation cancelled[/dim]")
            return
        try:
            mono, sr, tag, out_base = self._prepare_audio(frag)
        except Exception as e:
            self._log(f"[red]prepare failed:[/red] {e}")
            return
        pending = _PendingSeparation(
            description=description,
            mono=mono,
            sr=sr,
            tag=tag,
            out_base=out_base,
            fragment=frag,
        )
        self._start_separation(pending)

    def _prepare_audio(
        self, frag: Optional[Fragment]
    ) -> tuple[np.ndarray, int, str, Path]:
        data, sr = sf.read(str(self.wav_path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        if frag is not None:
            mono = slice_mono(mono, sr, frag.start, frag.end)
            tag = f"frag{self.fragments.index(frag) + 1:02d}"
        else:
            tag = "full"
        out_base = self.src.with_name(f"{self.src.stem}_{tag}")
        return mono.astype(np.float32), int(sr), tag, out_base

    def _start_separation(self, pending: _PendingSeparation) -> None:
        self._sep_busy = True
        self._sep_generation += 1
        generation = self._sep_generation
        self._last_description = pending.description
        # update history
        desc = pending.description.strip()
        if desc in self._description_history:
            self._description_history.remove(desc)
        self._description_history.insert(0, desc)
        self._description_history = self._description_history[:10]
        total_s = pending.mono.shape[0] / pending.sr
        self._log(
            f"isolating [b]{pending.description!r}[/b] on {pending.tag} "
            f"({fmt_time(total_s)}) — [dim]ctrl+k to cancel[/dim]"
        )
        self._update_sep_progress(0.0, total_s, generation)
        threading.Thread(
            target=self._run_separation,
            args=(pending, generation),
            daemon=True,
        ).start()

    def _run_separation(
        self, pending: _PendingSeparation, generation: int
    ) -> None:
        def progress_cb(done_s: float, total_s: float) -> None:
            if generation != self._sep_generation:
                raise _CancelledSeparation
            self.call_from_thread(
                self._update_sep_progress, done_s, total_s, generation
            )

        try:
            target, residual, target_sr = self.separator.separate_arrays(
                pending.mono,
                pending.sr,
                pending.description,
                progress_cb=progress_cb,
            )
        except _CancelledSeparation:
            self.call_from_thread(self._finish_sep_cancelled, generation)
            return
        except Exception as e:
            self.call_from_thread(self._finish_sep_error, generation, str(e))
            return
        self.call_from_thread(
            self._finish_sep_ok, generation, pending, target, residual, target_sr
        )

    def _update_sep_progress(
        self, done_s: float, total_s: float, generation: int
    ) -> None:
        if generation != self._sep_generation:
            return
        frac = min(1.0, done_s / total_s) if total_s > 0 else 0.0
        filled = int(self.SEP_BAR_WIDTH * frac)
        bar = "█" * filled + "░" * (self.SEP_BAR_WIDTH - filled)
        pct = int(frac * 100)
        self._w_sep_progress.update(
            f"[b]isolating[/b] {bar} {pct:3d}%   "
            f"{done_s:5.1f}s / {total_s:5.1f}s   "
            f"[dim]ctrl+k to cancel[/dim]"
        )
        self._w_sep_progress.styles.display = "block"

    def _hide_sep_progress(self) -> None:
        self._w_sep_progress.update("")
        self._w_sep_progress.styles.display = "none"

    def _finish_sep_cancelled(self, generation: int) -> None:
        if generation == self._sep_generation:
            self._hide_sep_progress()
        self._log("[yellow]isolation cancelled[/yellow]")

    def _finish_sep_error(self, generation: int, err: str) -> None:
        if generation != self._sep_generation:
            return  # cancelled — drop
        self._sep_busy = False
        self._hide_sep_progress()
        self._log(f"[red]isolation failed:[/red] {err}")

    def _finish_sep_ok(
        self,
        generation: int,
        pending: _PendingSeparation,
        target: np.ndarray,
        residual: np.ndarray,
        target_sr: int,
    ) -> None:
        if generation != self._sep_generation:
            self._log("[dim]cancelled result discarded[/dim]")
            return
        self._sep_busy = False
        self._hide_sep_progress()
        # pause main player so audio streams don't fight
        if self.player.playing:
            self.player.toggle()
        slug = _safe_slug(pending.description)
        out_base = pending.out_base.with_name(
            f"{pending.out_base.name}_{slug}"
        )
        data = SeparationResultData(
            description=pending.description,
            sample_rate=target_sr,
            target=target,
            residual=residual,
            out_base=out_base,
            original=pending.mono,
            original_sr=pending.sr,
        )
        self.push_screen(
            SeparationResultScreen(data),
            lambda decision: self._handle_sep_decision(decision, pending),
        )

    def _handle_sep_decision(
        self,
        decision: Optional[ResultDecision],
        pending: _PendingSeparation,
    ) -> None:
        if decision is None:
            self._log("[dim]result discarded[/dim]")
            return
        if decision.action in ("keep", "load") and decision.kept_paths is not None:
            t, r = decision.kept_paths
            self._log(f"[green]kept →[/green] {t.name} + {r.name}")
            if decision.action == "load":
                self._log(f"loading target [b]{t.name}[/b] …")
                self._on_file_chosen(t)
            return
        if decision.action == "rerun":
            self.push_screen(
                DescribePrompt(
                    scope=f"same audio ({pending.tag})",
                    default=pending.description,
                    history=self._description_history,
                ),
                lambda desc: self._on_describe_rerun(desc, pending),
            )

    def _on_describe_rerun(
        self,
        description: Optional[str],
        pending: _PendingSeparation,
    ) -> None:
        if not description:
            self._log("[dim]re-run cancelled[/dim]")
            return
        new_pending = _PendingSeparation(
            description=description,
            mono=pending.mono,
            sr=pending.sr,
            tag=pending.tag,
            out_base=pending.out_base,
            fragment=pending.fragment,
        )
        self._start_separation(new_pending)

    # --- session persistence ---

    def _load_session(self) -> None:
        if self.src is None:
            return
        sess = session.load(self.src)
        if sess.fragments:
            self.fragments = list(sess.fragments)
            self._refresh_list()
            self._refresh_marks()
            self._log(
                f"[dim]loaded {len(sess.fragments)} fragments "
                f"from {session.sidecar_path(self.src).name}[/dim]"
            )
        if sess.last_description:
            self._last_description = sess.last_description
        if sess.description_history:
            self._description_history = list(sess.description_history)

    def _save_session(self) -> None:
        if self.src is None:
            return
        sess = session.Session(
            fragments=list(self.fragments),
            last_description=self._last_description,
            description_history=list(self._description_history),
        )
        try:
            session.save(self.src, sess)
        except OSError:
            pass
