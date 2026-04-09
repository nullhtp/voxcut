"""
audiocut — TUI to keep only selected fragments of an audio file.

Controls:
  space       play / pause
  ← / →       seek ±5s
  shift+← / → seek ±1s
  - / +       speed 0.5x .. 2.0x
  i           mark in-point
  o           mark out-point (adds fragment)
  x           delete selected fragment from list
  s           save (prompts for mode + path)
  q           quit

Usage:
  python audiocut.py <input-audio>

Deps:
  pip install textual soundfile sounddevice numpy
  ffmpeg on PATH
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static


def fmt_time(t: float) -> str:
    if t < 0 or t != t:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    if h:
        return f"{h:d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any ffmpeg-supported input to float32 wav for fast random access."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-f", "wav", "-acodec", "pcm_f32le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


@dataclass
class Fragment:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

    def label(self, idx: int) -> str:
        return f"{idx+1:>2}. {fmt_time(self.start)} → {fmt_time(self.end)}  ({fmt_time(self.dur)})"


class Player:
    """Numpy-backed player with seek + speed (speed via samplerate, pitch shifts)."""

    def __init__(self, data: np.ndarray, samplerate: int):
        if data.ndim == 1:
            data = data[:, None]
        self.data = data.astype(np.float32, copy=False)
        self.sr = samplerate
        self.channels = data.shape[1]
        self.frame = 0
        self.speed = 1.0
        self.playing = False
        self.stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._open_stream()

    def _open_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
        self.stream = sd.OutputStream(
            samplerate=self.sr * self.speed,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if not self.playing:
                outdata[:] = 0
                return
            end = self.frame + frames
            chunk = self.data[self.frame:end]
            n = chunk.shape[0]
            outdata[:n] = chunk
            if n < frames:
                outdata[n:] = 0
                self.playing = False
            self.frame = min(end, self.data.shape[0])

    @property
    def duration(self) -> float:
        return self.data.shape[0] / self.sr

    @property
    def position(self) -> float:
        return self.frame / self.sr

    def toggle(self):
        with self._lock:
            self.playing = not self.playing
        if self.playing:
            self.stream.start()
        else:
            self.stream.stop()

    def seek(self, delta: float):
        with self._lock:
            target = max(0, min(self.data.shape[0] - 1, self.frame + int(delta * self.sr)))
            self.frame = target

    def seek_to(self, t: float):
        with self._lock:
            self.frame = max(0, min(self.data.shape[0] - 1, int(t * self.sr)))

    def set_speed(self, speed: float):
        speed = max(0.5, min(2.0, speed))
        was_playing = self.playing
        with self._lock:
            self.playing = False
        if self.stream:
            self.stream.stop(); self.stream.close()
        self.speed = speed
        self._open_stream()
        if was_playing:
            with self._lock:
                self.playing = True
            self.stream.start()

    def close(self):
        with self._lock:
            self.playing = False
        if self.stream:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass


class AudioCutApp(App):
    CSS = """
    Screen { layout: vertical; }
    #status { height: 3; padding: 1 2; background: $boost; }
    #progress { height: 1; padding: 0 2; color: $accent; }
    #marks { height: 3; padding: 0 2; }
    #list { height: 1fr; border: tall $primary; }
    #prompt { height: 3; border: tall $warning; display: none; }
    .hidden { display: none; }
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
        Binding("s", "save", "save"),
        Binding("q", "quit", "quit"),
    ]

    position = reactive(0.0)
    in_point: Optional[float] = None

    def __init__(self, src: Path, wav: Path):
        super().__init__()
        self.src = src
        self.wav_path = wav
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        self.player = Player(data, sr)
        self.fragments: list[Fragment] = []
        self._prompt_mode: Optional[str] = None  # "save"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"File: {self.src.name}   duration: {fmt_time(self.player.duration)}", id="status")
        yield Static("", id="progress")
        yield Static("", id="marks")
        yield ListView(id="list")
        yield Input(placeholder="", id="prompt")
        yield Footer()

    def on_mount(self):
        self.set_interval(0.1, self._tick)
        self._refresh_marks()
        self._refresh_list()

    def _tick(self):
        self.position = self.player.position
        p = self.player.position
        d = self.player.duration
        bar_w = 40
        filled = int(bar_w * (p / d)) if d else 0
        bar = "█" * filled + "░" * (bar_w - filled)
        state = "▶" if self.player.playing else "⏸"
        self.query_one("#progress", Static).update(
            f"{state} {bar} {fmt_time(p)} / {fmt_time(d)}   speed {self.player.speed:.1f}x"
        )

    def _refresh_marks(self):
        ip = fmt_time(self.in_point) if self.in_point is not None else "—"
        self.query_one("#marks", Static).update(f"in: {ip}    (press i to mark in, o to mark out)")

    def _refresh_list(self):
        lv = self.query_one("#list", ListView)
        lv.clear()
        for i, f in enumerate(self.fragments):
            lv.append(ListItem(Label(f.label(i))))

    # --- actions ---
    def action_toggle_play(self):
        self.player.toggle()

    def action_seek(self, delta: float):
        self.player.seek(delta)

    def action_speed(self, delta: float):
        self.player.set_speed(round(self.player.speed + delta, 2))

    def action_mark_in(self):
        self.in_point = self.player.position
        self._refresh_marks()

    def action_mark_out(self):
        if self.in_point is None:
            self.bell(); return
        start = self.in_point
        end = self.player.position
        if end <= start:
            self.bell(); return
        self.fragments.append(Fragment(start, end))
        self.fragments.sort(key=lambda f: f.start)
        self.in_point = None
        self._refresh_marks()
        self._refresh_list()

    def action_del_fragment(self):
        lv = self.query_one("#list", ListView)
        idx = lv.index
        if idx is None or not (0 <= idx < len(self.fragments)):
            self.bell(); return
        del self.fragments[idx]
        self._refresh_list()

    def action_save(self):
        if not self.fragments:
            self.bell(); return
        self._prompt_mode = "save_mode"
        inp = self.query_one("#prompt", Input)
        inp.placeholder = "save mode: (c)oncat single file  /  (s)eparate files"
        inp.value = ""
        inp.styles.display = "block"
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted):
        val = event.value.strip()
        inp = self.query_one("#prompt", Input)
        if self._prompt_mode == "save_mode":
            if val.lower().startswith("c"):
                self._save_mode = "concat"
            elif val.lower().startswith("s"):
                self._save_mode = "separate"
            else:
                self.bell(); return
            self._prompt_mode = "save_path"
            inp.value = ""
            default = str(self.src.with_name(self.src.stem + "_cut" + self.src.suffix))
            if self._save_mode == "separate":
                default = str(self.src.with_name(self.src.stem + "_cut"))  # dir
            inp.placeholder = f"output path (default: {default}) — enter to accept"
            self._default_out = default
            inp.focus()
            return
        if self._prompt_mode == "save_path":
            out = val or self._default_out
            inp.styles.display = "none"
            self._prompt_mode = None
            try:
                self._do_save(self._save_mode, Path(out))
                self.query_one("#status", Static).update(f"saved → {out}")
            except Exception as e:
                self.query_one("#status", Static).update(f"save failed: {e}")

    def _do_save(self, mode: str, out: Path):
        src = str(self.src)
        if mode == "concat":
            with tempfile.TemporaryDirectory() as td:
                parts = []
                for i, f in enumerate(self.fragments):
                    p = Path(td) / f"p{i:04d}{out.suffix or self.src.suffix}"
                    subprocess.run([
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", f"{f.start:.3f}", "-to", f"{f.end:.3f}",
                        "-i", src, "-c", "copy", str(p)
                    ], check=False)
                    if not p.exists() or p.stat().st_size == 0:
                        # fallback: re-encode
                        subprocess.run([
                            "ffmpeg", "-y", "-v", "error",
                            "-ss", f"{f.start:.3f}", "-to", f"{f.end:.3f}",
                            "-i", src, str(p)
                        ], check=True)
                    parts.append(p)
                listfile = Path(td) / "list.txt"
                listfile.write_text("".join(f"file '{p}'\n" for p in parts))
                out.parent.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([
                    "ffmpeg", "-y", "-v", "error",
                    "-f", "concat", "-safe", "0", "-i", str(listfile),
                    "-c", "copy", str(out)
                ])
                if r.returncode != 0:
                    subprocess.run([
                        "ffmpeg", "-y", "-v", "error",
                        "-f", "concat", "-safe", "0", "-i", str(listfile),
                        str(out)
                    ], check=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            ext = self.src.suffix
            for i, f in enumerate(self.fragments):
                p = out / f"{self.src.stem}_{i+1:02d}{ext}"
                r = subprocess.run([
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", f"{f.start:.3f}", "-to", f"{f.end:.3f}",
                    "-i", src, "-c", "copy", str(p)
                ])
                if r.returncode != 0 or not p.exists() or p.stat().st_size == 0:
                    subprocess.run([
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", f"{f.start:.3f}", "-to", f"{f.end:.3f}",
                        "-i", src, str(p)
                    ], check=True)

    def on_unmount(self):
        self.player.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    args = ap.parse_args()
    if not args.input.exists():
        print(f"no such file: {args.input}", file=sys.stderr); sys.exit(1)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr); sys.exit(1)

    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        decode_to_wav(args.input, tmp)
        AudioCutApp(args.input, tmp).run()
    finally:
        try: tmp.unlink()
        except Exception: pass


if __name__ == "__main__":
    main()
