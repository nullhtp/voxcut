"""Numpy-backed audio player with seek and speed control."""
from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd


class Player:
    """Plays a numpy buffer via sounddevice. Speed resamples the output stream
    (and therefore shifts pitch — fine for scrubbing)."""

    MIN_SPEED = 0.5
    MAX_SPEED = 2.0
    MIN_GAIN_DB = -24.0
    MAX_GAIN_DB = 24.0

    def __init__(self, data: np.ndarray, samplerate: int):
        if data.ndim == 1:
            data = data[:, None]
        self.data = data.astype(np.float32, copy=False)
        self.sr = int(samplerate)
        self.channels = self.data.shape[1]
        self.frame = 0
        self.speed = 1.0
        self.gain_db = 0.0
        self._gain_linear = 1.0
        self.playing = False
        self._lock = threading.Lock()
        self.stream: sd.OutputStream | None = None
        self._open_stream()

    # --- stream lifecycle ---

    def _open_stream(self) -> None:
        self._close_stream()
        self.stream = sd.OutputStream(
            samplerate=int(self.sr * self.speed),
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )

    def _close_stream(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            if not self.playing:
                outdata[:] = 0
                return
            end = self.frame + frames
            chunk = self.data[self.frame:end]
            n = chunk.shape[0]
            outdata[:n] = chunk * self._gain_linear
            if n < frames:
                outdata[n:] = 0
                self.playing = False
            self.frame = min(end, self.data.shape[0])

    # --- state ---

    @property
    def duration(self) -> float:
        return self.data.shape[0] / self.sr

    @property
    def position(self) -> float:
        return self.frame / self.sr

    def toggle(self) -> None:
        with self._lock:
            self.playing = not self.playing
        (self.stream.start if self.playing else self.stream.stop)()

    def seek(self, delta: float) -> None:
        with self._lock:
            self.frame = max(
                0,
                min(self.data.shape[0] - 1, self.frame + int(delta * self.sr)),
            )

    def seek_to(self, t: float) -> None:
        with self._lock:
            self.frame = max(
                0, min(self.data.shape[0] - 1, int(t * self.sr))
            )

    def set_speed(self, speed: float) -> None:
        speed = max(self.MIN_SPEED, min(self.MAX_SPEED, speed))
        was_playing = self.playing
        with self._lock:
            self.playing = False
        self._close_stream()
        self.speed = speed
        self._open_stream()
        if was_playing:
            with self._lock:
                self.playing = True
            self.stream.start()

    def set_gain(self, db: float) -> None:
        db = max(self.MIN_GAIN_DB, min(self.MAX_GAIN_DB, db))
        self.gain_db = db
        self._gain_linear = 10.0 ** (db / 20.0)

    def close(self) -> None:
        with self._lock:
            self.playing = False
        self._close_stream()
