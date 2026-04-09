"""SAM-Audio (MLX) voice separation service."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

ProgressCallback = Callable[[float, float], None]
"""``(done_seconds, total_seconds)`` — may raise to abort separation."""

SAM_REPO = "mlx-community/sam-audio-large-fp16"


@dataclass(frozen=True)
class SeparationResult:
    target_path: Path
    residual_path: Path


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file and return (mono float32 samples, samplerate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return mono.astype(np.float32), int(sr)


def slice_mono(mono: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(start * sr))
    b = min(mono.shape[0], int(end * sr))
    return mono[a:b]


def _linear_resample(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return mono
    n_out = int(round(mono.shape[0] * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, mono.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def _save_wav(path: Path, mono: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sr, subtype="FLOAT")


class SamSeparator:
    """Lazy wrapper around the MLX SAM-Audio model."""

    def __init__(self, repo: str = SAM_REPO) -> None:
        self.repo = repo
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        # Imported lazily — MLX is a heavy, platform-specific dependency.
        from mlx_audio.sts.models.sam_audio import SAMAudio
        from mlx_audio.sts.models.sam_audio.processor import SAMAudioProcessor

        model = SAMAudio.from_pretrained(self.repo)
        if getattr(model, "processor", None) is None:
            model.processor = SAMAudioProcessor.from_pretrained(self.repo)
        model.eval()
        self._model = model

    @property
    def sample_rate(self) -> int:
        self.load()
        return int(self._model.sample_rate)

    def separate_arrays(
        self,
        mono: np.ndarray,
        sr: int,
        description: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Run separation and return (target, residual, sample_rate) as numpy mono.

        Uses ``separate_streaming`` under the hood so progress can be reported
        per chunk. ``progress_cb`` is invoked as ``(done_seconds, total_seconds)``
        and may raise to abort the run — the exception propagates out unchanged
        so callers can use a sentinel for cancellation.
        """
        import mlx.core as mx

        self.load()
        target_sr = self.sample_rate
        mono = _linear_resample(mono.astype(np.float32), sr, target_sr)
        audio_arr = mx.array(mono)[None, None, :]

        total_samples = audio_arr.shape[2]
        total_seconds = total_samples / target_sr

        target_pieces: list[np.ndarray] = []
        residual_pieces: list[np.ndarray] = []
        samples_done = 0

        if progress_cb is not None:
            progress_cb(0.0, total_seconds)

        for chunk in self._model.separate_streaming(
            audios=audio_arr,
            descriptions=[description],
            chunk_seconds=10.0,
            overlap_seconds=3.0,
            verbose=False,
        ):
            tgt = np.array(chunk.target, copy=False).astype(np.float32).reshape(-1)
            res = np.array(chunk.residual, copy=False).astype(np.float32).reshape(-1)
            target_pieces.append(tgt)
            residual_pieces.append(res)
            samples_done += tgt.shape[0]
            if progress_cb is not None:
                progress_cb(samples_done / target_sr, total_seconds)

        target = (
            np.concatenate(target_pieces)
            if target_pieces
            else np.zeros(0, dtype=np.float32)
        )
        residual = (
            np.concatenate(residual_pieces)
            if residual_pieces
            else np.zeros(0, dtype=np.float32)
        )
        return target, residual, target_sr

    def separate(
        self,
        mono: np.ndarray,
        sr: int,
        description: str,
        out_base: Path,
    ) -> SeparationResult:
        """Run separation and write ``<out_base>_target.wav`` / ``_residual.wav``."""
        target, residual, target_sr = self.separate_arrays(mono, sr, description)
        target_path = out_base.with_name(out_base.name + "_target.wav")
        residual_path = out_base.with_name(out_base.name + "_residual.wav")
        _save_wav(target_path, target, target_sr)
        _save_wav(residual_path, residual, target_sr)
        return SeparationResult(target_path, residual_path)


def save_wav(path: Path, mono: np.ndarray, sr: int) -> None:
    """Public helper: write mono float32 data as a float WAV."""
    _save_wav(path, mono, sr)
