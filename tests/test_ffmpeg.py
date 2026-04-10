"""Tests for ffmpeg module (requires ffmpeg on PATH)."""
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from voxcut.ffmpeg import concat_cuts, cut, decode_to_wav, split_cuts
from voxcut.fragment import Fragment

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def test_decode_to_wav(wav_2s: Path, tmp_path: Path):
    dst = tmp_path / "decoded.wav"
    decode_to_wav(wav_2s, dst)
    assert dst.exists()
    data, sr = sf.read(str(dst))
    assert sr == 16000
    assert data.shape[0] > 0


def test_cut(wav_10s: Path, tmp_path: Path):
    dst = tmp_path / "cut.wav"
    cut(wav_10s, Fragment(1.0, 3.0), dst)
    assert dst.exists()
    data, sr = sf.read(str(dst))
    duration = data.shape[0] / sr
    assert 1.5 < duration < 2.5


def test_concat_cuts(wav_10s: Path, tmp_path: Path):
    dst = tmp_path / "concat.wav"
    frags = [Fragment(0.0, 1.0), Fragment(5.0, 6.0)]
    concat_cuts(wav_10s, frags, dst)
    assert dst.exists()
    data, sr = sf.read(str(dst))
    duration = data.shape[0] / sr
    assert 1.5 < duration < 2.5


def test_split_cuts(wav_10s: Path, tmp_path: Path):
    out_dir = tmp_path / "splits"
    frags = [Fragment(0.0, 2.0), Fragment(5.0, 7.0)]
    split_cuts(wav_10s, frags, out_dir)
    files = sorted(out_dir.iterdir())
    assert len(files) == 2


def test_cut_with_format_conversion(wav_10s: Path, tmp_path: Path):
    dst = tmp_path / "cut.mp3"
    cut(wav_10s, Fragment(1.0, 3.0), dst, reencode=True)
    assert dst.exists()
    assert dst.stat().st_size > 0
