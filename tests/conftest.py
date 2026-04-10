"""Shared test fixtures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def wav_10s(tmp_path: Path) -> Path:
    """Create a 10-second 16kHz mono sine wave."""
    sr = 16000
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), data, sr)
    return path


@pytest.fixture
def wav_2s(tmp_path: Path) -> Path:
    """Create a 2-second 16kHz mono sine wave."""
    sr = 16000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "short.wav"
    sf.write(str(path), data, sr)
    return path
