"""Thin wrappers around ``ffmpeg`` for decoding and cutting audio."""
from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from .fragment import Fragment

_BASE = ["ffmpeg", "-y", "-v", "error"]


def _run(cmd: list[str], *, check: bool = True) -> int:
    return subprocess.run(cmd, check=check).returncode


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any ffmpeg-supported input to float32 WAV for random access."""
    _run([*_BASE, "-i", str(src), "-f", "wav", "-acodec", "pcm_f32le", str(dst)])


def cut(src: Path, frag: Fragment, dst: Path) -> None:
    """Cut ``frag`` from ``src`` to ``dst``. Stream-copy first, re-encode on failure."""
    slice_args = [
        *_BASE,
        "-ss", f"{frag.start:.3f}",
        "-to", f"{frag.end:.3f}",
        "-i", str(src),
    ]
    copy_ok = (
        _run([*slice_args, "-c", "copy", str(dst)], check=False) == 0
        and dst.exists()
        and dst.stat().st_size > 0
    )
    if not copy_ok:
        _run([*slice_args, str(dst)])


def concat_cuts(src: Path, frags: Iterable[Fragment], dst: Path) -> None:
    """Cut each fragment from ``src`` and concatenate into a single ``dst`` file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = dst.suffix or src.suffix
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        parts: list[Path] = []
        for i, frag in enumerate(frags):
            part = td_path / f"p{i:04d}{ext}"
            cut(src, frag, part)
            parts.append(part)
        listfile = td_path / "list.txt"
        listfile.write_text("".join(f"file '{p}'\n" for p in parts))
        concat_args = [*_BASE, "-f", "concat", "-safe", "0", "-i", str(listfile)]
        if _run([*concat_args, "-c", "copy", str(dst)], check=False) != 0:
            _run([*concat_args, str(dst)])


def split_cuts(src: Path, frags: Iterable[Fragment], out_dir: Path) -> None:
    """Write each fragment of ``src`` as its own file inside ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix
    for i, frag in enumerate(frags):
        cut(src, frag, out_dir / f"{src.stem}_{i + 1:02d}{ext}")
