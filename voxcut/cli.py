"""Command-line entry points."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import ffmpeg
from .separator import SAM_REPO, SamSeparator, load_mono
from .tui import AudioTUI


def _require(condition: bool, msg: str) -> None:
    if not condition:
        sys.exit(msg)


def _require_ffmpeg() -> None:
    _require(shutil.which("ffmpeg") is not None, "ffmpeg not found on PATH")


def _cleanup(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass


def tui_main() -> None:
    ap = argparse.ArgumentParser(
        prog="voxcut",
        description="Play audio, cut fragments, and isolate voices with SAM-Audio.",
    )
    ap.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="input audio file (optional — picker opens if omitted)",
    )
    args = ap.parse_args()

    if args.input is not None:
        _require(args.input.exists(), f"no such file: {args.input}")
    _require_ffmpeg()

    # Pre-load SAM-Audio *before* Textual patches stdio: huggingface_hub's
    # snapshot_download spawns subprocesses that otherwise crash with
    # "bad value(s) in fds_to_keep".
    separator = SamSeparator()
    try:
        print(f"Loading {SAM_REPO} (first run downloads weights) ...")
        separator.load()
        print("SAM-Audio ready.")
    except KeyboardInterrupt:
        sys.exit(130)
    except (ImportError, OSError, RuntimeError) as e:
        print(f"(SAM-Audio preload failed: {e} — 'd' key will be disabled)")

    tmp: Path | None = None
    try:
        if args.input is not None:
            tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
            ffmpeg.decode_to_wav(args.input, tmp)
        AudioTUI(args.input, tmp, separator).run()
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed: {e}")
    finally:
        _cleanup(tmp)


def separate_main() -> None:
    ap = argparse.ArgumentParser(
        prog="voxcut-separate",
        description="Isolate a described voice from an audio file using SAM-Audio.",
    )
    ap.add_argument("input", type=Path)
    ap.add_argument("description", nargs="?", default="man speaking")
    ap.add_argument(
        "-o",
        "--out-prefix",
        type=Path,
        default=None,
        help="output basename without suffix (default: <input stem>)",
    )
    args = ap.parse_args()

    _require(args.input.exists(), f"no such file: {args.input}")
    _require_ffmpeg()

    separator = SamSeparator()
    try:
        print(f"Loading {SAM_REPO} ...")
        separator.load()
    except KeyboardInterrupt:
        sys.exit(130)

    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
        ffmpeg.decode_to_wav(args.input, tmp)
        mono, sr = load_mono(tmp)
        out_base = args.out_prefix or args.input.with_suffix("")
        print(f"Separating: {args.description!r} from {args.input}")
        result = separator.separate(mono, sr, args.description, out_base)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed: {e}")
    finally:
        _cleanup(tmp)

    print(f"Wrote {result.target_path.name}  (isolated: {args.description!r})")
    print(f"Wrote {result.residual_path.name} (everything else)")
