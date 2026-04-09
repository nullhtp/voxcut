# sam3-audio

Voice isolation and audio fragment cutting on Apple Silicon, powered by the
[MLX port of SAM-Audio](https://huggingface.co/mlx-community/sam-audio-large-fp16).

Two entry points:

- `audio_tui.py` — interactive TUI to scrub audio, cut fragments, and isolate
  voices with a text description.
- `separate.py` — minimal CLI that isolates one voice from an audio file.

> Apple Silicon only. The upstream `facebookresearch/sam-audio` does not run on
> Mac (xformers); this project uses the MLX community port instead.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`.

```bash
uv sync
```

## Usage

Isolate a voice from an audio file:

```bash
uv run python separate.py alex.mp3 "man speaking"
```

Outputs `<prefix>_target.wav` (isolated voice) and `<prefix>_residual.wav`
(everything else).

Launch the TUI:

```bash
uv run python audio_tui.py alex.mp3
```

### TUI controls

| Key            | Action                                           |
| -------------- | ------------------------------------------------ |
| `space`        | play / pause                                     |
| `← / →`        | seek ±5s                                         |
| `shift+← / →`  | seek ±1s                                         |
| `- / +`        | speed 0.5x .. 2.0x                               |
| `i` / `o`      | mark in / out point (adds fragment)              |
| `x`            | delete selected fragment                         |
| `s`            | save fragments                                   |
| `d`            | isolate voice on selection or whole file         |
| `q`            | quit                                             |

## Files

- `separate.py` — SAM-Audio voice isolation CLI.
- `audiocut.py` — standalone fragment-cutting TUI.
- `audio_tui.py` — combined cut + isolate TUI.
