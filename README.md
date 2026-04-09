# sam3-audio

Voice isolation and audio fragment cutting on Apple Silicon, powered by the
[MLX port of SAM-Audio](https://huggingface.co/mlx-community/sam-audio-large-fp16).

Two entry points:

- `sam3-audio` — interactive TUI: scrub audio, cut fragments, isolate voices.
- `sam3-separate` — minimal CLI that isolates one voice from an audio file.

> Apple Silicon only. Upstream `facebookresearch/sam-audio` does not run on Mac
> (xformers); this project uses the MLX community port instead.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`.

```bash
uv sync
```

## Usage

Isolate a voice from an audio file:

```bash
uv run sam3-separate alex.mp3 "man speaking"
```

Writes `<stem>_target.wav` (isolated voice) and `<stem>_residual.wav`
(everything else).

Launch the TUI:

```bash
uv run sam3-audio alex.mp3
```

### TUI controls

| Key            | Action                                           |
| -------------- | ------------------------------------------------ |
| `space`        | play / pause                                     |
| `← / →`        | seek ±5s                                         |
| `shift+← / →`  | seek ±1s                                         |
| `- / +`        | speed 0.5× .. 2.0×                               |
| `i` / `o`      | mark in / out point (adds fragment)              |
| `x`            | delete selected fragment                         |
| `s`            | save fragments (concat or separate files)       |
| `d`            | isolate voice on selection or whole file        |
| `f`            | open another file                                |
| `q`            | quit                                             |

## Layout

```
sam3_audio/
  timeutil.py    time formatting
  fragment.py    Fragment value object
  ffmpeg.py      decode / cut / concat / split helpers
  player.py      numpy-backed audio player
  separator.py   SAM-Audio (MLX) service
  tui.py         Textual application
  cli.py         entry points
```
