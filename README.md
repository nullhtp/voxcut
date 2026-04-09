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

Press `?` any time for the in-app help.

| Key              | Action                                                    |
| ---------------- | --------------------------------------------------------- |
| `space`          | play / pause                                              |
| `← / →`          | seek ±5s                                                  |
| `shift+← / →`    | seek ±1s                                                  |
| `- / +`          | speed 0.5× .. 2.0×                                        |
| click bar        | seek to clicked position (progress bar or waveform)       |
| `i` / `o`        | mark in / out point (adds fragment)                       |
| `enter`          | play selected fragment (auto-stops at out-point)          |
| `x` / `u`        | delete selected fragment / undo last delete               |
| `s`              | save fragments (modal: concat or separate files)          |
| `d`              | isolate voice on selection or whole file                  |
| `ctrl+k`         | cancel in-flight isolation (result is dropped)            |
| `f`              | open another file                                         |
| `?`              | help overlay                                              |
| `q`              | quit                                                      |

### Separation result screen

After `d` finishes you land in an audition modal: `t` plays the isolated
target, `r` plays the residual, `space` stops, `k` keeps (writes WAVs to disk),
`shift+r` re-runs with a different description, `esc` discards.

### Session persistence

Fragments and the last-used description auto-save to `<input>.sam3.json`
next to the audio file and reload the next time you open it.

## Layout

```
sam3_audio/
  timeutil.py    time formatting
  fragment.py    Fragment value object
  ffmpeg.py      decode / cut / concat / split helpers
  player.py      numpy-backed audio player
  waveform.py    peak-strip waveform rendering
  separator.py   SAM-Audio (MLX) service
  session.py     sidecar JSON persistence
  tui.py         Textual application
  cli.py         entry points
  screens/       modal dialogs: help, describe, save, result
```
