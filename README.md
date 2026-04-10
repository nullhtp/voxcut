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
uv run sam3-audio alex.mp3    # with a file
uv run sam3-audio             # no arg — picker opens on startup
```

### TUI controls

Press `?` any time for the in-app help.

| Key              | Action                                                    |
| ---------------- | --------------------------------------------------------- |
| `space`          | play / pause                                              |
| `← / →`          | seek ±5s                                                  |
| `shift+← / →`    | seek ±1s                                                  |
| `- / +`          | speed 0.5× .. 2.0×                                        |
| `v` / `shift+v`  | volume +3dB / -3dB                                        |
| click bar        | seek to clicked position (progress bar or waveform)       |
| `i` / `o`        | mark in / out point (adds fragment)                       |
| `enter`          | play selected fragment (auto-stops at out-point)          |
| `[` / `]`        | nudge selected fragment in/out ±0.1s (expand)             |
| `{` / `}`        | nudge selected fragment in/out ±0.1s (contract)           |
| `x` / `u`        | delete selected fragment / undo last delete               |
| `s`              | save fragments (modal: mode + format + path)              |
| `d`              | isolate voice on selection or whole file                  |
| `shift+d`        | batch isolate all fragments (one description)             |
| `ctrl+k`         | cancel in-flight isolation (result is dropped)            |
| `f`              | open another file                                         |
| `?`              | help overlay                                              |
| `q`              | quit                                                      |

### Separation result screen

After `d` finishes you land in an audition modal:

| Key          | Action                                    |
| ------------ | ----------------------------------------- |
| `t` / `r` / `o` | play target / residual / original     |
| `space`      | stop playback                             |
| `k`          | keep (write WAVs to disk)                 |
| `l`          | keep + load target into main TUI          |
| `shift+r`    | re-run with a different description       |
| `esc`        | discard                                   |

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
