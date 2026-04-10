# sam3-audio

Audio editing TUI for Apple Silicon: cut fragments, isolate voices, and scrub
audio — powered by [SAM-Audio (MLX)](https://huggingface.co/mlx-community/sam-audio-large-fp16).

![TUI](https://img.shields.io/badge/interface-TUI-blue)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-only-black)
![Python](https://img.shields.io/badge/python-3.11+-green)

## Features

- **Playback** — play, seek, speed control (0.5-2x), volume gain (±dB),
  click-to-seek on progress bar and waveform
- **Fragment editing** — mark in/out, nudge boundaries with audio preview,
  split at cursor, merge adjacent, undo delete
- **Visual waveform** — responsive peak strip with fragment overlay (green),
  pending in-point (yellow), ghost selection region, and playback cursor
- **Voice isolation** — describe a voice in plain text, SAM-Audio isolates it.
  Audition target/residual/original A/B, re-run, keep, or load result back
  into the TUI for further editing
- **Batch isolation** — isolate the same voice across all fragments at once
- **Session persistence** — fragments and description history auto-save to a
  sidecar JSON file and reload on next open
- **Export** — save as concatenated file or separate files, with format
  conversion (WAV/MP3/FLAC/OGG)
- **Context-aware UI** — status bar shows selected fragment details and cursor
  position; footer shows only essential keys; full reference via `?`

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`.

```bash
uv sync
```

## Usage

### CLI: isolate a voice

```bash
uv run sam3-separate alex.mp3 "man speaking"
```

Writes `alex_target.wav` (isolated) and `alex_residual.wav` (everything else).

### TUI: interactive editor

```bash
uv run sam3-audio recording.mp3    # open a file
uv run sam3-audio                  # no arg — file picker on startup
```

## Keybindings

Press `?` inside the TUI for the full reference. Summary:

### Playback

| Key              | Action                                  |
| ---------------- | --------------------------------------- |
| `space`          | play / pause                            |
| `← / →`          | seek ±5s                                |
| `shift+← / →`    | seek ±1s                                |
| `- / +`          | speed 0.5× .. 2.0×                      |
| `v / V`          | volume +3dB / -3dB                      |
| click            | seek on progress bar or waveform        |

### Fragments

| Key              | Action                                  |
| ---------------- | --------------------------------------- |
| `i / o`          | mark in / out point                     |
| `p` / `enter`    | play selected fragment                  |
| `g / G`          | seek to fragment start / end            |
| `[ / ]`          | nudge in/out ±0.1s (expand) with preview |
| `{ / }`          | nudge in/out ±0.1s (contract) with preview |
| `S`              | split fragment at cursor                |
| `m`              | merge with next fragment                |
| `x / u`          | delete / undo delete                    |
| `s`              | save (mode + format + path dialog)      |

### Voice isolation

| Key              | Action                                  |
| ---------------- | --------------------------------------- |
| `d`              | isolate voice (fragment or whole file)  |
| `D`              | batch isolate all fragments             |
| `ctrl+k`         | cancel in-flight isolation              |

### Separation result screen

| Key              | Action                                  |
| ---------------- | --------------------------------------- |
| `t / r / o`      | play target / residual / original       |
| `space`          | stop                                    |
| `k`              | keep (write to disk)                    |
| `l`              | keep + load target into TUI             |
| `R`              | re-run with different description       |
| `esc`            | discard                                 |

### General

| Key              | Action                                  |
| ---------------- | --------------------------------------- |
| `f`              | open another file                       |
| `?`              | help overlay                            |
| `q`              | quit (confirms if unsaved)              |

## Session persistence

Fragments, last description, and description history auto-save to
`<input>.sam3.json` next to the audio file. Reloaded automatically on next
open. The sidecar is gitignored by default.

## Project layout

```
sam3_audio/
  cli.py              entry points (sam3-audio, sam3-separate)
  tui.py              Textual application
  player.py           numpy-backed audio player with gain
  fragment.py          Fragment value object
  ffmpeg.py           decode / cut / concat / split via ffmpeg
  waveform.py         responsive peak-strip with overlays
  separator.py        SAM-Audio (MLX) service
  session.py          sidecar JSON persistence
  timeutil.py         time formatting
  screens/
    help.py           keybinding reference overlay
    describe_prompt.py  voice description input with history
    save_dialog.py    save mode + format + path
    separation_result.py  audition target/residual/original
    confirm.py        yes/no confirmation dialog
```

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.11+
- `ffmpeg` on PATH
- [uv](https://docs.astral.sh/uv/) for dependency management
