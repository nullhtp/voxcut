"""Isolate a single voice from audio using MLX port of SAM-Audio (Apple Silicon).

Run inside the qwen3-tts conda env:
    conda activate qwen3-tts
    python separate.py alex.mp3 "man speaking"
"""
import sys
from mlx_audio.sts.models.sam_audio import SAMAudio, save_audio

REPO = "mlx-community/sam-audio-large-fp16"


def main(audio_path: str, description: str, out_prefix: str = "alex") -> None:
    print(f"Loading {REPO} ...")
    model = SAMAudio.from_pretrained(REPO)
    model.eval()

    print(f"Separating: {description!r} from {audio_path}")
    result = model.separate_long(
        audios=[audio_path],
        descriptions=[description],
        chunk_seconds=10.0,
        overlap_seconds=3.0,
        verbose=True,
    )

    sr = model.sample_rate
    save_audio(result.target[0], f"{out_prefix}_target.wav", sample_rate=sr)
    save_audio(result.residual[0], f"{out_prefix}_residual.wav", sample_rate=sr)
    print(f"Wrote {out_prefix}_target.wav  (isolated: {description!r})")
    print(f"Wrote {out_prefix}_residual.wav (everything else)")


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "alex.mp3"
    desc = sys.argv[2] if len(sys.argv) > 2 else "man speaking"
    main(audio, desc)
